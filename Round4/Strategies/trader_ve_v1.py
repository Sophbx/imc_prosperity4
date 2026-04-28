"""
VE-only trader, R4 v1.

Trades ONLY VELVETFRUIT_EXTRACT. No HG, no vouchers.

----------------------------------------------------------------------
The setup
----------------------------------------------------------------------
On VE, three Marks matter:
  - Mark 14, Mark 01: passive MMs at the inside (capture spread)
  - Mark 55: aggressive taker, ~400 trades/day, pays half-spread
  - Mark 67: pure buyer, ~55 trades/day, qty avg ~9
  - Mark 49: pure seller, ~35 trades/day, qty avg ~10

Two alphas:
  (a) Spread capture from Mark 55. Pool ≈ $6.5K/day; we aim to capture
      a meaningful share by quoting 1 tick inside.
  (b) Directional signal: after Mark 67 buys or Mark 49 sells, VE drifts
      up by ~+1.5-2 over the next 10k ticks. This is a VE-low-and-recovers
      pattern — both Marks tend to print near local lows.

----------------------------------------------------------------------
Three layers
----------------------------------------------------------------------
Layer 1 — Passive MM
  Quote 1 tick inside the existing inside, no inventory tilt below brake
  threshold (80), hard tilt above. Same template as VEV_4000 v3d.

Layer 2 — Aggressive signal entry
  When market_trades on this tick contain a Mark 67 buy OR Mark 49 sell of
  size ≥ TRIGGER_MIN_QTY, IMMEDIATELY cross VE's ask to go long. Size =
  AGGRESSIVE_MULT × counterparty's qty. No exit timing logic, no hedging.

Layer 3 — Exit via natural MM
  When signal is "active" (recent enough), suppress our own ASK side so
  we don't unwind into the rising mid. Once signal has fully decayed,
  resume two-sided MM — Mark 55 (taker) will lift us at the higher mid,
  closing the long at a profit.

State (via traderData):
  - signal: float, decays geometrically each tick
  - signal_active_until_tick: timestamp at which signal is considered
    "active" for ask suppression
"""
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple
import json


class Trader:
    VE_SYM = "VELVETFRUIT_EXTRACT"
    VE_LIMIT = 200

    # ---- Layer 1 constants ----
    QUOTE_SIZE = 50
    INV_TILT = 0.0
    BRAKE_THRESH = 80
    BRAKE_TILT = 0.04
    MIN_HALF_SPREAD = 2

    # ---- Layer 2 constants ----
    TRIGGER_MIN_QTY = 3        # ignore tiny Mark 67/49 prints
    AGGRESSIVE_MULT = 2.0      # buy this many × Mark 67/49 qty
    AGGRESSIVE_MAX_QTY = 60    # per-tick cap on aggressive entry

    # ---- Layer 3 constants ----
    SIGNAL_PER_LOT = 0.10
    SIGNAL_DECAY = 0.9999      # geometric decay per tick (~10k ticks half-life)
    ASK_SUPPRESS_THRESHOLD = 0.05   # when signal > this, suppress ask quote

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        # 1) Restore state
        try:
            mem = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            mem = {}
        signal: float = float(mem.get("signal", 0.0))

        result: Dict[str, List[Order]] = {self.VE_SYM: []}

        # 2) Read VE book
        ve_book = state.order_depths.get(self.VE_SYM)
        if not self._book_ok(ve_book):
            new_mem = {"signal": signal * self.SIGNAL_DECAY}
            return result, 0, json.dumps(new_mem)

        ve_bb = max(ve_book.buy_orders); ve_ba = min(ve_book.sell_orders)
        ve_pos = int(state.position.get(self.VE_SYM, 0))

        # 3) Decay signal
        signal *= self.SIGNAL_DECAY

        # 4) Read market trades from last tick — look for Mark 67 buys / Mark 49 sells
        market_trades = state.market_trades.get(self.VE_SYM, []) or []
        triggered_qty = 0
        for trade in market_trades:
            qty = int(trade.quantity)
            # Mark 67 buys VE: signal +
            if trade.buyer == "Mark 67" and qty >= self.TRIGGER_MIN_QTY:
                triggered_qty += qty
                signal += qty * self.SIGNAL_PER_LOT
            # Mark 49 sells VE: signal +
            elif trade.seller == "Mark 49" and qty >= self.TRIGGER_MIN_QTY:
                triggered_qty += qty
                signal += qty * self.SIGNAL_PER_LOT

        # 5) Layer 2: aggressive long entry on triggered signal
        if triggered_qty > 0:
            target_qty = min(
                int(triggered_qty * self.AGGRESSIVE_MULT),
                self.AGGRESSIVE_MAX_QTY,
                max(0, self.VE_LIMIT - ve_pos),
                -int(ve_book.sell_orders[ve_ba]),  # available at the ask
            )
            if target_qty > 0:
                result[self.VE_SYM].append(Order(self.VE_SYM, ve_ba, +target_qty))

        # 6) Layer 3: passive MM with ASK suppression while signal is hot
        existing_buy = sum(o.quantity for o in result[self.VE_SYM] if o.quantity > 0)
        existing_sell = sum(-o.quantity for o in result[self.VE_SYM] if o.quantity < 0)
        bid_room = max(0, self.VE_LIMIT - ve_pos - existing_buy)
        ask_room = max(0, self.VE_LIMIT + ve_pos - existing_sell)

        passive = self._passive_quotes(
            ve_bb, ve_ba, ve_pos, bid_room, ask_room,
            ask_suppressed=(signal >= self.ASK_SUPPRESS_THRESHOLD),
        )
        result[self.VE_SYM].extend(passive)

        # 7) Save state
        new_mem = {"signal": signal}
        return result, 0, json.dumps(new_mem)

    @staticmethod
    def _book_ok(book) -> bool:
        return book is not None and book.buy_orders and book.sell_orders

    def _passive_quotes(
        self,
        ve_bb: int,
        ve_ba: int,
        ve_pos: int,
        bid_room: int,
        ask_room: int,
        ask_suppressed: bool,
    ) -> List[Order]:
        orders: List[Order] = []

        # Skew: piecewise (0 below brake, hard above)
        if abs(ve_pos) <= self.BRAKE_THRESH:
            skew = -self.INV_TILT * ve_pos
        else:
            excess = abs(ve_pos) - self.BRAKE_THRESH
            sign = 1 if ve_pos > 0 else -1
            skew = -self.BRAKE_TILT * excess * sign

        bid_px = ve_bb + 1
        ask_px = ve_ba - 1
        bid_px = int(round(bid_px + skew))
        ask_px = int(round(ask_px + skew))

        natural_mid = (ve_bb + ve_ba) / 2.0
        bid_px = min(bid_px, int(natural_mid - self.MIN_HALF_SPREAD))
        ask_px = max(ask_px, int(natural_mid + self.MIN_HALF_SPREAD))

        bid_px = min(bid_px, ve_ba - 1)
        ask_px = max(ask_px, ve_bb + 1)
        if bid_px >= ask_px:
            bid_px = ask_px - 1

        bid_size = min(self.QUOTE_SIZE, bid_room)
        ask_size = min(self.QUOTE_SIZE, ask_room)

        # ALWAYS quote bid
        if bid_size > 0:
            orders.append(Order(self.VE_SYM, bid_px, +bid_size))

        # SUPPRESS ask while signal is hot — don't sell back to a rising VE
        if ask_size > 0 and not ask_suppressed:
            orders.append(Order(self.VE_SYM, ask_px, -ask_size))

        return orders
