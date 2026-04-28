"""
VEV_4000 trader, R4 v4: v3d + VE directional signal injection.

Diff from v3d:
  - NEW: track Mark 67 buys + Mark 49 sells on VE as a bullish accumulator.
  - Apply that signal as an extra quote-price bias on top of inventory skew.

Why this should work
--------------------
From counterparty analysis (Notebook 02 + earlier per-Mark fwd-return work):
  - After Mark 67 BUYS VE  → VE mid +1.57 over next 10k ticks (vs 0.02 baseline)
  - After Mark 49 SELLS VE → VE mid +1.92 over next 10k ticks
  - VEV_4000 = VE - 4000 (delta-1), so the same prediction applies to VEV_4000
    fair value.

When the signal is strongly positive, we shift our quotes UP. Mark 38 still
hits us at price priority, but our fill prices are slightly higher (we
buy higher when we expect VE to keep going up). Net effect: fills happen
at prices closer to the new (higher) fair, capturing some of the predicted
move on the inventory we accumulate.

Implementation
--------------
ve_signal: float, accumulated bullish pressure
  - Decays exponentially per tick: SIGNAL_DECAY (= 0.9995, half-life ~1400 ticks)
  - Increases by SIGNAL_PER_LOT for each lot of Mark 67 buy or Mark 49 sell
  - Capped at SIGNAL_MAX (prevent runaway from clusters)

In _passive_quotes:
  total_skew = inv_skew + ve_signal
  bid_px = base_bid + total_skew
  ask_px = base_ask + total_skew

When VE is moving up (Mark 67/49 active), our quotes shift up by 1-3 ticks,
so we end up paying 1-3 ticks more on each buy fill — but that fill is now
at a price closer to the post-move fair. Net per-share alpha is modest, but
applied across ~135 fills/day plus any inventory drift, expected $200-500/day.

Risk: if signal fires but VE doesn't actually move up (false signal), we paid
extra for nothing. The half-life of 1400 ticks limits the duration of any
single signal's bias.
"""
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple
import json


class Trader:
    # ---- products & limits ----
    VEV_SYM = "VEV_4000"
    VE_SYM = "VELVETFRUIT_EXTRACT"
    VEV_LIMIT = 300

    # ---- Layer 1: passive MM constants ----
    QUOTE_SIZE = 60
    INV_TILT = 0.0
    BRAKE_THRESH = 100
    BRAKE_TILT = 0.04
    MIN_HALF_SPREAD = 8

    # ---- Layer 3: stale-quote sniping ----
    VE_HISTORY_LEN = 10
    VE_TRIGGER_MOVE = 3
    VE_TRIGGER_LOOKBACK = 5
    SNIPE_EDGE_MIN = 3
    SNIPE_MAX_QTY = 60

    # ---- NEW Layer 4: VE directional signal ----
    SIGNAL_DECAY = 0.9995          # per tick; half-life ~1386 ticks
    SIGNAL_PER_LOT = 0.17          # 1 lot of M67 buy or M49 sell ≈ +0.17 fair shift
    SIGNAL_MAX = 5.0               # cap absolute bias (ticks)

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        # 1) Restore state
        try:
            mem = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            mem = {}
        ve_hist: List[float] = list(mem.get("ve_mid_history", []))
        ve_signal: float = float(mem.get("ve_signal", 0.0))

        result: Dict[str, List[Order]] = {self.VEV_SYM: []}

        # 2) Read books
        vev_book = state.order_depths.get(self.VEV_SYM)
        ve_book = state.order_depths.get(self.VE_SYM)
        if not self._book_ok(vev_book):
            return result, 0, json.dumps({
                "ve_mid_history": ve_hist, "ve_signal": ve_signal
            })

        vev_bb = max(vev_book.buy_orders); vev_ba = min(vev_book.sell_orders)

        ve_mid = None
        if self._book_ok(ve_book):
            ve_bb = max(ve_book.buy_orders); ve_ba = min(ve_book.sell_orders)
            ve_mid = (ve_bb + ve_ba) / 2.0
            ve_hist.append(ve_mid)
            if len(ve_hist) > self.VE_HISTORY_LEN:
                ve_hist = ve_hist[-self.VE_HISTORY_LEN:]

        # 3) Position
        vev_pos = int(state.position.get(self.VEV_SYM, 0))

        # 4) NEW: update VE directional signal from market trades
        ve_signal *= self.SIGNAL_DECAY
        ve_market_trades = state.market_trades.get(self.VE_SYM, []) or []
        for t in ve_market_trades:
            if t.buyer == "Mark 67":
                ve_signal += self.SIGNAL_PER_LOT * t.quantity
            elif t.seller == "Mark 49":
                ve_signal += self.SIGNAL_PER_LOT * t.quantity
        # Cap
        if ve_signal > self.SIGNAL_MAX:
            ve_signal = self.SIGNAL_MAX
        elif ve_signal < -self.SIGNAL_MAX:
            ve_signal = -self.SIGNAL_MAX

        # 5) Layer 3: stale-quote snipe
        if ve_mid is not None:
            vev_fair = ve_mid - 4000.0
            snipe_orders = self._maybe_snipe(vev_book, vev_pos, vev_fair, ve_hist)
            result[self.VEV_SYM].extend(snipe_orders)

        # 6) Layer 1: passive MM (with inv skew + VE signal skew)
        existing_buy = sum(o.quantity for o in result[self.VEV_SYM] if o.quantity > 0)
        existing_sell = sum(-o.quantity for o in result[self.VEV_SYM] if o.quantity < 0)
        bid_room = max(0, self.VEV_LIMIT - vev_pos - existing_buy)
        ask_room = max(0, self.VEV_LIMIT + vev_pos - existing_sell)
        passive = self._passive_quotes(
            vev_bb, vev_ba, vev_pos, bid_room, ask_room, ve_signal
        )
        result[self.VEV_SYM].extend(passive)

        # 7) Save state
        return result, 0, json.dumps({
            "ve_mid_history": ve_hist, "ve_signal": ve_signal
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _book_ok(book) -> bool:
        return book is not None and book.buy_orders and book.sell_orders

    def _maybe_snipe(self, vev_book, vev_pos, vev_fair, ve_hist):
        orders: List[Order] = []
        if len(ve_hist) < self.VE_TRIGGER_LOOKBACK + 1:
            return orders
        recent_change = ve_hist[-1] - ve_hist[-1 - self.VE_TRIGGER_LOOKBACK]
        if abs(recent_change) < self.VE_TRIGGER_MOVE:
            return orders

        vev_bb = max(vev_book.buy_orders); vev_ba = min(vev_book.sell_orders)

        if vev_bb >= vev_fair + self.SNIPE_EDGE_MIN:
            avail = int(vev_book.buy_orders[vev_bb])
            qty = min(avail, self.SNIPE_MAX_QTY, max(0, self.VEV_LIMIT + vev_pos))
            if qty > 0:
                orders.append(Order(self.VEV_SYM, vev_bb, -qty))
        if vev_ba <= vev_fair - self.SNIPE_EDGE_MIN:
            avail = -int(vev_book.sell_orders[vev_ba])
            qty = min(avail, self.SNIPE_MAX_QTY, max(0, self.VEV_LIMIT - vev_pos))
            if qty > 0:
                orders.append(Order(self.VEV_SYM, vev_ba, +qty))
        return orders

    def _passive_quotes(
        self,
        vev_bb: int,
        vev_ba: int,
        vev_pos: int,
        bid_room: int,
        ask_room: int,
        ve_signal: float,
    ) -> List[Order]:
        orders: List[Order] = []

        bid_px = vev_bb + 1
        ask_px = vev_ba - 1

        # Inventory skew (piecewise, brake at BRAKE_THRESH)
        if abs(vev_pos) <= self.BRAKE_THRESH:
            inv_skew = -self.INV_TILT * vev_pos
        else:
            excess = abs(vev_pos) - self.BRAKE_THRESH
            sign = 1 if vev_pos > 0 else -1
            inv_skew = -self.BRAKE_TILT * excess * sign

        # NEW: VE directional signal — bullish signal pushes both quotes up
        total_skew = inv_skew + ve_signal

        bid_px = int(round(bid_px + total_skew))
        ask_px = int(round(ask_px + total_skew))

        natural_mid = (vev_bb + vev_ba) / 2.0
        bid_px = min(bid_px, int(natural_mid - self.MIN_HALF_SPREAD))
        ask_px = max(ask_px, int(natural_mid + self.MIN_HALF_SPREAD))

        bid_px = min(bid_px, vev_ba - 1)
        ask_px = max(ask_px, vev_bb + 1)
        if bid_px >= ask_px:
            bid_px = ask_px - 1

        bid_size = min(self.QUOTE_SIZE, bid_room)
        ask_size = min(self.QUOTE_SIZE, ask_room)

        if bid_size > 0:
            orders.append(Order(self.VEV_SYM, bid_px, +bid_size))
        if ask_size > 0:
            orders.append(Order(self.VEV_SYM, ask_px, -ask_size))
        return orders
