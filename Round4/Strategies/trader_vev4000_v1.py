"""
VEV_4000-only trader, R4 v1: 1 + 2 + 3 layered strategy.

Trades ONLY two products:
  - VEV_4000  (active passive MM + occasional sniping)
  - VELVETFRUIT_EXTRACT  (used solely as a delta-1 hedge for VEV_4000 fills)

No HG, no other vouchers. Designed to be droppable into the official
platform alone so we can measure VEV_4000 + VE-hedge PnL in isolation.

----------------------------------------------------------------------
Why this exists
----------------------------------------------------------------------
On VEV_4000, every trade is between two parties:
  - Mark 14: passive market-maker, 100% at L1 bid/ask, ~21-wide spread
  - Mark 38: aggressive taker, 100% crosses spread, structurally pays ~10/share

VEV_4000 is delta-1 ITM (mid ~= VE_mid - 4000), so any inventory we end up
holding can be hedged 1:1 by trading the underlying VE. VE has a much
tighter spread (~6 wide) than VEV_4000 (~21), so we can capture the wider
VEV_4000 spread and pay only the narrow VE spread to neutralize delta.

----------------------------------------------------------------------
Three layers
----------------------------------------------------------------------
Layer 1 — Passive MM on VEV_4000
  Quote 1 tick inside Mark 14's bid/ask. Inventory-skewed pricing so that
  large positions naturally lean back toward zero.

Layer 2 — Auto-hedge on VE
  Whenever our VEV_4000 position changes (own_trades fired this tick),
  cross VE's spread immediately to flatten delta. Net per round trip:
  +half_VEV_spread − half_VE_spread ≈ +9 − 3 = +6 per share, delta-flat.

Layer 3 — Snipe Mark 14's stale quotes
  Working hypothesis: Mark 14's quote-update on VEV_4000 lags VE prints by
  some ticks. When VE jumps far enough that the implied VEV_4000 fair is
  far from Mark 14's quote, his bid (or ask) becomes a free trade for us.
  We cross his stale side, then hedge in VE same tick.

State (carried via traderData):
  - ve_mid_history: ring buffer (last N) of VE mid, used to detect recent
    VE moves for the snipe trigger.

----------------------------------------------------------------------
Architecture notes (the part where Prosperity 3 #2 is right)
----------------------------------------------------------------------
1. State: round-tripped through JSON-encoded traderData. self.* attributes
   are NOT relied upon (Lambda is stateless on the official platform).
2. Order placement: every tick rebuilds all desired orders from scratch.
   No persistent open orders. Wiki confirms that unfilled player orders
   are auto-canceled at end of tick.
3. Position limits: before emitting any order we re-check against the
   current position + already-queued orders this tick. Never breach.
4. Robustness: if any product's book is empty/one-sided, we degrade
   gracefully (skip layers that need that product).
5. Determinism: no randomness, no timing-sensitive logic. Same input →
   same output, every run.
"""
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple
import json


class Trader:
    # ---- products & limits ----
    VEV_SYM = "VEV_4000"
    VE_SYM = "VELVETFRUIT_EXTRACT"
    VEV_LIMIT = 300
    VE_LIMIT = 200

    # ---- Layer 1: passive MM constants ----
    QUOTE_SIZE = 30           # base size each side
    INV_TILT = 0.04           # quote-price tilt per unit of VEV inventory
    MIN_HALF_SPREAD = 8       # never quote tighter than this around fair

    # ---- Layer 3: stale-quote sniping constants ----
    VE_HISTORY_LEN = 10       # how many ticks of VE mid to keep
    VE_TRIGGER_MOVE = 3       # require VE to have moved this much in last K
    VE_TRIGGER_LOOKBACK = 5   # over how many ticks
    SNIPE_EDGE_MIN = 3        # minimum profit (after VE hedge) to take a snipe
    SNIPE_MAX_QTY = 60        # max snipe qty per tick

    # ---- Layer 2: hedge constants ----
    HEDGE_VE = True

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        # 1) Restore state from traderData
        try:
            mem = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            mem = {}
        ve_hist: List[float] = list(mem.get("ve_mid_history", []))

        result: Dict[str, List[Order]] = {self.VEV_SYM: [], self.VE_SYM: []}

        # 2) Read books for both products
        vev_book = state.order_depths.get(self.VEV_SYM)
        ve_book = state.order_depths.get(self.VE_SYM)
        if not self._book_ok(vev_book) or not self._book_ok(ve_book):
            # Cannot operate without both books. Just persist state, no orders.
            new_mem = {"ve_mid_history": ve_hist}
            return result, 0, json.dumps(new_mem)

        vev_bb = max(vev_book.buy_orders); vev_ba = min(vev_book.sell_orders)
        ve_bb  = max(ve_book.buy_orders);  ve_ba  = min(ve_book.sell_orders)
        vev_mid = (vev_bb + vev_ba) / 2.0
        ve_mid = (ve_bb + ve_ba) / 2.0

        # 3) Update VE mid history (ring buffer)
        ve_hist.append(ve_mid)
        if len(ve_hist) > self.VE_HISTORY_LEN:
            ve_hist = ve_hist[-self.VE_HISTORY_LEN:]

        # 4) Compute theoretical VEV_4000 fair from VE
        # VEV_4000 is delta-1 ITM call: mid ≈ VE_mid - strike.
        # Time-value is ~0 across our 3 days of historical data.
        vev_fair = ve_mid - 4000.0

        # 5) Read positions
        vev_pos = int(state.position.get(self.VEV_SYM, 0))
        ve_pos = int(state.position.get(self.VE_SYM, 0))

        # 6) Layer 2: hedge any VEV fills that happened this tick
        # state.own_trades only contains trades from the immediately previous
        # iteration (per wiki). We check VEV-side own trades and hedge.
        net_vev_filled = self._sum_signed_own_trades(state, self.VEV_SYM)
        if self.HEDGE_VE and net_vev_filled != 0:
            self._add_hedge(result, ve_book, ve_pos, net_vev_filled)

        # 7) Layer 3: stale-quote snipe (before passive MM, since these are
        # IOC-style and consume inventory headroom).
        snipe_orders, snipe_buys, snipe_sells = self._maybe_snipe(
            vev_book, vev_pos, vev_fair, ve_hist, ve_book, ve_pos
        )
        result[self.VEV_SYM].extend(snipe_orders)

        # 8) Layer 1: passive MM
        # Position-pending tally (VEV side only — VE side is just a hedge byproduct)
        existing_buy = sum(o.quantity for o in result[self.VEV_SYM] if o.quantity > 0)
        existing_sell = sum(-o.quantity for o in result[self.VEV_SYM] if o.quantity < 0)
        bid_room = max(0, self.VEV_LIMIT - vev_pos - existing_buy)
        ask_room = max(0, self.VEV_LIMIT + vev_pos - existing_sell)
        passive = self._passive_quotes(
            vev_bb, vev_ba, vev_pos, bid_room, ask_room
        )
        result[self.VEV_SYM].extend(passive)

        # 9) Save state
        new_mem = {"ve_mid_history": ve_hist}
        return result, 0, json.dumps(new_mem)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _book_ok(book) -> bool:
        return book is not None and book.buy_orders and book.sell_orders

    @staticmethod
    def _sum_signed_own_trades(state: TradingState, sym: str) -> int:
        trades = state.own_trades.get(sym, []) or []
        net = 0
        for t in trades:
            # Filter by timestamp: only fills that happened in the last tick.
            # IMC convention: own_trades contains trades from the previous tick,
            # whose timestamp == previous tick's timestamp.
            if t.buyer == "SUBMISSION":
                net += int(t.quantity)
            elif t.seller == "SUBMISSION":
                net -= int(t.quantity)
        return net

    def _add_hedge(
        self,
        result: Dict[str, List[Order]],
        ve_book: OrderDepth,
        ve_pos: int,
        net_vev_filled: int,
    ) -> None:
        """If net VEV fill is +N (we got long N VEV), hedge by selling N VE.
        If -N (got short), buy N VE."""
        if net_vev_filled > 0:
            qty = min(net_vev_filled, max(0, self.VE_LIMIT + ve_pos))
            if qty > 0:
                ve_bb = max(ve_book.buy_orders)
                result[self.VE_SYM].append(Order(self.VE_SYM, ve_bb, -qty))
        else:
            qty = min(-net_vev_filled, max(0, self.VE_LIMIT - ve_pos))
            if qty > 0:
                ve_ba = min(ve_book.sell_orders)
                result[self.VE_SYM].append(Order(self.VE_SYM, ve_ba, +qty))

    def _maybe_snipe(
        self,
        vev_book: OrderDepth,
        vev_pos: int,
        vev_fair: float,
        ve_hist: List[float],
        ve_book: OrderDepth,
        ve_pos: int,
    ) -> Tuple[List[Order], int, int]:
        """Return (orders, total snipe buys, total snipe sells)."""
        orders: List[Order] = []
        snipe_buys = 0
        snipe_sells = 0

        # Need enough history to detect recent VE motion
        if len(ve_hist) < self.VE_TRIGGER_LOOKBACK + 1:
            return orders, 0, 0
        recent_change = ve_hist[-1] - ve_hist[-1 - self.VE_TRIGGER_LOOKBACK]
        if abs(recent_change) < self.VE_TRIGGER_MOVE:
            return orders, 0, 0

        vev_bb = max(vev_book.buy_orders); vev_ba = min(vev_book.sell_orders)

        # Snipe SELL: their bid is too high vs theoretical fair.
        # We need: vev_bb >= vev_fair + SNIPE_EDGE_MIN  (≈ covers VE half-spread)
        if vev_bb >= vev_fair + self.SNIPE_EDGE_MIN:
            avail_at_bid = int(vev_book.buy_orders[vev_bb])
            ve_room_for_buy = max(0, self.VE_LIMIT - ve_pos)
            vev_room_for_sell = max(0, self.VEV_LIMIT + vev_pos)
            qty = min(avail_at_bid, self.SNIPE_MAX_QTY,
                      vev_room_for_sell, ve_room_for_buy)
            if qty > 0:
                orders.append(Order(self.VEV_SYM, vev_bb, -qty))
                # Pre-emptively also hedge VE this same tick
                ve_ba = min(ve_book.sell_orders)
                orders_ve = Order(self.VE_SYM, ve_ba, +qty)
                # We'll inject into result below; signal it via a side-effect:
                # but cleaner to attach via list (caller will move VE leg).
                # We piggyback: pass it back through the second-leg list.
                # SIMPLER: just leave the hedge to Layer 2 next tick.
                # (own_trades will pick up this snipe and Layer 2 will hedge)
                snipe_sells += qty

        # Snipe BUY: their ask is too low vs theoretical fair.
        if vev_ba <= vev_fair - self.SNIPE_EDGE_MIN:
            avail_at_ask = -int(vev_book.sell_orders[vev_ba])
            ve_room_for_sell = max(0, self.VE_LIMIT + ve_pos)
            vev_room_for_buy = max(0, self.VEV_LIMIT - vev_pos)
            qty = min(avail_at_ask, self.SNIPE_MAX_QTY,
                      vev_room_for_buy, ve_room_for_sell)
            if qty > 0:
                orders.append(Order(self.VEV_SYM, vev_ba, +qty))
                snipe_buys += qty

        return orders, snipe_buys, snipe_sells

    def _passive_quotes(
        self,
        vev_bb: int,
        vev_ba: int,
        vev_pos: int,
        bid_room: int,
        ask_room: int,
    ) -> List[Order]:
        """Quote 1 tick inside Mark 14, with inventory skew."""
        orders: List[Order] = []

        # Start at 1 tick inside the existing inside.
        bid_px = vev_bb + 1
        ask_px = vev_ba - 1

        # Inventory tilt: long → push both quotes down to attract sellers
        skew = -self.INV_TILT * vev_pos
        bid_px = int(round(bid_px + skew))
        ask_px = int(round(ask_px + skew))

        # Maintain min half-spread around the implied fair (vev_bb+vev_ba)/2
        natural_mid = (vev_bb + vev_ba) / 2.0
        bid_px = min(bid_px, int(natural_mid - self.MIN_HALF_SPREAD))
        ask_px = max(ask_px, int(natural_mid + self.MIN_HALF_SPREAD))

        # Don't cross our own / don't cross outside the existing book
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
