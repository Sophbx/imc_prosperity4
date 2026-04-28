"""
VEV_4000-only trader, R4 v3d.

Diff from v3a: piecewise inventory skew with brake.

  - |pos| <= BRAKE_THRESH (100): no skew (= v3c behavior, full drift)
  - |pos| >  BRAKE_THRESH:       hard brake with TILT=0.04

Idea: get the v3c upside (let inventory ride for mean reversion) but
prevent runaway losses if the day is trending. Below 100 we behave like
v3c. Above 100 the brake kicks in and pulls us back.

Picks the best of v3c and v3a's behavior depending on regime.
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
    INV_TILT = 0.0            # below brake: no skew
    BRAKE_THRESH = 100        # |pos| at which the brake kicks in
    BRAKE_TILT = 0.04         # post-brake skew (= v3a's tilt strength)
    MIN_HALF_SPREAD = 8

    # ---- Layer 3: stale-quote sniping constants ----
    VE_HISTORY_LEN = 10
    VE_TRIGGER_MOVE = 3
    VE_TRIGGER_LOOKBACK = 5
    SNIPE_EDGE_MIN = 3        # no longer needs to cover VE-hedge spread (no hedge)
    SNIPE_MAX_QTY = 60

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        # 1) Restore state
        try:
            mem = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            mem = {}
        ve_hist: List[float] = list(mem.get("ve_mid_history", []))

        result: Dict[str, List[Order]] = {self.VEV_SYM: []}

        # 2) Read books
        vev_book = state.order_depths.get(self.VEV_SYM)
        ve_book = state.order_depths.get(self.VE_SYM)
        if not self._book_ok(vev_book):
            return result, 0, json.dumps({"ve_mid_history": ve_hist})

        vev_bb = max(vev_book.buy_orders); vev_ba = min(vev_book.sell_orders)

        # VE book is optional — only used for Layer 3 fair check
        ve_mid = None
        if self._book_ok(ve_book):
            ve_bb = max(ve_book.buy_orders); ve_ba = min(ve_book.sell_orders)
            ve_mid = (ve_bb + ve_ba) / 2.0
            ve_hist.append(ve_mid)
            if len(ve_hist) > self.VE_HISTORY_LEN:
                ve_hist = ve_hist[-self.VE_HISTORY_LEN:]

        # 3) Position
        vev_pos = int(state.position.get(self.VEV_SYM, 0))

        # 4) Layer 3: stale-quote snipe (only if VE info available)
        if ve_mid is not None:
            vev_fair = ve_mid - 4000.0
            snipe_orders = self._maybe_snipe(
                vev_book, vev_pos, vev_fair, ve_hist
            )
            result[self.VEV_SYM].extend(snipe_orders)

        # 5) Layer 1: passive MM
        existing_buy = sum(o.quantity for o in result[self.VEV_SYM] if o.quantity > 0)
        existing_sell = sum(-o.quantity for o in result[self.VEV_SYM] if o.quantity < 0)
        bid_room = max(0, self.VEV_LIMIT - vev_pos - existing_buy)
        ask_room = max(0, self.VEV_LIMIT + vev_pos - existing_sell)
        passive = self._passive_quotes(vev_bb, vev_ba, vev_pos, bid_room, ask_room)
        result[self.VEV_SYM].extend(passive)

        # 6) Save state
        return result, 0, json.dumps({"ve_mid_history": ve_hist})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _book_ok(book) -> bool:
        return book is not None and book.buy_orders and book.sell_orders

    def _maybe_snipe(
        self,
        vev_book: OrderDepth,
        vev_pos: int,
        vev_fair: float,
        ve_hist: List[float],
    ) -> List[Order]:
        orders: List[Order] = []

        if len(ve_hist) < self.VE_TRIGGER_LOOKBACK + 1:
            return orders
        recent_change = ve_hist[-1] - ve_hist[-1 - self.VE_TRIGGER_LOOKBACK]
        if abs(recent_change) < self.VE_TRIGGER_MOVE:
            return orders

        vev_bb = max(vev_book.buy_orders); vev_ba = min(vev_book.sell_orders)

        # Snipe SELL: Mark 14's bid is too high vs theoretical fair.
        if vev_bb >= vev_fair + self.SNIPE_EDGE_MIN:
            avail_at_bid = int(vev_book.buy_orders[vev_bb])
            vev_room_for_sell = max(0, self.VEV_LIMIT + vev_pos)
            qty = min(avail_at_bid, self.SNIPE_MAX_QTY, vev_room_for_sell)
            if qty > 0:
                orders.append(Order(self.VEV_SYM, vev_bb, -qty))

        # Snipe BUY: Mark 14's ask is too low vs theoretical fair.
        if vev_ba <= vev_fair - self.SNIPE_EDGE_MIN:
            avail_at_ask = -int(vev_book.sell_orders[vev_ba])
            vev_room_for_buy = max(0, self.VEV_LIMIT - vev_pos)
            qty = min(avail_at_ask, self.SNIPE_MAX_QTY, vev_room_for_buy)
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
    ) -> List[Order]:
        orders: List[Order] = []

        bid_px = vev_bb + 1
        ask_px = vev_ba - 1

        # Piecewise skew: 0 below brake threshold, linear above
        if abs(vev_pos) <= self.BRAKE_THRESH:
            skew = -self.INV_TILT * vev_pos
        else:
            excess = abs(vev_pos) - self.BRAKE_THRESH
            sign = 1 if vev_pos > 0 else -1
            skew = -self.BRAKE_TILT * excess * sign
        bid_px = int(round(bid_px + skew))
        ask_px = int(round(ask_px + skew))

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
