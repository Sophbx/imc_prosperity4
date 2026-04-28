"""
VE-only swing trader, R4 v2: v5 (idle MM) + swing.

Diff from v1: in the "do nothing" early window (ts 0-480k) we run v5's
wall-mid + EMA passive MM. From ts 480k onward we flatten v5's residual
inventory and hand off to the swing strategy from v1.

Phase map:
  ts < 480k         : v5 logic (passive MM with Mark 55 anti-signal)
  ts 480k - 500k    : FLATTEN v5's residual position (passive at edge,
                       cross spread if still long/short by ts 495k)
  ts 500k - 700k    : SWING ENTRY (accumulate long up to 200)
  ts 700k - 900k    : HOLD 200 long
  ts 900k - 970k    : SWING EXIT (passive unwind)
  ts 970k - 1000k   : FORCE FLAT (cross spread)

The v5 logic is embedded inline (no nested-class dance) so this remains a
single self-contained trader file. v5's traderData state (ema_val, signal)
persists via JSON-encoded mem dict.
"""
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Optional, Tuple
import json
import math


class Trader:
    VE_SYM = "VELVETFRUIT_EXTRACT"
    VE_LIMIT = 200

    # ---- v5 config ----
    EMA_ALPHA = 0.05
    WALL_WEIGHT = 0.70
    TAKE_EDGE = 1.0
    MAKE_EDGE = 1
    QUOTE_SIZE_MM = 25
    INV_SKEW = 0.08
    TRIGGER_MIN_QTY = 3
    SIGNAL_PER_LOT = 0.04
    SIGNAL_DECAY = 0.9995
    MAX_FAIR_SHIFT = 2.0

    # ---- swing phase boundaries ----
    V5_END = 480_000
    ENTRY_START = 500_000
    ENTRY_END = 700_000
    EXIT_START = 900_000
    EXIT_FORCE = 970_000

    # ---- swing config ----
    SWING_QUOTE_SIZE = 80
    FORCE_FLATTEN_TS = 495_000  # last 5k of buffer: cross spread if still not flat

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        try:
            mem = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            mem = {}
        ema_val: Optional[float] = mem.get("ema_val")
        signal: float = float(mem.get("signal", 0.0))

        result: Dict[str, List[Order]] = {self.VE_SYM: []}
        depth = state.order_depths.get(self.VE_SYM)
        if not self._book_ok(depth):
            new_mem = {"ema_val": ema_val, "signal": signal * self.SIGNAL_DECAY}
            return result, 0, json.dumps(new_mem)

        bb = max(depth.buy_orders); ba = min(depth.sell_orders)
        if bb >= ba:
            return result, 0, json.dumps({"ema_val": ema_val, "signal": signal})

        pos = int(state.position.get(self.VE_SYM, 0))
        ts = int(state.timestamp)

        # PHASE 1: v5 MM (ts < 480k)
        if ts < self.V5_END:
            new_ema, new_signal = self._v5_logic(state, depth, ema_val, signal, pos, result)
            ema_val, signal = new_ema, new_signal

        # PHASE 1.5: BUFFER FLATTEN (ts 480k - 500k)
        elif ts < self.ENTRY_START:
            self._flatten(result, depth, pos, ts)

        # PHASE 2: SWING ENTRY (500k - 700k)
        elif ts < self.ENTRY_END:
            room = max(0, self.VE_LIMIT - pos)
            if room > 0:
                size = min(self.SWING_QUOTE_SIZE, room)
                result[self.VE_SYM].append(Order(self.VE_SYM, bb + 1, +size))

        # PHASE 3: HOLD (700k - 900k) — no orders

        # PHASE 4: SWING EXIT PASSIVE (900k - 970k)
        elif ts < self.EXIT_FORCE:
            if pos > 0:
                size = min(self.SWING_QUOTE_SIZE, pos)
                result[self.VE_SYM].append(Order(self.VE_SYM, ba - 1, -size))

        # PHASE 5: FORCE FLAT (970k+)
        else:
            if pos > 0:
                avail = int(depth.buy_orders[bb])
                size = min(pos, avail)
                if size > 0:
                    result[self.VE_SYM].append(Order(self.VE_SYM, bb, -size))
            elif pos < 0:
                avail = -int(depth.sell_orders[ba])
                size = min(-pos, avail)
                if size > 0:
                    result[self.VE_SYM].append(Order(self.VE_SYM, ba, +size))

        new_mem = {"ema_val": ema_val, "signal": signal}
        return result, 0, json.dumps(new_mem)

    # ------------------------------------------------------------------
    # v5 logic (inline, condensed copy of trader_ve_v5.py's run loop)
    # ------------------------------------------------------------------
    def _v5_logic(
        self,
        state: TradingState,
        depth: OrderDepth,
        ema_val: Optional[float],
        signal: float,
        pos: int,
        result: Dict[str, List[Order]],
    ):
        mid = (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0
        wall = self._wall_mid(depth) or mid

        ema_val = wall if ema_val is None else (
            self.EMA_ALPHA * wall + (1.0 - self.EMA_ALPHA) * ema_val
        )
        fair_base = self.WALL_WEIGHT * wall + (1.0 - self.WALL_WEIGHT) * ema_val

        signal *= self.SIGNAL_DECAY
        for trade in state.market_trades.get(self.VE_SYM, []) or []:
            qty = int(trade.quantity)
            if qty < self.TRIGGER_MIN_QTY:
                continue
            if trade.seller == "Mark 55":
                signal += qty * self.SIGNAL_PER_LOT
            elif trade.buyer == "Mark 55":
                signal -= qty * self.SIGNAL_PER_LOT

        bias = max(-self.MAX_FAIR_SHIFT, min(self.MAX_FAIR_SHIFT, signal))
        fair_eff = fair_base + bias

        # take crossed
        for ask in sorted(depth.sell_orders.keys()):
            if ask <= fair_eff - self.TAKE_EDGE:
                avail = -depth.sell_orders[ask]
                qty = max(0, min(avail, self.VE_LIMIT - pos))
                if qty > 0:
                    result[self.VE_SYM].append(Order(self.VE_SYM, ask, qty))
                    pos += qty
        for bid in sorted(depth.buy_orders.keys(), reverse=True):
            if bid >= fair_eff + self.TAKE_EDGE:
                avail = depth.buy_orders[bid]
                qty = max(0, min(avail, self.VE_LIMIT + pos))
                if qty > 0:
                    result[self.VE_SYM].append(Order(self.VE_SYM, bid, -qty))
                    pos -= qty

        # make
        bb = max(depth.buy_orders); ba = min(depth.sell_orders)
        fair_adj = fair_eff - self.INV_SKEW * pos
        buy_px = int(math.floor(fair_adj - self.MAKE_EDGE))
        sell_px = int(math.ceil(fair_adj + self.MAKE_EDGE))
        buy_px = min(buy_px, bb + 1)
        sell_px = max(sell_px, ba - 1)
        if buy_px < ba:
            buy_qty = max(0, min(self.QUOTE_SIZE_MM, self.VE_LIMIT - pos))
            if buy_qty > 0:
                result[self.VE_SYM].append(Order(self.VE_SYM, buy_px, buy_qty))
        if sell_px > bb:
            sell_qty = max(0, min(self.QUOTE_SIZE_MM, self.VE_LIMIT + pos))
            if sell_qty > 0:
                result[self.VE_SYM].append(Order(self.VE_SYM, sell_px, -sell_qty))

        return ema_val, signal

    def _flatten(
        self,
        result: Dict[str, List[Order]],
        depth: OrderDepth,
        pos: int,
        ts: int,
    ) -> None:
        """Buffer-window flatten: passive at edge first, cross if very late."""
        if pos == 0: return
        bb = max(depth.buy_orders); ba = min(depth.sell_orders)
        if ts >= self.FORCE_FLATTEN_TS:
            # Force cross
            if pos > 0:
                avail = int(depth.buy_orders[bb])
                size = min(pos, avail)
                if size > 0:
                    result[self.VE_SYM].append(Order(self.VE_SYM, bb, -size))
            else:
                avail = -int(depth.sell_orders[ba])
                size = min(-pos, avail)
                if size > 0:
                    result[self.VE_SYM].append(Order(self.VE_SYM, ba, +size))
        else:
            # Passive at edge
            if pos > 0:
                result[self.VE_SYM].append(Order(self.VE_SYM, ba - 1, -min(self.SWING_QUOTE_SIZE, pos)))
            else:
                result[self.VE_SYM].append(Order(self.VE_SYM, bb + 1, +min(self.SWING_QUOTE_SIZE, -pos)))

    @staticmethod
    def _book_ok(book) -> bool:
        return book is not None and book.buy_orders and book.sell_orders

    @staticmethod
    def _wall_mid(depth: OrderDepth) -> Optional[float]:
        if not depth.buy_orders or not depth.sell_orders:
            return None
        bid_wall = max(depth.buy_orders.items(), key=lambda x: abs(x[1]))[0]
        ask_wall = min(depth.sell_orders.items(), key=lambda x: -abs(x[1]))[0]
        return (bid_wall + ask_wall) / 2.0
