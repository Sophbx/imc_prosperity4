"""
VE-only trader, R4 v6.

Diff from v5: MAX_FAIR_SHIFT lowered 2.0 -> 1.0.

Hypothesis: v5 worked by injecting Mark 55 anti-signal as a fair-value
bias of up to ±2 ticks. With VE's natural spread of ~6 (half-spread 3),
±2 ticks may be too aggressive — the bias is 67% of half-spread, which
means we're regularly priced past natural fair. Reducing to ±1 keeps the
signal direction but cuts the magnitude in half. Test whether the smaller
shift trades total alpha for path stability.

----------------------------------------------------------------------
Why Mark 55 anti-signal
----------------------------------------------------------------------
From the per-Mark VE analysis (3 days):
  Mark 55: 1198 trades (4x Mark 67/49 combined), prem_to_mid = +2.48
           edge_5k = -2.09  ←  his trades systematically LOSE next 5k ticks

So Mark 55's direction is anti-predictive:
  - Mark 55 SELLS  → mid rises afterward  → bias fair UP   → we lean LONG
  - Mark 55 BUYS   → mid falls afterward  → bias fair DOWN → we lean SHORT

Compared to v4's Mark 67/49 signal (90 events/day, both bias UP), Mark 55
fires ~400 events/day with both directions. **More frequent, two-sided,
higher resolution** — but each event has slightly weaker single-event
expectation (-2.09/share vs Mark 67/49's +1.5-2/share).

Empirical question whether the higher event count beats the smaller
per-event size.
"""
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Optional, Tuple
import json
import math


class Trader:
    VE_SYM = "VELVETFRUIT_EXTRACT"
    VE_LIMIT = 200

    # ---- Layer 1 (R3 v4 baseline) ----
    EMA_ALPHA = 0.05
    WALL_WEIGHT = 0.70
    TAKE_EDGE = 1.0
    MAKE_EDGE = 1
    QUOTE_SIZE = 25
    INV_SKEW = 0.08

    # ---- Layer 2 (Mark 55 anti-signal) ----
    TRIGGER_MIN_QTY = 3
    SIGNAL_PER_LOT = 0.04   # smaller per-lot weight since Mark 55 fires 4x more
    SIGNAL_DECAY = 0.9995   # faster decay (~2k tick half-life) — Mark 55 events
                            # are short-horizon mean-reversion, not long-horizon trend
    MAX_FAIR_SHIFT = 1.0   # halved from v5

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        try:
            mem = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            mem = {}
        ema_val: Optional[float] = mem.get("ema_val")
        signal: float = float(mem.get("signal", 0.0))

        result: Dict[str, List[Order]] = {self.VE_SYM: []}
        depth = state.order_depths.get(self.VE_SYM)
        if depth is None or not depth.buy_orders or not depth.sell_orders:
            new_mem = {"ema_val": ema_val, "signal": signal * self.SIGNAL_DECAY}
            return result, 0, json.dumps(new_mem)

        mid = self._get_mid(depth)
        wall = self._get_wall_mid(depth)
        if wall is None: wall = mid
        if mid is None: mid = wall
        if mid is None or wall is None:
            new_mem = {"ema_val": ema_val, "signal": signal * self.SIGNAL_DECAY}
            return result, 0, json.dumps(new_mem)

        ema_val = wall if ema_val is None else (
            self.EMA_ALPHA * wall + (1.0 - self.EMA_ALPHA) * ema_val
        )
        fair_base = self.WALL_WEIGHT * wall + (1.0 - self.WALL_WEIGHT) * ema_val

        # ----- Mark 55 ANTI-signal -----
        signal *= self.SIGNAL_DECAY
        for trade in state.market_trades.get(self.VE_SYM, []) or []:
            qty = int(trade.quantity)
            if qty < self.TRIGGER_MIN_QTY:
                continue
            if trade.seller == "Mark 55":
                # he just sold low → bias fair UP
                signal += qty * self.SIGNAL_PER_LOT
            elif trade.buyer == "Mark 55":
                # he just bought high → bias fair DOWN
                signal -= qty * self.SIGNAL_PER_LOT

        bias = max(-self.MAX_FAIR_SHIFT, min(self.MAX_FAIR_SHIFT, signal))
        fair_eff = fair_base + bias

        position = int(state.position.get(self.VE_SYM, 0))
        orders: List[Order] = []
        position = self._take_crossed(orders, depth, fair_eff, position)
        self._make_quotes(orders, depth, fair_eff, position)
        result[self.VE_SYM] = orders

        new_mem = {"ema_val": ema_val, "signal": signal}
        return result, 0, json.dumps(new_mem)

    @staticmethod
    def _get_mid(depth: OrderDepth) -> Optional[float]:
        if not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0

    @staticmethod
    def _get_wall_mid(depth: OrderDepth) -> Optional[float]:
        if not depth.buy_orders or not depth.sell_orders:
            return None
        bid_wall_price = max(depth.buy_orders.items(), key=lambda x: abs(x[1]))[0]
        ask_wall_price = min(depth.sell_orders.items(), key=lambda x: -abs(x[1]))[0]
        return (bid_wall_price + ask_wall_price) / 2.0

    def _take_crossed(
        self,
        orders: List[Order],
        depth: OrderDepth,
        fair: float,
        position: int,
    ) -> int:
        for ask in sorted(depth.sell_orders.keys()):
            if ask <= fair - self.TAKE_EDGE:
                avail = -depth.sell_orders[ask]
                qty = max(0, min(avail, self.VE_LIMIT - position))
                if qty > 0:
                    orders.append(Order(self.VE_SYM, ask, qty))
                    position += qty
        for bid in sorted(depth.buy_orders.keys(), reverse=True):
            if bid >= fair + self.TAKE_EDGE:
                avail = depth.buy_orders[bid]
                qty = max(0, min(avail, self.VE_LIMIT + position))
                if qty > 0:
                    orders.append(Order(self.VE_SYM, bid, -qty))
                    position -= qty
        return position

    def _make_quotes(
        self,
        orders: List[Order],
        depth: OrderDepth,
        fair: float,
        position: int,
    ) -> None:
        bb = max(depth.buy_orders) if depth.buy_orders else None
        ba = min(depth.sell_orders) if depth.sell_orders else None
        if bb is None or ba is None:
            return

        fair_adj = fair - self.INV_SKEW * position
        buy_px = int(math.floor(fair_adj - self.MAKE_EDGE))
        sell_px = int(math.ceil(fair_adj + self.MAKE_EDGE))

        buy_px = min(buy_px, bb + 1)
        sell_px = max(sell_px, ba - 1)

        if buy_px < ba:
            buy_qty = max(0, min(self.QUOTE_SIZE, self.VE_LIMIT - position))
            if buy_qty > 0:
                orders.append(Order(self.VE_SYM, buy_px, buy_qty))
        if sell_px > bb:
            sell_qty = max(0, min(self.QUOTE_SIZE, self.VE_LIMIT + position))
            if sell_qty > 0:
                orders.append(Order(self.VE_SYM, sell_px, -sell_qty))
