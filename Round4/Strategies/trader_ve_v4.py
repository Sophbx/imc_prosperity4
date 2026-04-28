"""
VE-only trader, R4 v4.

Built on R3 v4's wall_mid + EMA passive MM baseline (which made $18,603 / 3
days on R3 dataset = $6,200/day). Adds a Mark 67 / Mark 49 directional
signal that biases the fair value upward whenever those Marks print.

----------------------------------------------------------------------
Strategy
----------------------------------------------------------------------
Layer 1 (baseline) — copied from R3 v4.trade_extract:
  fair = 0.70 * wall_mid + 0.30 * EMA(wall_mid, alpha=0.05)
  TAKE: cross any ask < fair - 1 or bid > fair + 1.
  MAKE: quote at floor(fair-1) / ceil(fair+1), inside best_bid+1 / best_ask-1.
  Inventory skew: 0.08 / position unit.

Layer 2 (NEW) — Mark 67 / Mark 49 signal:
  Each tick: signal *= SIGNAL_DECAY (geometric).
  When market_trades contains Mark 67 buy or Mark 49 sell of qty ≥ TRIGGER,
  signal += qty * SIGNAL_PER_LOT.
  fair_effective = fair + clamp(signal, ±MAX_FAIR_SHIFT)
  → biases the entire take + make logic upward when signal is hot.

Net effect: passive MM with a "lean long" tilt during signal windows.
We never cross spread on the signal; we just shift our quotes.
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

    # ---- Layer 2 (Mark 67/49 signal) ----
    TRIGGER_MIN_QTY = 3
    SIGNAL_PER_LOT = 0.05
    SIGNAL_DECAY = 0.9999      # ~10k tick half-life
    MAX_FAIR_SHIFT = 2.0

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

        # Compute fair_base = wall_mid * 0.7 + EMA * 0.3
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

        # Update signal (Layer 2)
        signal *= self.SIGNAL_DECAY
        for trade in state.market_trades.get(self.VE_SYM, []) or []:
            qty = int(trade.quantity)
            if qty < self.TRIGGER_MIN_QTY:
                continue
            if trade.buyer == "Mark 67":
                signal += qty * self.SIGNAL_PER_LOT
            elif trade.seller == "Mark 49":
                signal += qty * self.SIGNAL_PER_LOT

        bias = max(-self.MAX_FAIR_SHIFT, min(self.MAX_FAIR_SHIFT, signal))
        fair_eff = fair_base + bias

        # Take + make
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
        """Largest-volume bid level + largest-volume ask level, averaged."""
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
        # Cross any ask priced below fair - take_edge (buy).
        for ask in sorted(depth.sell_orders.keys()):
            if ask <= fair - self.TAKE_EDGE:
                avail = -depth.sell_orders[ask]
                qty = max(0, min(avail, self.VE_LIMIT - position))
                if qty > 0:
                    orders.append(Order(self.VE_SYM, ask, qty))
                    position += qty
        # Cross any bid priced above fair + take_edge (sell).
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

        # Inside the existing inside (avoid stepping outside):
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
