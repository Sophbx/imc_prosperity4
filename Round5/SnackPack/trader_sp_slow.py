from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import json
import math


class Trader:
    """
    Low-frequency long-only extreme mean-reversion strategy.

    Main idea:
        Trade much less.
        Buy only when price is extremely low.
        Hold.
        Sell all only when price is clearly high.

    No shorting.
    """

    PRODUCTS = [
        "SNACKPACK_STRAWBERRY",
        "SNACKPACK_RASPBERRY",
        "SNACKPACK_VANILLA",
        "SNACKPACK_CHOCOLATE",
        "SNACKPACK_PISTACHIO",
    ]

    RASPBERRY = "SNACKPACK_RASPBERRY"
    VANILLA = "SNACKPACK_VANILLA"
    CHOCOLATE = "SNACKPACK_CHOCOLATE"
    STRAWBERRY = "SNACKPACK_STRAWBERRY"
    PISTACHIO = "SNACKPACK_PISTACHIO"

    POSITION_LIMIT = 10
    MAX_TRADE = 10

    HIST_LEN = 300
    MIN_HIST = 80

    FAST_ALPHA = 0.12
    SLOW_ALPHA = 0.025

    # Product-specific buy thresholds.
    # More negative = stricter / trades less.
    BUY_Z = {
        "SNACKPACK_STRAWBERRY": -2.0,
        "SNACKPACK_RASPBERRY": -2.5,
        "SNACKPACK_VANILLA": -2.6,
        "SNACKPACK_CHOCOLATE": -2.3,
        "SNACKPACK_PISTACHIO": -1.9,
    }

    # Sell/exit threshold.
    # We sell all when price has clearly recovered.
    SELL_Z = {
        "SNACKPACK_STRAWBERRY": 1.1,
        "SNACKPACK_RASPBERRY": 1.2,
        "SNACKPACK_VANILLA": 1.3,
        "SNACKPACK_CHOCOLATE": 1.2,
        "SNACKPACK_PISTACHIO": 1.0,
    }

    # Emergency stop: if we bought and the product keeps collapsing,
    # reduce damage. Keep loose because this strategy is meant to hold.
    STOP_Z = {
        "SNACKPACK_STRAWBERRY": -3.4,
        "SNACKPACK_RASPBERRY": -3.6,
        "SNACKPACK_VANILLA": -3.8,
        "SNACKPACK_CHOCOLATE": -3.5,
        "SNACKPACK_PISTACHIO": -3.3,
    }

    def _load(self, traderData: str) -> Dict:
        if traderData:
            try:
                data = json.loads(traderData)
                if isinstance(data, dict):
                    data.setdefault("hist", {})
                    data.setdefault("ema_fast", {})
                    data.setdefault("ema_slow", {})
                    return data
            except Exception:
                pass

        return {
            "hist": {},
            "ema_fast": {},
            "ema_slow": {},
        }

    def _save(self, data: Dict) -> str:
        try:
            return json.dumps(data, separators=(",", ":"))[:50000]
        except Exception:
            return ""

    def _best_bid_ask(self, depth: OrderDepth):
        if not depth.buy_orders or not depth.sell_orders:
            return None

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())

        bid_volume = depth.buy_orders[best_bid]
        ask_volume = -depth.sell_orders[best_ask]

        return best_bid, bid_volume, best_ask, ask_volume

    def _mid(self, depth: OrderDepth) -> Optional[float]:
        best = self._best_bid_ask(depth)
        if best is None:
            return None

        best_bid, _, best_ask, _ = best
        return (best_bid + best_ask) / 2.0

    def _wall_mid(self, depth: OrderDepth) -> Optional[float]:
        if not depth.buy_orders or not depth.sell_orders:
            return self._mid(depth)

        bid_wall = max(depth.buy_orders.items(), key=lambda x: abs(x[1]))[0]
        ask_wall = min(depth.sell_orders.items(), key=lambda x: abs(x[1]))[0]

        return (bid_wall + ask_wall) / 2.0

    def _update_hist(self, mem: Dict, product: str, x: float) -> List[float]:
        hist = mem.setdefault("hist", {}).setdefault(product, [])
        hist.append(x)

        if len(hist) > self.HIST_LEN:
            hist.pop(0)

        return hist

    def _update_ema(self, mem: Dict, map_name: str, product: str, x: float, alpha: float) -> float:
        mp = mem.setdefault(map_name, {})
        prev = mp.get(product)

        if prev is None:
            val = x
        else:
            val = alpha * x + (1.0 - alpha) * prev

        mp[product] = val
        return val

    def _mean_std(self, values: List[float]) -> Tuple[float, float]:
        mean = sum(values) / len(values)
        var = sum((x - mean) ** 2 for x in values) / max(1, len(values) - 1)
        std = math.sqrt(max(var, 1e-6))
        return mean, std

    def _imbalance(self, depth: OrderDepth) -> float:
        bid_items = sorted(depth.buy_orders.items(), reverse=True)[:3]
        ask_items = sorted(depth.sell_orders.items())[:3]

        bid_vol = sum(abs(v) for _, v in bid_items)
        ask_vol = sum(abs(v) for _, v in ask_items)

        total = bid_vol + ask_vol
        if total <= 0:
            return 0.0

        return (bid_vol - ask_vol) / total

    def _rebalance_to_target(
        self,
        orders: Dict[str, List[Order]],
        virtual_pos: Dict[str, int],
        product: str,
        target_pos: int,
        best_bid: int,
        bid_volume: int,
        best_ask: int,
        ask_volume: int,
    ):
        current_pos = virtual_pos.get(product, 0)
        delta = target_pos - current_pos

        if delta > 0:
            qty = min(delta, ask_volume, self.MAX_TRADE, self.POSITION_LIMIT - current_pos)
            if qty > 0:
                orders.setdefault(product, []).append(Order(product, best_ask, qty))
                virtual_pos[product] = current_pos + qty

        elif delta < 0:
            qty = min(-delta, bid_volume, self.MAX_TRADE, self.POSITION_LIMIT + current_pos)
            if qty > 0:
                orders.setdefault(product, []).append(Order(product, best_bid, -qty))
                virtual_pos[product] = current_pos - qty

    def _target_extreme_long_only(
        self,
        product: str,
        current_pos: int,
        z: float,
        observed: float,
        fast: float,
        slow: float,
        imb: float,
    ) -> int:
        """
        Long-only extreme strategy.

        Buy:
            only when price is extremely low.

        Sell:
            only when price has clearly recovered.

        No shorting.
        """

        buy_z = self.BUY_Z[product]
        sell_z = self.SELL_Z[product]
        stop_z = self.STOP_Z[product]

        trend = fast - slow

        # If currently flat, only buy at extreme low.
        if current_pos == 0:
            extremely_low = z <= buy_z

            # Avoid buying if the book is extremely bearish.
            # But do not make this too strict, or we may miss lows.
            book_not_disastrous = imb > -0.45

            # For Vanilla/Raspberry, avoid catching a strong falling knife.
            if product in [self.VANILLA, self.RASPBERRY]:
                trend_not_collapsing = trend > -8.0
            else:
                trend_not_collapsing = trend > -10.0

            if extremely_low and book_not_disastrous and trend_not_collapsing:
                return self.POSITION_LIMIT

            return 0

        # If already long, hold until clear recovery.
        if current_pos > 0:
            recovered = z >= sell_z

            # Also exit if price rises above slow EMA and z is positive.
            recovered_to_trend_mean = observed > slow and z > 0.4

            # Emergency stop if this is not a dip but a collapse.
            collapse_stop = z <= stop_z and trend < -10.0

            if recovered or recovered_to_trend_mean or collapse_stop:
                return 0

            return current_pos

        # Should not have short position, but if somehow short, flatten.
        if current_pos < 0:
            return 0

        return current_pos

    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {}
        virtual_pos = dict(state.position)

        mem = self._load(state.traderData)

        for product in self.PRODUCTS:
            depth = state.order_depths.get(product)
            if depth is None:
                continue

            best = self._best_bid_ask(depth)
            if best is None:
                continue

            best_bid, bid_volume, best_ask, ask_volume = best

            mid = self._mid(depth)
            wall = self._wall_mid(depth)

            if mid is None and wall is None:
                continue
            if wall is None:
                wall = mid
            if mid is None:
                mid = wall

            observed = 0.75 * mid + 0.25 * wall

            hist = self._update_hist(mem, product, observed)

            fast = self._update_ema(
                mem=mem,
                map_name="ema_fast",
                product=product,
                x=observed,
                alpha=self.FAST_ALPHA,
            )

            slow = self._update_ema(
                mem=mem,
                map_name="ema_slow",
                product=product,
                x=observed,
                alpha=self.SLOW_ALPHA,
            )

            if len(hist) < self.MIN_HIST:
                continue

            mean, std = self._mean_std(hist)
            z = (observed - mean) / std

            imb = self._imbalance(depth)
            current_pos = virtual_pos.get(product, 0)

            target_pos = self._target_extreme_long_only(
                product=product,
                current_pos=current_pos,
                z=z,
                observed=observed,
                fast=fast,
                slow=slow,
                imb=imb,
            )

            if target_pos == current_pos:
                continue

            self._rebalance_to_target(
                orders=orders,
                virtual_pos=virtual_pos,
                product=product,
                target_pos=target_pos,
                best_bid=best_bid,
                bid_volume=bid_volume,
                best_ask=best_ask,
                ask_volume=ask_volume,
            )

        return orders, 0, self._save(mem)