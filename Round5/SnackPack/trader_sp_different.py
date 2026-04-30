from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import json
import math


class Trader:
    PRODUCTS = [
        "SNACKPACK_STRAWBERRY",
        "SNACKPACK_RASPBERRY",
        "SNACKPACK_VANILLA",
        "SNACKPACK_CHOCOLATE",
        "SNACKPACK_PISTACHIO",
    ]

    STRAWBERRY = "SNACKPACK_STRAWBERRY"
    RASPBERRY = "SNACKPACK_RASPBERRY"
    VANILLA = "SNACKPACK_VANILLA"
    CHOCOLATE = "SNACKPACK_CHOCOLATE"
    PISTACHIO = "SNACKPACK_PISTACHIO"

    POSITION_LIMIT = 10
    MAX_TRADE = 4

    HIST_LEN = 220
    MIN_HIST = 35

    FAST_ALPHA = 0.20
    SLOW_ALPHA = 0.045

    RANGE_LEN = 120

    def _load(self, traderData: str) -> Dict:
        if traderData:
            try:
                data = json.loads(traderData)
                if isinstance(data, dict):
                    data.setdefault("hist", {})
                    data.setdefault("ema_fast", {})
                    data.setdefault("ema_slow", {})
                    data.setdefault("prev_fast", {})
                    data.setdefault("prev_slow", {})
                    return data
            except Exception:
                pass

        return {
            "hist": {},
            "ema_fast": {},
            "ema_slow": {},
            "prev_fast": {},
            "prev_slow": {},
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

    def _update_ema_pair(self, mem: Dict, product: str, x: float) -> Tuple[float, float, float, float]:
        fast_map = mem.setdefault("ema_fast", {})
        slow_map = mem.setdefault("ema_slow", {})
        prev_fast_map = mem.setdefault("prev_fast", {})
        prev_slow_map = mem.setdefault("prev_slow", {})

        old_fast = fast_map.get(product)
        old_slow = slow_map.get(product)

        if old_fast is None:
            fast = x
        else:
            fast = self.FAST_ALPHA * x + (1.0 - self.FAST_ALPHA) * old_fast

        if old_slow is None:
            slow = x
        else:
            slow = self.SLOW_ALPHA * x + (1.0 - self.SLOW_ALPHA) * old_slow

        prev_fast_map[product] = old_fast if old_fast is not None else fast
        prev_slow_map[product] = old_slow if old_slow is not None else slow

        fast_map[product] = fast
        slow_map[product] = slow

        return fast, slow, prev_fast_map[product], prev_slow_map[product]

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

    def _clamp_buy(self, virtual_pos: Dict[str, int], product: str, desired: int) -> int:
        current_pos = virtual_pos.get(product, 0)
        capacity = self.POSITION_LIMIT - current_pos
        return max(0, min(desired, capacity, self.MAX_TRADE))

    def _clamp_sell(self, virtual_pos: Dict[str, int], product: str, desired: int) -> int:
        current_pos = virtual_pos.get(product, 0)
        capacity = self.POSITION_LIMIT + current_pos
        return max(0, min(desired, capacity, self.MAX_TRADE))

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
            qty = self._clamp_buy(virtual_pos, product, min(delta, ask_volume))
            if qty > 0:
                orders.setdefault(product, []).append(Order(product, best_ask, qty))
                virtual_pos[product] = current_pos + qty

        elif delta < 0:
            qty = self._clamp_sell(virtual_pos, product, min(-delta, bid_volume))
            if qty > 0:
                orders.setdefault(product, []).append(Order(product, best_bid, -qty))
                virtual_pos[product] = current_pos - qty

    # ------------------------------------------------------------------
    # 1. VANILLA: trend-following Dual Thrust
    # ------------------------------------------------------------------
    def _target_vanilla(
        self,
        current_pos: int,
        observed: float,
        z: float,
        upper: float,
        lower: float,
        fast: float,
        slow: float,
        prev_fast: float,
        prev_slow: float,
        imb: float,
    ) -> int:
        trend = fast - slow
        prev_trend = prev_fast - prev_slow
        trend_slope = trend - prev_trend

        uptrend = trend > 2.0 and trend_slope >= -0.5 and observed > slow
        downtrend = trend < -2.0 and trend_slope <= 0.5 and observed < slow

        # Breakout continuation, not mean reversion.
        buy_breakout = observed > upper and uptrend and imb > -0.25
        sell_breakout = observed < lower and downtrend and imb < 0.25

        # Exit if trend weakens or flips.
        if current_pos > 0 and (trend < 0.5 or observed < fast):
            return 0

        if current_pos < 0 and (trend > -0.5 or observed > fast):
            return 0

        if buy_breakout:
            if z > 2.4:
                return 5
            return 8

        if sell_breakout:
            if z < -2.4:
                return -5
            return -8

        return current_pos

    # ------------------------------------------------------------------
    # 2. PISTACHIO: conservative mean reversion
    # ------------------------------------------------------------------
    def _target_pistachio(
        self,
        current_pos: int,
        observed: float,
        z: float,
        upper: float,
        lower: float,
        fast: float,
        slow: float,
        imb: float,
    ) -> int:
        trend = fast - slow

        # Pistachio looked most range-like, but use conservative entries.
        buy_signal = (
            z < -1.35
            and observed <= lower + 2.0
            and imb > -0.20
            and trend > -2.5
        )

        sell_signal = (
            z > 1.35
            and observed >= upper - 2.0
            and imb < 0.20
            and trend < 2.5
        )

        if abs(z) < 0.35:
            return 0

        if buy_signal:
            return 4

        if sell_signal:
            return -4

        return current_pos

    # ------------------------------------------------------------------
    # 3. STRAWBERRY: reversal-confirmed mean reversion
    # ------------------------------------------------------------------
    def _target_strawberry(
        self,
        current_pos: int,
        observed: float,
        z: float,
        upper: float,
        lower: float,
        fast: float,
        slow: float,
        prev_fast: float,
        prev_slow: float,
        imb: float,
    ) -> int:
        trend = fast - slow
        prev_trend = prev_fast - prev_slow
        trend_slope = trend - prev_trend

        fast_turning_up = fast > prev_fast and trend_slope > -0.2
        fast_turning_down = fast < prev_fast and trend_slope < 0.2

        # Only buy low after some sign of upward turn.
        buy_signal = (
            z < -1.10
            and observed <= lower + 4.0
            and imb > -0.15
            and fast_turning_up
            and trend > -4.0
        )

        # Only sell high after some sign of downward turn.
        sell_signal = (
            z > 1.10
            and observed >= upper - 4.0
            and imb < 0.15
            and fast_turning_down
            and trend < 4.0
        )

        if abs(z) < 0.30:
            return 0

        if buy_signal:
            return 4

        if sell_signal:
            return -4

        return current_pos

    # ------------------------------------------------------------------
    # 4. RASPBERRY: trend-protected mean reversion
    # ------------------------------------------------------------------
    def _target_raspberry(
        self,
        current_pos: int,
        observed: float,
        z: float,
        upper: float,
        lower: float,
        fast: float,
        slow: float,
        prev_fast: float,
        prev_slow: float,
        imb: float,
    ) -> int:
        trend = fast - slow
        prev_trend = prev_fast - prev_slow
        trend_slope = trend - prev_trend

        strong_downtrend = trend < -3.0 and observed < slow and trend_slope < 0.5
        strong_uptrend = trend > 3.0 and observed > slow and trend_slope > -0.5

        # If caught against regime, flatten.
        if current_pos > 0 and strong_downtrend:
            return 0

        if current_pos < 0 and strong_uptrend:
            return 0

        # Mean reversion, but avoid catching a falling knife.
        buy_signal = (
            z < -1.25
            and observed <= lower + 3.0
            and imb > -0.10
            and not strong_downtrend
            and fast >= prev_fast
        )

        sell_signal = (
            z > 1.25
            and observed >= upper - 3.0
            and imb < 0.10
            and not strong_uptrend
            and fast <= prev_fast
        )

        # If the trend is strongly one-way, trade continuation lightly.
        continuation_sell = (
            strong_downtrend
            and observed < lower
            and imb < 0.20
        )

        continuation_buy = (
            strong_uptrend
            and observed > upper
            and imb > -0.20
        )

        if abs(z) < 0.35:
            return 0

        if buy_signal:
            return 4

        if sell_signal:
            return -4

        if continuation_sell:
            return -3

        if continuation_buy:
            return 3

        return current_pos

    # ------------------------------------------------------------------
    # 5. CHOCOLATE: conservative / almost disabled
    # ------------------------------------------------------------------
    def _target_chocolate(
        self,
        current_pos: int,
        observed: float,
        z: float,
        upper: float,
        lower: float,
        fast: float,
        slow: float,
        prev_fast: float,
        prev_slow: float,
        imb: float,
    ) -> int:
        trend = fast - slow
        prev_trend = prev_fast - prev_slow
        trend_slope = trend - prev_trend

        uptrend = trend > 3.0 and trend_slope > -0.3 and observed > slow
        downtrend = trend < -3.0 and trend_slope < 0.3 and observed < slow

        # Chocolate performed badly with active logic, so keep it tiny.
        if current_pos > 0 and not uptrend:
            return 0

        if current_pos < 0 and not downtrend:
            return 0

        if uptrend and observed > upper and imb > 0.0 and z < 2.2:
            return 2

        if downtrend and observed < lower and imb < 0.0 and z > -2.2:
            return -2

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

            # Wall-mid is still useful for microstructure, but use a blend
            # so the signal is not too noisy.
            observed = 0.65 * mid + 0.35 * wall

            hist = self._update_hist(mem, product, observed)
            fast, slow, prev_fast, prev_slow = self._update_ema_pair(mem, product, observed)

            if len(hist) < self.MIN_HIST:
                continue

            mean, std = self._mean_std(hist)
            z = (observed - mean) / std

            recent = hist[-self.RANGE_LEN:]
            recent_open = recent[0]
            recent_high = max(recent)
            recent_low = min(recent)
            recent_range = max(recent_high - recent_low, 1.0)

            # Default bands.
            upper = recent_open + 0.35 * recent_range
            lower = recent_open - 0.35 * recent_range

            imb = self._imbalance(depth)
            current_pos = virtual_pos.get(product, 0)

            if product == self.VANILLA:
                target_pos = self._target_vanilla(
                    current_pos, observed, z, upper, lower,
                    fast, slow, prev_fast, prev_slow, imb
                )

            elif product == self.PISTACHIO:
                target_pos = self._target_pistachio(
                    current_pos, observed, z, upper, lower,
                    fast, slow, imb
                )

            elif product == self.STRAWBERRY:
                target_pos = self._target_strawberry(
                    current_pos, observed, z, upper, lower,
                    fast, slow, prev_fast, prev_slow, imb
                )

            elif product == self.RASPBERRY:
                target_pos = self._target_raspberry(
                    current_pos, observed, z, upper, lower,
                    fast, slow, prev_fast, prev_slow, imb
                )

            elif product == self.CHOCOLATE:
                target_pos = self._target_chocolate(
                    current_pos, observed, z, upper, lower,
                    fast, slow, prev_fast, prev_slow, imb
                )

            else:
                target_pos = current_pos

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