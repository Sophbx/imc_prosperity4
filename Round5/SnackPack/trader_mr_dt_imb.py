from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import json
import math


class Trader:
    """
    All-five SNACKPACK strategy.

    Special logic:
        SNACKPACK_CHOCOLATE:
            Trend-aware pullback strategy.
            Avoids shorting into strong uptrends and avoids buying into strong downtrends.

    Other four products:
        Strict hybrid:
            Mean reversion + Dual Thrust confirmation + imbalance filter.
    """

    PRODUCTS = [
        "SNACKPACK_STRAWBERRY",
        "SNACKPACK_RASPBERRY",
        "SNACKPACK_VANILLA",
        "SNACKPACK_CHOCOLATE",
        "SNACKPACK_PISTACHIO",
    ]

    CHOCOLATE = "SNACKPACK_CHOCOLATE"

    POSITION_LIMIT = 10

    # Shared history settings
    HIST_LEN = 180
    MIN_HIST = 25

    # Default hybrid signal for non-chocolate products
    ENTRY_Z = 1.15
    STRONG_Z = 1.75
    EXIT_Z = 0.35

    RANGE_LEN = 100
    K_UP = 0.35
    K_DOWN = 0.35
    BAND_TOL = 2.0

    BUY_IMB_MIN = -0.35
    SELL_IMB_MAX = 0.35

    # Chocolate-specific settings
    CHOC_TREND_THRESHOLD = 1.5
    CHOC_PULLBACK_TOL = 3.0
    CHOC_TAKE_PROFIT_Z = 1.8
    CHOC_RANGE_ENTRY_Z = 1.2

    # Execution
    BASE_TARGET = 5
    STRONG_TARGET = 10
    MAX_TRADE = 4

    # EMA
    FAST_ALPHA = 0.18
    SLOW_ALPHA = 0.04

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

        bid_wall_price = max(depth.buy_orders.items(), key=lambda x: abs(x[1]))[0]
        ask_wall_price = min(depth.sell_orders.items(), key=lambda x: abs(x[1]))[0]

        return (bid_wall_price + ask_wall_price) / 2.0

    def _update_hist(self, mem: Dict, product: str, x: float) -> List[float]:
        hist = mem.setdefault("hist", {}).setdefault(product, [])
        hist.append(x)

        if len(hist) > self.HIST_LEN:
            hist.pop(0)

        return hist

    def _update_ema(
        self,
        mem: Dict,
        map_name: str,
        product: str,
        x: float,
        alpha: float,
    ) -> float:
        ema_map = mem.setdefault(map_name, {})
        prev = ema_map.get(product)

        if prev is None:
            ema = x
        else:
            ema = alpha * x + (1.0 - alpha) * prev

        ema_map[product] = ema
        return ema

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

    def _add_buy(
        self,
        orders: Dict[str, List[Order]],
        virtual_pos: Dict[str, int],
        product: str,
        price: int,
        qty: int,
    ):
        final_qty = self._clamp_buy(virtual_pos, product, qty)

        if final_qty > 0:
            orders.setdefault(product, []).append(Order(product, price, final_qty))
            virtual_pos[product] = virtual_pos.get(product, 0) + final_qty

    def _add_sell(
        self,
        orders: Dict[str, List[Order]],
        virtual_pos: Dict[str, int],
        product: str,
        price: int,
        qty: int,
    ):
        final_qty = self._clamp_sell(virtual_pos, product, qty)

        if final_qty > 0:
            orders.setdefault(product, []).append(Order(product, price, -final_qty))
            virtual_pos[product] = virtual_pos.get(product, 0) - final_qty

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
            self._add_buy(
                orders=orders,
                virtual_pos=virtual_pos,
                product=product,
                price=best_ask,
                qty=min(delta, ask_volume),
            )

        elif delta < 0:
            self._add_sell(
                orders=orders,
                virtual_pos=virtual_pos,
                product=product,
                price=best_bid,
                qty=min(-delta, bid_volume),
            )

    def _target_default_hybrid(
        self,
        current_pos: int,
        z: float,
        observed: float,
        upper_band: float,
        lower_band: float,
        imbalance: float,
        fast: float,
        slow: float,
    ) -> int:
        """
        Default strategy for non-chocolate snack products.

        Mean reversion:
            z high  -> sell
            z low   -> buy

        Dual Thrust confirmation:
            sell only near upper band
            buy only near lower band

        Imbalance filter:
            avoid buying into very bearish book
            avoid selling into very bullish book
        """

        trend = fast - slow

        buy_signal = (
            z < -self.ENTRY_Z
            and observed <= lower_band + self.BAND_TOL
            and imbalance >= self.BUY_IMB_MIN
        )

        sell_signal = (
            z > self.ENTRY_Z
            and observed >= upper_band - self.BAND_TOL
            and imbalance <= self.SELL_IMB_MAX
        )

        strong_buy = (
            z < -self.STRONG_Z
            and observed <= lower_band + self.BAND_TOL
            and imbalance > 0.0
        )

        strong_sell = (
            z > self.STRONG_Z
            and observed >= upper_band - self.BAND_TOL
            and imbalance < 0.0
        )

        if abs(z) < self.EXIT_Z:
            return 0

        target = current_pos

        if buy_signal:
            # Do not aggressively buy while price is strongly falling.
            if trend < -2.0 and not strong_buy:
                target = max(current_pos, 0)
            else:
                target = self.STRONG_TARGET if strong_buy else self.BASE_TARGET

        elif sell_signal:
            # Do not aggressively sell while price is strongly rising.
            if trend > 2.0 and not strong_sell:
                target = min(current_pos, 0)
            else:
                target = -self.STRONG_TARGET if strong_sell else -self.BASE_TARGET

        return target

    def _target_chocolate(
        self,
        current_pos: int,
        z: float,
        observed: float,
        mean: float,
        upper_band: float,
        lower_band: float,
        imbalance: float,
        fast: float,
        slow: float,
    ) -> int:
        """
        Chocolate-specific logic.

        CHOCOLATE seems to trend more strongly.
        Therefore:
            In uptrend:
                do not short local highs.
                buy pullbacks instead.

            In downtrend:
                do not buy local lows.
                sell rebounds instead.

            In range:
                use stricter mean reversion.
        """

        trend = fast - slow

        uptrend = trend > self.CHOC_TREND_THRESHOLD and observed > slow
        downtrend = trend < -self.CHOC_TREND_THRESHOLD and observed < slow

        # If currently against a clear trend, flatten.
        if current_pos < 0 and uptrend:
            return 0

        if current_pos > 0 and downtrend:
            return 0

        # -----------------------------
        # 1. Uptrend regime
        # -----------------------------
        if uptrend:
            pullback_buy = (
                observed <= fast + self.CHOC_PULLBACK_TOL
                and observed >= mean - 0.5 * self.CHOC_PULLBACK_TOL
                and imbalance > -0.30
                and z < 1.20
            )

            if pullback_buy:
                return self.BASE_TARGET

            # Take profit if it becomes very stretched upward.
            if current_pos > 0 and z > self.CHOC_TAKE_PROFIT_Z:
                return 0

            # Do not short just because price is high in an uptrend.
            return current_pos

        # -----------------------------
        # 2. Downtrend regime
        # -----------------------------
        if downtrend:
            rebound_sell = (
                observed >= fast - self.CHOC_PULLBACK_TOL
                and observed <= mean + 0.5 * self.CHOC_PULLBACK_TOL
                and imbalance < 0.30
                and z > -1.20
            )

            if rebound_sell:
                return -self.BASE_TARGET

            # Take profit if it becomes very stretched downward.
            if current_pos < 0 and z < -self.CHOC_TAKE_PROFIT_Z:
                return 0

            # Do not buy just because price is low in a downtrend.
            return current_pos

        # -----------------------------
        # 3. Range regime
        # -----------------------------
        range_buy = (
            z < -self.CHOC_RANGE_ENTRY_Z
            and observed <= lower_band + self.BAND_TOL
            and imbalance >= self.BUY_IMB_MIN
        )

        range_sell = (
            z > self.CHOC_RANGE_ENTRY_Z
            and observed >= upper_band - self.BAND_TOL
            and imbalance <= self.SELL_IMB_MAX
        )

        if range_buy:
            return self.BASE_TARGET

        if range_sell:
            return -self.BASE_TARGET

        if abs(z) < self.EXIT_Z:
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

            # Main observed price.
            observed = wall

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

            recent = hist[-self.RANGE_LEN:]
            recent_open = recent[0]
            recent_high = max(recent)
            recent_low = min(recent)
            recent_range = max(recent_high - recent_low, 1.0)

            upper_band = recent_open + self.K_UP * recent_range
            lower_band = recent_open - self.K_DOWN * recent_range

            imb = self._imbalance(depth)

            current_pos = virtual_pos.get(product, 0)

            # Product-specific dispatch.
            if product == self.CHOCOLATE:
                target_pos = self._target_chocolate(
                    current_pos=current_pos,
                    z=z,
                    observed=observed,
                    mean=mean,
                    upper_band=upper_band,
                    lower_band=lower_band,
                    imbalance=imb,
                    fast=fast,
                    slow=slow,
                )
            else:
                target_pos = self._target_default_hybrid(
                    current_pos=current_pos,
                    z=z,
                    observed=observed,
                    upper_band=upper_band,
                    lower_band=lower_band,
                    imbalance=imb,
                    fast=fast,
                    slow=slow,
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

        conversions = 0
        traderData = self._save(mem)

        return orders, conversions, traderData