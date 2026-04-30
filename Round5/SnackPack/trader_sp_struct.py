from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import json
import math


class Trader:
    """
    Merged Snackpack Strategy

    RASPBERRY:
        Structural relationship from friend's profitable strategy:
            fair_raspberry = 19574 - mid_price(SNACKPACK_PISTACHIO)

        If Raspberry is cheap relative to fair:
            buy Raspberry

        If Raspberry is expensive relative to fair:
            sell Raspberry

        If Raspberry returns close to fair:
            flatten

    Other 4 products:
        Slow/simple long-only extreme mean reversion:
            buy only when price is very low
            hold
            sell when recovered
            no shorting
    """

    STRAWBERRY = "SNACKPACK_STRAWBERRY"
    RASPBERRY = "SNACKPACK_RASPBERRY"
    VANILLA = "SNACKPACK_VANILLA"
    CHOCOLATE = "SNACKPACK_CHOCOLATE"
    PISTACHIO = "SNACKPACK_PISTACHIO"

    PRODUCTS = [
        STRAWBERRY,
        RASPBERRY,
        VANILLA,
        CHOCOLATE,
        PISTACHIO,
    ]

    SIMPLE_PRODUCTS = [
        STRAWBERRY,
        VANILLA,
        CHOCOLATE,
        PISTACHIO,
    ]

    POSITION_LIMIT = 10
    MAX_TRADE = 10

    # ---------------------------------------------------------
    # Raspberry structural edge
    # ---------------------------------------------------------
    RASPBERRY_REF_SUM = 19574.0
    RASPBERRY_EDGE = 80.0
    RASPBERRY_STRONG_EDGE = 130.0
    RASPBERRY_EXIT = 20.0

    # ---------------------------------------------------------
    # Slow/simple long-only logic for the other four products
    # ---------------------------------------------------------
    HIST_LEN = 300
    MIN_HIST = 80

    FAST_ALPHA = 0.12
    SLOW_ALPHA = 0.025

    BUY_Z = {
        STRAWBERRY: -2.0,
        VANILLA: -2.6,
        CHOCOLATE: -2.4,
        PISTACHIO: -1.9,
    }

    SELL_Z = {
        STRAWBERRY: 1.1,
        VANILLA: 1.3,
        CHOCOLATE: 1.2,
        PISTACHIO: 1.0,
    }

    STOP_Z = {
        STRAWBERRY: -3.4,
        VANILLA: -3.8,
        CHOCOLATE: -3.5,
        PISTACHIO: -3.3,
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
            qty = min(
                delta,
                ask_volume,
                self.MAX_TRADE,
                self.POSITION_LIMIT - current_pos,
            )

            if qty > 0:
                orders.setdefault(product, []).append(Order(product, best_ask, qty))
                virtual_pos[product] = current_pos + qty

        elif delta < 0:
            qty = min(
                -delta,
                bid_volume,
                self.MAX_TRADE,
                self.POSITION_LIMIT + current_pos,
            )

            if qty > 0:
                orders.setdefault(product, []).append(Order(product, best_bid, -qty))
                virtual_pos[product] = current_pos - qty

    # ---------------------------------------------------------
    # 1. Raspberry structural relationship logic
    # ---------------------------------------------------------
    def _trade_raspberry_structural(
        self,
        orders: Dict[str, List[Order]],
        virtual_pos: Dict[str, int],
        quote: Dict[str, Tuple[int, int, int, int]],
        mid: Dict[str, float],
    ):
        if self.RASPBERRY not in quote:
            return

        if self.PISTACHIO not in mid:
            return

        best_bid, bid_volume, best_ask, ask_volume = quote[self.RASPBERRY]

        raspberry_mid = mid[self.RASPBERRY]
        pistachio_mid = mid[self.PISTACHIO]

        fair_raspberry = self.RASPBERRY_REF_SUM - pistachio_mid

        current_pos = virtual_pos.get(self.RASPBERRY, 0)
        target_pos = current_pos

        # Positive means Raspberry is expensive.
        mispricing = raspberry_mid - fair_raspberry

        # Exit first when close to fair.
        if abs(mispricing) < self.RASPBERRY_EXIT:
            target_pos = 0

        # Raspberry cheap: buy.
        elif best_ask < fair_raspberry - self.RASPBERRY_EDGE:
            if best_ask < fair_raspberry - self.RASPBERRY_STRONG_EDGE:
                target_pos = self.POSITION_LIMIT
            else:
                target_pos = self.POSITION_LIMIT // 2

        # Raspberry expensive: sell.
        elif best_bid > fair_raspberry + self.RASPBERRY_EDGE:
            if best_bid > fair_raspberry + self.RASPBERRY_STRONG_EDGE:
                target_pos = -self.POSITION_LIMIT
            else:
                target_pos = -self.POSITION_LIMIT // 2

        self._rebalance_to_target(
            orders=orders,
            virtual_pos=virtual_pos,
            product=self.RASPBERRY,
            target_pos=target_pos,
            best_bid=best_bid,
            bid_volume=bid_volume,
            best_ask=best_ask,
            ask_volume=ask_volume,
        )

    # ---------------------------------------------------------
    # 2. Simple slow long-only logic for non-Raspberry products
    # ---------------------------------------------------------
    def _target_simple_long_only(
        self,
        product: str,
        current_pos: int,
        z: float,
        observed: float,
        fast: float,
        slow: float,
        imb: float,
    ) -> int:
        buy_z = self.BUY_Z[product]
        sell_z = self.SELL_Z[product]
        stop_z = self.STOP_Z[product]

        trend = fast - slow

        # No shorting for simple products.
        if current_pos < 0:
            return 0

        # Flat: only buy at extreme low.
        if current_pos == 0:
            extremely_low = z <= buy_z
            book_ok = imb > -0.45

            # Avoid catching a strong falling knife.
            if product == self.VANILLA:
                trend_ok = trend > -8.0
            elif product == self.CHOCOLATE:
                trend_ok = trend > -9.0
            else:
                trend_ok = trend > -10.0

            if extremely_low and book_ok and trend_ok:
                return self.POSITION_LIMIT

            return 0

        # Long: hold until clear recovery.
        if current_pos > 0:
            recovered_by_z = z >= sell_z
            recovered_to_mean = observed > slow and z > 0.4

            # Emergency stop only in extreme collapse.
            collapse_stop = z <= stop_z and trend < -10.0

            if recovered_by_z or recovered_to_mean or collapse_stop:
                return 0

            return current_pos

        return current_pos

    def _trade_simple_product(
        self,
        orders: Dict[str, List[Order]],
        virtual_pos: Dict[str, int],
        mem: Dict,
        product: str,
        depth: OrderDepth,
        quote: Tuple[int, int, int, int],
    ):
        best_bid, bid_volume, best_ask, ask_volume = quote

        mid = self._mid(depth)
        wall = self._wall_mid(depth)

        if mid is None and wall is None:
            return

        if wall is None:
            wall = mid

        if mid is None:
            mid = wall

        # Conservative observed price.
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
            return

        mean, std = self._mean_std(hist)
        z = (observed - mean) / std

        imb = self._imbalance(depth)
        current_pos = virtual_pos.get(product, 0)

        target_pos = self._target_simple_long_only(
            product=product,
            current_pos=current_pos,
            z=z,
            observed=observed,
            fast=fast,
            slow=slow,
            imb=imb,
        )

        if target_pos == current_pos:
            return

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

    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {}
        virtual_pos = dict(state.position)

        mem = self._load(state.traderData)

        quote: Dict[str, Tuple[int, int, int, int]] = {}
        mid: Dict[str, float] = {}

        # Read quotes.
        for product in self.PRODUCTS:
            depth = state.order_depths.get(product)
            if depth is None:
                continue

            best = self._best_bid_ask(depth)
            if best is None:
                continue

            best_bid, bid_volume, best_ask, ask_volume = best
            quote[product] = (best_bid, bid_volume, best_ask, ask_volume)
            mid[product] = (best_bid + best_ask) / 2.0

        # 1. Raspberry uses structural relationship.
        self._trade_raspberry_structural(
            orders=orders,
            virtual_pos=virtual_pos,
            quote=quote,
            mid=mid,
        )

        # 2. Other four use simple slow long-only logic.
        for product in self.SIMPLE_PRODUCTS:
            if product not in quote:
                continue

            depth = state.order_depths.get(product)
            if depth is None:
                continue

            self._trade_simple_product(
                orders=orders,
                virtual_pos=virtual_pos,
                mem=mem,
                product=product,
                depth=depth,
                quote=quote[product],
            )

        conversions = 0
        traderData = self._save(mem)

        return orders, conversions, traderData