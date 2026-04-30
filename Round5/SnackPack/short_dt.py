from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple, Optional
import json
import math


class Trader:
    """
    Serious all-five SNACKPACK statistical arbitrage strategy.

    Products:
        SNACKPACK_STRAWBERRY
        SNACKPACK_RASPBERRY
        SNACKPACK_VANILLA
        SNACKPACK_CHOCOLATE
        SNACKPACK_PISTACHIO

    Main logic:
        For every pair (A, B), learn rolling pair-sum:
            A_mid + B_mid ≈ constant

        Then estimate fair value of A using each other product B:
            fair_A_from_B = rolling_sum(A, B) - B_mid

        Average those fair estimates to get fair_A.

        If A is rich relative to fair_A -> sell A.
        If A is cheap relative to fair_A -> buy A.

    This is a generalized version of:
        fair_raspberry = 19574 - pistachio_mid
    """

    PRODUCTS = [
        "SNACKPACK_STRAWBERRY",
        "SNACKPACK_RASPBERRY",
        "SNACKPACK_VANILLA",
        "SNACKPACK_CHOCOLATE",
        "SNACKPACK_PISTACHIO",
    ]

    POSITION_LIMIT = 10

    # How fast pair-sum constants update.
    PAIR_SUM_ALPHA = 0.03

    # Residual z-score memory.
    RESID_HIST_LEN = 120

    # Entry thresholds.
    ENTRY_Z = 1.15
    STRONG_Z = 1.75
    EXIT_Z = 0.35

    # Additional absolute edge filter.
    # Prevents tiny statistical signals from trading.
    MIN_EDGE = 8.0

    # Size control.
    BASE_SIZE = 3
    STRONG_SIZE = 5
    MAX_TRADE = 5

    # Trend / pullback filter.
    FAST_ALPHA = 0.20
    SLOW_ALPHA = 0.05

    def _load(self, traderData: str) -> Dict:
        if traderData:
            try:
                data = json.loads(traderData)
                if isinstance(data, dict):
                    data.setdefault("pair_sum", {})
                    data.setdefault("resid_hist", {})
                    data.setdefault("ema_fast", {})
                    data.setdefault("ema_slow", {})
                    return data
            except Exception:
                pass

        return {
            "pair_sum": {},
            "resid_hist": {},
            "ema_fast": {},
            "ema_slow": {},
        }

    def _save(self, data: Dict) -> str:
        try:
            return json.dumps(data, separators=(",", ":"))[:50000]
        except Exception:
            return ""

    def _pair_key(self, a: str, b: str) -> str:
        if a < b:
            return a + "|" + b
        return b + "|" + a

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

    def _update_ema_map(self, mem: Dict, map_name: str, key: str, x: float, alpha: float) -> float:
        mp = mem.setdefault(map_name, {})
        prev = mp.get(key)

        if prev is None:
            val = x
        else:
            val = alpha * x + (1.0 - alpha) * prev

        mp[key] = val
        return val

    def _update_pair_sum(self, mem: Dict, a: str, b: str, mid_a: float, mid_b: float) -> float:
        key = self._pair_key(a, b)
        pair_sum_map = mem.setdefault("pair_sum", {})

        current_sum = mid_a + mid_b
        prev = pair_sum_map.get(key)

        if prev is None:
            updated = current_sum
        else:
            updated = self.PAIR_SUM_ALPHA * current_sum + (1.0 - self.PAIR_SUM_ALPHA) * prev

        pair_sum_map[key] = updated
        return updated

    def _get_pair_sum(self, mem: Dict, a: str, b: str) -> Optional[float]:
        key = self._pair_key(a, b)
        return mem.setdefault("pair_sum", {}).get(key)

    def _update_residual_stats(self, mem: Dict, product: str, resid: float):
        hist_map = mem.setdefault("resid_hist", {})
        hist = hist_map.setdefault(product, [])

        hist.append(resid)
        if len(hist) > self.RESID_HIST_LEN:
            hist.pop(0)

        mean = sum(hist) / len(hist)
        var = sum((x - mean) ** 2 for x in hist) / max(1, len(hist) - 1)
        std = math.sqrt(max(var, 1e-6))

        z = (resid - mean) / std
        return z, mean, std, len(hist)

    def _add_buy(
        self,
        orders: Dict[str, List[Order]],
        virtual_pos: Dict[str, int],
        product: str,
        price: int,
        qty: int,
    ):
        current_pos = virtual_pos.get(product, 0)
        capacity = self.POSITION_LIMIT - current_pos
        final_qty = min(qty, capacity, self.MAX_TRADE)

        if final_qty > 0:
            orders.setdefault(product, []).append(Order(product, price, final_qty))
            virtual_pos[product] = current_pos + final_qty

    def _add_sell(
        self,
        orders: Dict[str, List[Order]],
        virtual_pos: Dict[str, int],
        product: str,
        price: int,
        qty: int,
    ):
        current_pos = virtual_pos.get(product, 0)
        capacity = self.POSITION_LIMIT + current_pos
        final_qty = min(qty, capacity, self.MAX_TRADE)

        if final_qty > 0:
            orders.setdefault(product, []).append(Order(product, price, -final_qty))
            virtual_pos[product] = current_pos - final_qty

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

    def _target_from_signal(
        self,
        product: str,
        current_pos: int,
        z: float,
        resid: float,
        fast: float,
        slow: float,
    ) -> int:
        """
        Reversed version.

        Original:
            z > 0 / resid > 0 means product is rich -> sell.
            z < 0 / resid < 0 means product is cheap -> buy.

        Reversed:
            z > 0 / resid > 0 means product is strong -> buy.
            z < 0 / resid < 0 means product is weak -> sell.

        This tests whether the market has momentum/continuation instead of mean reversion.
        """

        trend = fast - slow

        target = current_pos

        positive_signal = z > self.ENTRY_Z and resid > self.MIN_EDGE
        negative_signal = z < -self.ENTRY_Z and resid < -self.MIN_EDGE

        very_positive = z > self.STRONG_Z and resid > self.MIN_EDGE
        very_negative = z < -self.STRONG_Z and resid < -self.MIN_EDGE

        # Exit when signal normalizes.
        if abs(z) < self.EXIT_Z or abs(resid) < self.MIN_EDGE * 0.5:
            return 0

        # Reversed logic:
        # Product above fair -> buy / follow strength.
        if positive_signal:
            # If short-term trend also agrees, go stronger.
            if trend > 0:
                target = self.POSITION_LIMIT if very_positive else self.POSITION_LIMIT // 2
            else:
                target = self.POSITION_LIMIT // 2

        # Product below fair -> sell / follow weakness.
        elif negative_signal:
            # If short-term trend also agrees, go stronger.
            if trend < 0:
                target = -self.POSITION_LIMIT if very_negative else -self.POSITION_LIMIT // 2
            else:
                target = -self.POSITION_LIMIT // 2

        return target

    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {}
        virtual_pos = dict(state.position)

        mem = self._load(state.traderData)

        quote = {}
        mid = {}

        # 1. Read quotes and mids for all available snack products.
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

        # Need at least two products to update pair relationships.
        available = [p for p in self.PRODUCTS if p in mid]

        if len(available) < 2:
            return orders, 0, self._save(mem)

        # 2. Update rolling pair-sum constants for every available pair.
        n = len(available)
        for i in range(n):
            for j in range(i + 1, n):
                a = available[i]
                b = available[j]
                self._update_pair_sum(mem, a, b, mid[a], mid[b])

        # 3. For each product, estimate fair value using the other products.
        signals = []

        for product in available:
            fair_estimates = []

            for other in available:
                if other == product:
                    continue

                pair_sum = self._get_pair_sum(mem, product, other)
                if pair_sum is None:
                    continue

                fair_from_other = pair_sum - mid[other]
                fair_estimates.append(fair_from_other)

            if not fair_estimates:
                continue

            fair = sum(fair_estimates) / len(fair_estimates)

            product_mid = mid[product]
            resid = product_mid - fair

            z, resid_mean, resid_std, hist_len = self._update_residual_stats(
                mem=mem,
                product=product,
                resid=resid,
            )

            fast = self._update_ema_map(
                mem=mem,
                map_name="ema_fast",
                key=product,
                x=product_mid,
                alpha=self.FAST_ALPHA,
            )

            slow = self._update_ema_map(
                mem=mem,
                map_name="ema_slow",
                key=product,
                x=product_mid,
                alpha=self.SLOW_ALPHA,
            )

            # Avoid trading too early before residual history is meaningful.
            if hist_len < 20:
                continue

            score = abs(z) * abs(resid)

            signals.append(
                {
                    "product": product,
                    "fair": fair,
                    "mid": product_mid,
                    "resid": resid,
                    "z": z,
                    "score": score,
                    "fast": fast,
                    "slow": slow,
                }
            )

        if not signals:
            return orders, 0, self._save(mem)

        # 4. Trade the strongest signals first.
        # This prevents all five products from firing too aggressively at once.
        signals.sort(key=lambda x: x["score"], reverse=True)

        max_products_to_trade = 3
        traded = 0

        for sig in signals:
            if traded >= max_products_to_trade:
                break

            product = sig["product"]

            if product not in quote:
                continue

            best_bid, bid_volume, best_ask, ask_volume = quote[product]
            current_pos = virtual_pos.get(product, 0)

            target_pos = self._target_from_signal(
                product=product,
                current_pos=current_pos,
                z=sig["z"],
                resid=sig["resid"],
                fast=sig["fast"],
                slow=sig["slow"],
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

            traded += 1

        conversions = 0
        traderData = self._save(mem)

        return orders, conversions, traderData