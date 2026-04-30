from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple
from collections import defaultdict, deque


class Trader:
    """
    Focused Final Round V4.

    Core products:
    1. MICROCHIP_CIRCLE
       - Momentum strategy.
       - No flatten rule.
       - Holds position until strong opposite signal.

    2. ROBOT_DISHES
       - Momentum strategy.
       - Has flatten rule.
       - Flattens when momentum disappears.

    Added candidate products:
    3. ROBOT_LAUNDRY
       - Relative value using ROBOT_DISHES as reference.

    4. SNACKPACK_VANILLA
       - Relative value using SNACKPACK_CHOCOLATE as reference.

    5. SNACKPACK_RASPBERRY
       - Relative value using SNACKPACK_PISTACHIO as reference.

    6. ROBOT_MOPPING
       - Slower momentum than before.

    7. MICROCHIP_TRIANGLE
       - Lower-edge mean reversion.
    """

    POSITION_LIMIT = 10
    MAX_HISTORY = 1200

    # ------------------------------------------------------------
    # 1. Core strategy: MICROCHIP_CIRCLE
    # No flatten. Hold until opposite strong signal.
    # product: (short_window, long_window, entry_edge)
    # ------------------------------------------------------------
    HOLD_MOMENTUM: Dict[str, Tuple[int, int, float]] = {
        "MICROCHIP_CIRCLE": (100, 200, 30),
    }

    # ------------------------------------------------------------
    # 2. Momentum with flatten.
    # Used for ROBOT_DISHES and improved ROBOT_MOPPING.
    # product: (short_window, long_window, entry_edge, exit_edge)
    # ------------------------------------------------------------
    EXIT_MOMENTUM: Dict[str, Tuple[int, int, float, float]] = {
        "ROBOT_DISHES": (5, 100, 120, 35),
        "ROBOT_MOPPING": (20, 100, 80, 25),
    }

    # ------------------------------------------------------------
    # 3. Single-product mean reversion with flatten.
    # Used for MICROCHIP_TRIANGLE.
    # product: (window, entry_edge, exit_edge)
    # ------------------------------------------------------------
    MEAN_REVERSION: Dict[str, Tuple[int, float, float]] = {
        "MICROCHIP_TRIANGLE": (1000, 100, 30),
    }

    # ------------------------------------------------------------
    # 4. Relative-value reference strategies.
    # fair_product = constant_sum - reference_mid
    # ------------------------------------------------------------
    ROBOT_LAUNDRY_REF_SUM = 19841
    ROBOT_LAUNDRY_EDGE = 250
    ROBOT_LAUNDRY_EXIT = 60

    VANILLA_REF_SUM = 19941
    VANILLA_EDGE = 50
    VANILLA_EXIT = 12

    RASPBERRY_REF_SUM = 19574
    RASPBERRY_EDGE = 80
    RASPBERRY_EXIT = 20

    def __init__(self):
        self.mid_history = defaultdict(lambda: deque(maxlen=self.MAX_HISTORY))

    # ------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------

    def _best_bid_ask(self, order_depth: OrderDepth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        bid_volume = order_depth.buy_orders[best_bid]
        ask_volume = -order_depth.sell_orders[best_ask]

        return best_bid, bid_volume, best_ask, ask_volume

    def _rolling_mean(self, product: str, window: int):
        hist = self.mid_history[product]

        if len(hist) < window:
            return None

        recent = list(hist)[-window:]
        return sum(recent) / window

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
        final_qty = min(qty, capacity)

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
        final_qty = min(qty, capacity)

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

    # ------------------------------------------------------------
    # Strategy 1: hold momentum
    # MICROCHIP_CIRCLE
    # ------------------------------------------------------------

    def _trade_hold_momentum(
        self,
        orders: Dict[str, List[Order]],
        virtual_pos: Dict[str, int],
        quote: Dict[str, Tuple[int, int, int, int]],
    ):
        for product, (short_window, long_window, entry_edge) in self.HOLD_MOMENTUM.items():
            if product not in quote:
                continue

            short_mean = self._rolling_mean(product, short_window)
            long_mean = self._rolling_mean(product, long_window)

            if short_mean is None or long_mean is None:
                continue

            signal = short_mean - long_mean
            best_bid, bid_volume, best_ask, ask_volume = quote[product]
            current_pos = virtual_pos.get(product, 0)

            if signal > entry_edge:
                target_pos = self.POSITION_LIMIT
            elif signal < -entry_edge:
                target_pos = -self.POSITION_LIMIT
            else:
                target_pos = current_pos

            self._rebalance_to_target(
                orders,
                virtual_pos,
                product,
                target_pos,
                best_bid,
                bid_volume,
                best_ask,
                ask_volume,
            )

    # ------------------------------------------------------------
    # Strategy 2: momentum with flatten
    # ROBOT_DISHES, ROBOT_MOPPING
    # ------------------------------------------------------------

    def _trade_exit_momentum(
        self,
        orders: Dict[str, List[Order]],
        virtual_pos: Dict[str, int],
        quote: Dict[str, Tuple[int, int, int, int]],
    ):
        for product, (short_window, long_window, entry_edge, exit_edge) in self.EXIT_MOMENTUM.items():
            if product not in quote:
                continue

            short_mean = self._rolling_mean(product, short_window)
            long_mean = self._rolling_mean(product, long_window)

            if short_mean is None or long_mean is None:
                continue

            signal = short_mean - long_mean
            best_bid, bid_volume, best_ask, ask_volume = quote[product]
            current_pos = virtual_pos.get(product, 0)

            if signal > entry_edge:
                target_pos = self.POSITION_LIMIT
            elif signal < -entry_edge:
                target_pos = -self.POSITION_LIMIT
            elif abs(signal) < exit_edge:
                target_pos = 0
            else:
                target_pos = current_pos

            self._rebalance_to_target(
                orders,
                virtual_pos,
                product,
                target_pos,
                best_bid,
                bid_volume,
                best_ask,
                ask_volume,
            )

    # ------------------------------------------------------------
    # Strategy 3: mean reversion with flatten
    # MICROCHIP_TRIANGLE
    # ------------------------------------------------------------

    def _trade_mean_reversion(
        self,
        orders: Dict[str, List[Order]],
        virtual_pos: Dict[str, int],
        quote: Dict[str, Tuple[int, int, int, int]],
        current_mid: Dict[str, float],
    ):
        for product, (window, entry_edge, exit_edge) in self.MEAN_REVERSION.items():
            if product not in quote or product not in current_mid:
                continue

            fair = self._rolling_mean(product, window)

            if fair is None:
                continue

            best_bid, bid_volume, best_ask, ask_volume = quote[product]
            mid = current_mid[product]
            current_pos = virtual_pos.get(product, 0)

            if best_ask < fair - entry_edge:
                target_pos = self.POSITION_LIMIT
            elif best_bid > fair + entry_edge:
                target_pos = -self.POSITION_LIMIT
            elif abs(mid - fair) < exit_edge:
                target_pos = 0
            else:
                target_pos = current_pos

            self._rebalance_to_target(
                orders,
                virtual_pos,
                product,
                target_pos,
                best_bid,
                bid_volume,
                best_ask,
                ask_volume,
            )

    # ------------------------------------------------------------
    # Strategy 4: relative value using one product as reference
    # ------------------------------------------------------------

    def _trade_reference_value(
        self,
        orders: Dict[str, List[Order]],
        virtual_pos: Dict[str, int],
        quote: Dict[str, Tuple[int, int, int, int]],
        current_mid: Dict[str, float],
        product: str,
        reference_product: str,
        ref_sum: float,
        entry_edge: float,
        exit_edge: float,
    ):
        if product not in quote:
            return

        if reference_product not in current_mid:
            return

        if product not in current_mid:
            return

        fair = ref_sum - current_mid[reference_product]

        best_bid, bid_volume, best_ask, ask_volume = quote[product]
        mid = current_mid[product]
        current_pos = virtual_pos.get(product, 0)

        if best_ask < fair - entry_edge:
            target_pos = self.POSITION_LIMIT

        elif best_bid > fair + entry_edge:
            target_pos = -self.POSITION_LIMIT

        elif abs(mid - fair) < exit_edge:
            target_pos = 0

        else:
            target_pos = current_pos

        self._rebalance_to_target(
            orders,
            virtual_pos,
            product,
            target_pos,
            best_bid,
            bid_volume,
            best_ask,
            ask_volume,
        )

    def _trade_all_reference_value(
        self,
        orders: Dict[str, List[Order]],
        virtual_pos: Dict[str, int],
        quote: Dict[str, Tuple[int, int, int, int]],
        current_mid: Dict[str, float],
    ):
        # ROBOT_LAUNDRY fair value from ROBOT_DISHES.
        self._trade_reference_value(
            orders=orders,
            virtual_pos=virtual_pos,
            quote=quote,
            current_mid=current_mid,
            product="ROBOT_LAUNDRY",
            reference_product="ROBOT_DISHES",
            ref_sum=self.ROBOT_LAUNDRY_REF_SUM,
            entry_edge=self.ROBOT_LAUNDRY_EDGE,
            exit_edge=self.ROBOT_LAUNDRY_EXIT,
        )

        # SNACKPACK_VANILLA fair value from SNACKPACK_CHOCOLATE.
        self._trade_reference_value(
            orders=orders,
            virtual_pos=virtual_pos,
            quote=quote,
            current_mid=current_mid,
            product="SNACKPACK_VANILLA",
            reference_product="SNACKPACK_CHOCOLATE",
            ref_sum=self.VANILLA_REF_SUM,
            entry_edge=self.VANILLA_EDGE,
            exit_edge=self.VANILLA_EXIT,
        )

        # SNACKPACK_RASPBERRY fair value from SNACKPACK_PISTACHIO.
        self._trade_reference_value(
            orders=orders,
            virtual_pos=virtual_pos,
            quote=quote,
            current_mid=current_mid,
            product="SNACKPACK_RASPBERRY",
            reference_product="SNACKPACK_PISTACHIO",
            ref_sum=self.RASPBERRY_REF_SUM,
            entry_edge=self.RASPBERRY_EDGE,
            exit_edge=self.RASPBERRY_EXIT,
        )

    # ------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------

    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {}
        virtual_pos = dict(state.position)

        quote = {}
        current_mid = {}

        active_products = {
            # Core traded products
            "MICROCHIP_CIRCLE",
            "ROBOT_DISHES",

            # Candidate traded products
            "ROBOT_MOPPING",
            "ROBOT_LAUNDRY",
            "MICROCHIP_TRIANGLE",
            "SNACKPACK_VANILLA",
            "SNACKPACK_RASPBERRY",

            # Reference-only products
            "SNACKPACK_CHOCOLATE",
            "SNACKPACK_PISTACHIO",
        }

        for product, depth in state.order_depths.items():
            if product not in active_products:
                continue

            best = self._best_bid_ask(depth)

            if best is None:
                continue

            best_bid, bid_volume, best_ask, ask_volume = best

            quote[product] = (best_bid, bid_volume, best_ask, ask_volume)
            current_mid[product] = (best_bid + best_ask) / 2

        # 1. MICROCHIP_CIRCLE: hold momentum
        self._trade_hold_momentum(orders, virtual_pos, quote)

        # 2. ROBOT_DISHES + ROBOT_MOPPING: momentum with flatten
        self._trade_exit_momentum(orders, virtual_pos, quote)

        # 3. MICROCHIP_TRIANGLE: mean reversion with lower edge
        self._trade_mean_reversion(orders, virtual_pos, quote, current_mid)

        # 4. ROBOT_LAUNDRY, SNACKPACK_VANILLA, SNACKPACK_RASPBERRY:
        # reference-value strategies
        self._trade_all_reference_value(orders, virtual_pos, quote, current_mid)

        # Update mid history after trading decisions.
        for product, mid in current_mid.items():
            self.mid_history[product].append(mid)

        conversions = 0
        traderData = ""

        return orders, conversions, traderData