from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple
from collections import defaultdict, deque


class Trader:
    """
    Focused Final Round V3.

    Based on latest backtest:
    - ROBOT_DISHES improved after adding flatten logic.
    - MICROCHIP_CIRCLE got worse after adding flatten logic.

    Therefore:
    1. MICROCHIP_CIRCLE uses old momentum logic:
       if signal > entry_edge: go long
       if signal < -entry_edge: go short
       otherwise hold current position

    2. ROBOT_DISHES uses momentum with flatten:
       if signal > entry_edge: go long
       if signal < -entry_edge: go short
       if abs(signal) < exit_edge: flatten to 0
       otherwise hold current position
    """

    POSITION_LIMIT = 10
    MAX_HISTORY = 500

    # MICROCHIP_CIRCLE: no flatten
    # product: (short_window, long_window, entry_edge)
    HOLD_MOMENTUM: Dict[str, Tuple[int, int, float]] = {
        "MICROCHIP_CIRCLE": (100, 200, 30),
    }

    # ROBOT_DISHES: with flatten
    # product: (short_window, long_window, entry_edge, exit_edge)
    EXIT_MOMENTUM: Dict[str, Tuple[int, int, float, float]] = {
        "ROBOT_DISHES": (5, 100, 120, 35),
    }

    def __init__(self):
        self.mid_history = defaultdict(lambda: deque(maxlen=self.MAX_HISTORY))

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

    def _trade_hold_momentum(
        self,
        orders: Dict[str, List[Order]],
        virtual_pos: Dict[str, int],
        quote: Dict[str, Tuple[int, int, int, int]],
    ):
        """
        Used for MICROCHIP_CIRCLE.

        No flatten rule.
        Once it enters a position, it keeps the position until a strong opposite signal appears.
        """
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
                orders=orders,
                virtual_pos=virtual_pos,
                product=product,
                target_pos=target_pos,
                best_bid=best_bid,
                bid_volume=bid_volume,
                best_ask=best_ask,
                ask_volume=ask_volume,
            )

    def _trade_exit_momentum(
        self,
        orders: Dict[str, List[Order]],
        virtual_pos: Dict[str, int],
        quote: Dict[str, Tuple[int, int, int, int]],
    ):
        """
        Used for ROBOT_DISHES.

        Has flatten rule.
        If momentum disappears, close the position.
        """
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

        quote = {}
        current_mid = {}

        active_products = set(self.HOLD_MOMENTUM.keys()) | set(self.EXIT_MOMENTUM.keys())

        for product, depth in state.order_depths.items():
            if product not in active_products:
                continue

            best = self._best_bid_ask(depth)

            if best is None:
                continue

            best_bid, bid_volume, best_ask, ask_volume = best

            quote[product] = (best_bid, bid_volume, best_ask, ask_volume)
            current_mid[product] = (best_bid + best_ask) / 2

        self._trade_hold_momentum(orders, virtual_pos, quote)
        self._trade_exit_momentum(orders, virtual_pos, quote)

        # Update history after decision, so current timestamp is not used for its own signal.
        for product, mid in current_mid.items():
            self.mid_history[product].append(mid)

        conversions = 0
        traderData = ""

        return orders, conversions, traderData