from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple
from collections import defaultdict, deque
import jsonpickle
import math


class Trader:
    """
    Current good strategy + friend's MICROCHIP_SQUARE / MICROCHIP_RECTANGLE pair strategy.

    Existing strategy:
    1. MICROCHIP_CIRCLE
       - Momentum, no flatten.

    2. MICROCHIP_OVAL
       - Faster momentum, no flatten.

    3. ROBOT_MOPPING
       - Faster momentum, no flatten.

    4. ROBOT_DISHES
       - Momentum with flatten.

    Added friend's strategy:
    5. MICROCHIP_SQUARE vs MICROCHIP_RECTANGLE
       - Spread/z-score pair trading.
       - SQUARE is Y.
       - RECTANGLE is X.
       - spread = SQUARE_mid - ALPHA - BETA * RECTANGLE_mid.
    """

    POSITION_LIMIT = 10
    MAX_HISTORY = 500

    ENABLE_SQUARE_RECTANGLE_PAIR = True

    # ------------------------------------------------------------
    # Existing hold momentum strategy
    # ------------------------------------------------------------
    HOLD_MOMENTUM: Dict[str, Tuple[int, int, float]] = {
        "MICROCHIP_CIRCLE": (100, 200, 30),
        "MICROCHIP_OVAL": (50, 300, 15),
        "ROBOT_MOPPING": (50, 300, 25),
    }

    # ------------------------------------------------------------
    # Existing exit momentum strategy
    # ------------------------------------------------------------
    EXIT_MOMENTUM: Dict[str, Tuple[int, int, float, float]] = {
        "ROBOT_DISHES": (5, 100, 120, 35),
    }

    # ------------------------------------------------------------
    # Friend's SQUARE / RECTANGLE pair strategy constants
    # ------------------------------------------------------------
    PAIR_Y = "MICROCHIP_SQUARE"
    PAIR_X = "MICROCHIP_RECTANGLE"

    PAIR_POS_LIMITS = {
        "MICROCHIP_SQUARE": 10,
        "MICROCHIP_RECTANGLE": 10,
    }

    ALPHA = 32346.12
    BETA = -2.1473

    ENTRY_STABLE = 2.60
    ENTRY_UNSTABLE = 999.0
    EXIT_STABLE = 0.75
    EXIT_UNSTABLE = 0.75

    BASE_SIZE_STABLE = 1
    BASE_SIZE_UNSTABLE = 0

    MAX_CLIP_STABLE = 1
    MAX_CLIP_UNSTABLE = 1

    Z_WINDOW = 500
    VEL_LAG = 50
    VEL_THR = 0.60
    CRISIS_Z = 3.00

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

        if best_bid >= best_ask:
            return None

        bid_volume = order_depth.buy_orders[best_bid]
        ask_volume = -order_depth.sell_orders[best_ask]

        return best_bid, bid_volume, best_ask, ask_volume

    def _rolling_mean(self, product: str, window: int):
        hist = self.mid_history[product]

        if len(hist) < window:
            return None

        recent = list(hist)[-window:]
        return sum(recent) / window

    def _load_mem(self, state: TradingState):
        if state.traderData:
            try:
                mem = jsonpickle.decode(state.traderData)
                if isinstance(mem, dict):
                    return mem
            except Exception:
                return {}
        return {}

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
            orders.setdefault(product, []).append(Order(product, int(price), int(final_qty)))
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
            orders.setdefault(product, []).append(Order(product, int(price), -int(final_qty)))
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
    # MICROCHIP_CIRCLE / MICROCHIP_OVAL / ROBOT_MOPPING
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
                orders=orders,
                virtual_pos=virtual_pos,
                product=product,
                target_pos=target_pos,
                best_bid=best_bid,
                bid_volume=bid_volume,
                best_ask=best_ask,
                ask_volume=ask_volume,
            )

    # ------------------------------------------------------------
    # Strategy 2: exit momentum
    # ROBOT_DISHES
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
                orders=orders,
                virtual_pos=virtual_pos,
                product=product,
                target_pos=target_pos,
                best_bid=best_bid,
                bid_volume=bid_volume,
                best_ask=best_ask,
                ask_volume=ask_volume,
            )

    # ------------------------------------------------------------
    # Friend's pair strategy helpers
    # ------------------------------------------------------------

    def _compute_rolling_z(self, spread_hist, window):
        if len(spread_hist) < max(30, window // 5):
            return None

        recent = spread_hist[-window:]
        mu = sum(recent) / len(recent)

        var = sum((x - mu) ** 2 for x in recent) / max(1, len(recent) - 1)
        std = math.sqrt(var)

        if std <= 1e-9:
            return None

        return (spread_hist[-1] - mu) / std

    def _compute_z_velocity(self, spread_hist, window, lag):
        if len(spread_hist) < window + lag + 5:
            return 0.0

        current_z = self._compute_rolling_z(spread_hist, window)

        old_hist = spread_hist[:-lag]
        old_z = self._compute_rolling_z(old_hist, window)

        if current_z is None or old_z is None:
            return 0.0

        return current_z - old_z

    def _classify_pair_regime(self, z, z_vel):
        if abs(z) > self.CRISIS_Z and abs(z_vel) > self.VEL_THR:
            return "crisis"

        if abs(z_vel) > self.VEL_THR:
            return "unstable"

        return "stable"

    def _update_spread_direction(self, z, regime, current_dir):
        """
        spread_dir:
            +1 = long spread
            -1 = short spread
             0 = flat

        spread = SQUARE_mid - ALPHA - BETA * RECTANGLE_mid

        If z is high:
            spread is too expensive -> short spread.

        If z is low:
            spread is too cheap -> long spread.
        """

        if regime == "stable":
            entry_z = self.ENTRY_STABLE
            exit_z = self.EXIT_STABLE

        elif regime == "unstable":
            entry_z = self.ENTRY_UNSTABLE
            exit_z = self.EXIT_UNSTABLE

        else:
            return current_dir

        # Exit logic
        if current_dir != 0:
            if abs(z) < exit_z:
                return 0

            # If crossed to the opposite side, close.
            if current_dir == +1 and z > 0:
                return 0

            if current_dir == -1 and z < 0:
                return 0

            return current_dir

        # Entry logic
        if z > entry_z:
            return -1

        if z < -entry_z:
            return +1

        return 0

    def _compute_pair_targets(self, spread_dir, regime):
        if spread_dir == 0:
            return 0, 0

        if regime == "stable":
            base_size = self.BASE_SIZE_STABLE
        else:
            base_size = self.BASE_SIZE_UNSTABLE

        target_y = spread_dir * base_size

        # Since BETA < 0, the X leg has the same sign as Y.
        target_x = spread_dir * int(round(abs(self.BETA) * base_size))

        target_y = max(-self.PAIR_POS_LIMITS[self.PAIR_Y], min(self.PAIR_POS_LIMITS[self.PAIR_Y], target_y))
        target_x = max(-self.PAIR_POS_LIMITS[self.PAIR_X], min(self.PAIR_POS_LIMITS[self.PAIR_X], target_x))

        return target_y, target_x

    def _reduce_only_target(self, current_pos, proposed_target):
        """
        In crisis mode, avoid increasing exposure.
        Only let target move toward zero or smaller absolute exposure.
        """

        if current_pos == 0:
            return 0

        if abs(proposed_target) > abs(current_pos):
            return current_pos

        if current_pos * proposed_target < 0:
            return 0

        return proposed_target

    def _quote_toward_target_passive(
        self,
        orders: Dict[str, List[Order]],
        virtual_pos: Dict[str, int],
        product: str,
        best_bid: int,
        best_ask: int,
        target_pos: int,
        pos_limit: int,
        max_clip: int,
        offset: int = 1,
    ):
        """
        Friend's quoting style:
        - If need to buy, quote best_bid + offset.
        - If need to sell, quote best_ask - offset.

        This is more passive than hitting best ask / best bid.
        """

        if best_bid is None or best_ask is None:
            return

        if best_bid >= best_ask:
            return

        current_pos = virtual_pos.get(product, 0)
        gap = target_pos - current_pos

        # Need to buy.
        if gap > 0:
            buy_room = pos_limit - current_pos
            qty = min(gap, buy_room, max_clip)

            if qty <= 0:
                return

            price = best_bid + offset

            if price >= best_ask:
                price = best_ask - 1

            if price > best_bid:
                orders.setdefault(product, []).append(Order(product, int(price), int(qty)))
                virtual_pos[product] = current_pos + int(qty)

        # Need to sell.
        elif gap < 0:
            sell_room = pos_limit + current_pos
            qty = min(-gap, sell_room, max_clip)

            if qty <= 0:
                return

            price = best_ask - offset

            if price <= best_bid:
                price = best_bid + 1

            if price < best_ask:
                orders.setdefault(product, []).append(Order(product, int(price), -int(qty)))
                virtual_pos[product] = current_pos - int(qty)

    # ------------------------------------------------------------
    # Friend's pair strategy:
    # MICROCHIP_SQUARE / MICROCHIP_RECTANGLE
    # ------------------------------------------------------------

    def _trade_square_rectangle_pair(
        self,
        orders: Dict[str, List[Order]],
        virtual_pos: Dict[str, int],
        quote: Dict[str, Tuple[int, int, int, int]],
        current_mid: Dict[str, float],
        mem: Dict,
    ):
        if not self.ENABLE_SQUARE_RECTANGLE_PAIR:
            return

        if self.PAIR_Y not in quote or self.PAIR_X not in quote:
            return

        if self.PAIR_Y not in current_mid or self.PAIR_X not in current_mid:
            return

        y_mid = current_mid[self.PAIR_Y]
        x_mid = current_mid[self.PAIR_X]

        spread = y_mid - self.ALPHA - self.BETA * x_mid

        pair_mem = mem.setdefault("square_rectangle_pair", {})
        pair_mem.setdefault("spread_hist", [])
        pair_mem["spread_hist"].append(spread)

        max_len = self.Z_WINDOW + self.VEL_LAG + 10

        if len(pair_mem["spread_hist"]) > max_len:
            pair_mem["spread_hist"] = pair_mem["spread_hist"][-max_len:]

        if len(pair_mem["spread_hist"]) < 100:
            return

        z = self._compute_rolling_z(pair_mem["spread_hist"], self.Z_WINDOW)

        if z is None:
            return

        z_vel = self._compute_z_velocity(pair_mem["spread_hist"], self.Z_WINDOW, self.VEL_LAG)

        regime = self._classify_pair_regime(z, z_vel)

        current_spread_dir = pair_mem.get("spread_dir", 0)
        new_spread_dir = self._update_spread_direction(z, regime, current_spread_dir)
        pair_mem["spread_dir"] = new_spread_dir

        target_y, target_x = self._compute_pair_targets(new_spread_dir, regime)

        y_pos = int(virtual_pos.get(self.PAIR_Y, 0))
        x_pos = int(virtual_pos.get(self.PAIR_X, 0))

        # Crisis rule: reduce only.
        if regime == "crisis":
            target_y = self._reduce_only_target(y_pos, target_y)
            target_x = self._reduce_only_target(x_pos, target_x)

        if regime == "stable":
            max_clip = self.MAX_CLIP_STABLE
            offset = 1
        else:
            max_clip = self.MAX_CLIP_UNSTABLE
            offset = 1

        y_best_bid, _, y_best_ask, _ = quote[self.PAIR_Y]
        x_best_bid, _, x_best_ask, _ = quote[self.PAIR_X]

        self._quote_toward_target_passive(
            orders=orders,
            virtual_pos=virtual_pos,
            product=self.PAIR_Y,
            best_bid=y_best_bid,
            best_ask=y_best_ask,
            target_pos=target_y,
            pos_limit=self.PAIR_POS_LIMITS[self.PAIR_Y],
            max_clip=max_clip,
            offset=offset,
        )

        self._quote_toward_target_passive(
            orders=orders,
            virtual_pos=virtual_pos,
            product=self.PAIR_X,
            best_bid=x_best_bid,
            best_ask=x_best_ask,
            target_pos=target_x,
            pos_limit=self.PAIR_POS_LIMITS[self.PAIR_X],
            max_clip=max_clip,
            offset=offset,
        )

    # ------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------

    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {}
        virtual_pos = dict(state.position)

        mem = self._load_mem(state)

        quote = {}
        current_mid = {}

        active_products = {
            # Existing strategy
            "MICROCHIP_CIRCLE",
            "MICROCHIP_OVAL",
            "ROBOT_DISHES",
            "ROBOT_MOPPING",

            # Friend's pair strategy
            "MICROCHIP_SQUARE",
            "MICROCHIP_RECTANGLE",
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

        # Existing strategy
        self._trade_hold_momentum(orders, virtual_pos, quote)
        self._trade_exit_momentum(orders, virtual_pos, quote)

        # Friend's SQUARE / RECTANGLE pair strategy
        self._trade_square_rectangle_pair(
            orders=orders,
            virtual_pos=virtual_pos,
            quote=quote,
            current_mid=current_mid,
            mem=mem,
        )

        # Update mid history after trading decisions.
        for product, mid in current_mid.items():
            self.mid_history[product].append(mid)

        conversions = 0
        traderData = jsonpickle.encode(mem)

        return orders, conversions, traderData