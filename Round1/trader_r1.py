from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Optional, Tuple
import json
import math


class Trader:
    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM": 80,
        "INTARIAN_PEPPER_ROOT": 80,
    }

    # INTARIAN_PEPPER_ROOT is extremely close to a linear upward drift in the
    # training data: fair ~= anchor + 0.001 * timestamp
    #. 
    PEPPER_SLOPE = 0.001
    PEPPER_ENTRY_PREMIUM_EARLY = 12
    PEPPER_ENTRY_PREMIUM_LATE = 6
    PEPPER_PASSIVE_SIZE = 20
    PEPPER_SCALP_EDGE = 7
    PEPPER_SCALP_SIZE = 10
    PEPPER_ANCHOR_ALPHA = 0.05

    # ASH_COATED_OSMIUM looks close to stationary around a moving fair value.
    ASH_HISTORY_LEN = 40
    ASH_TAKE_EDGE = 3
    ASH_QUOTE_EDGE = 2
    ASH_ORDER_SIZE = 12
    ASH_INVENTORY_SKEW = 0.05  # reservation price shift per unit of inventory

    def run(self, state: TradingState):
        data = self._load_data(state.traderData)
        result: Dict[str, List[Order]] = {}

        for product, order_depth in state.order_depths.items():
            if product == "INTARIAN_PEPPER_ROOT":
                result[product] = self._trade_pepper_root(state, order_depth, data)
            elif product == "ASH_COATED_OSMIUM":
                result[product] = self._trade_ash_osmium(state, order_depth, data)

        trader_data = self._dump_data(data)
        conversions = 0
        return result, conversions, trader_data

    def _trade_pepper_root(
        self,
        state: TradingState,
        order_depth: OrderDepth,
        data: Dict,
    ) -> List[Order]:
        product = "INTARIAN_PEPPER_ROOT"
        limit = self.POSITION_LIMITS[product]
        position = state.position.get(product, 0)
        orders: List[Order] = []

        best_bid, best_ask = self._best_bid_ask(order_depth)
        mid = self._mid_price(order_depth)
        if mid is None:
            return orders

        anchor_obs = mid - self.PEPPER_SLOPE * state.timestamp
        anchor = data.get("pepper_anchor")
        if anchor is None or state.timestamp == 0:
            anchor = anchor_obs
        else:
            anchor = (1.0 - self.PEPPER_ANCHOR_ALPHA) * float(anchor) + self.PEPPER_ANCHOR_ALPHA * anchor_obs
        data["pepper_anchor"] = anchor

        fair = anchor + self.PEPPER_SLOPE * state.timestamp
        early = state.timestamp <= 5_000
        entry_premium = self.PEPPER_ENTRY_PREMIUM_EARLY if early else self.PEPPER_ENTRY_PREMIUM_LATE

        # Core view: keep a large long inventory because the fair value drifts upward.
        target_position = limit

        if position < target_position:
            buy_capacity = target_position - position
            max_buy_price = fair + entry_premium
            filled = self._buy_through(
                product=product,
                order_depth=order_depth,
                max_price=max_buy_price,
                quantity=buy_capacity,
                orders=orders,
            )
            position += filled

        # Optional small scalp when the book is unusually rich versus trend fair.
        # Keep the strategy net long by never selling below 40 inventory.
        if position > 40:
            sellable = min(self.PEPPER_SCALP_SIZE, position - 40)
            filled = self._sell_through(
                product=product,
                order_depth=order_depth,
                min_price=fair + self.PEPPER_SCALP_EDGE,
                quantity=sellable,
                orders=orders,
            )
            position -= filled

        # If we are still below target, rest a passive bid inside the spread.
        if position < target_position and best_bid is not None:
            quote_bid = min(best_bid + 1, math.floor(fair + 4))
            if best_ask is None or quote_bid < best_ask:
                quote_qty = min(self.PEPPER_PASSIVE_SIZE, target_position - position)
                if quote_qty > 0:
                    orders.append(Order(product, int(quote_bid), int(quote_qty)))

        return orders

    def _trade_ash_osmium(
        self,
        state: TradingState,
        order_depth: OrderDepth,
        data: Dict,
    ) -> List[Order]:
        product = "ASH_COATED_OSMIUM"
        limit = self.POSITION_LIMITS[product]
        position = state.position.get(product, 0)
        orders: List[Order] = []

        best_bid, best_ask = self._best_bid_ask(order_depth)
        mid = self._mid_price(order_depth)
        if mid is None:
            return orders

        ash_mids = data.get("ash_mids", [])
        ash_mids.append(mid)
        if len(ash_mids) > self.ASH_HISTORY_LEN:
            ash_mids = ash_mids[-self.ASH_HISTORY_LEN :]
        data["ash_mids"] = ash_mids

        fair = sum(ash_mids) / len(ash_mids)
        reservation_price = fair - self.ASH_INVENTORY_SKEW * position

        # Take clearly stale quotes first.
        buy_capacity = limit - position
        filled_buy = self._buy_through(
            product=product,
            order_depth=order_depth,
            max_price=math.floor(reservation_price - self.ASH_TAKE_EDGE),
            quantity=buy_capacity,
            orders=orders,
        )
        position += filled_buy

        sell_capacity = limit + position
        filled_sell = self._sell_through(
            product=product,
            order_depth=order_depth,
            min_price=math.ceil(reservation_price + self.ASH_TAKE_EDGE),
            quantity=sell_capacity,
            orders=orders,
        )
        position -= filled_sell

        buy_capacity = limit - position
        sell_capacity = limit + position

        # Provide passive liquidity inside the spread with an inventory-aware skew.
        if best_bid is not None and best_ask is not None and best_bid < best_ask:
            bid_quote = min(best_bid + 1, math.floor(reservation_price - self.ASH_QUOTE_EDGE))
            ask_quote = max(best_ask - 1, math.ceil(reservation_price + self.ASH_QUOTE_EDGE))

            if bid_quote < ask_quote:
                buy_qty = min(self.ASH_ORDER_SIZE, buy_capacity)
                sell_qty = min(self.ASH_ORDER_SIZE, sell_capacity)

                if buy_qty > 0:
                    orders.append(Order(product, int(bid_quote), int(buy_qty)))
                if sell_qty > 0:
                    orders.append(Order(product, int(ask_quote), int(-sell_qty)))

        return orders

    def _buy_through(
        self,
        product: str,
        order_depth: OrderDepth,
        max_price: float,
        quantity: int,
        orders: List[Order],
    ) -> int:
        remaining = int(quantity)
        filled = 0
        for ask_price in sorted(order_depth.sell_orders.keys()):
            ask_volume = abs(int(order_depth.sell_orders[ask_price]))
            if ask_price > max_price or remaining <= 0:
                break
            take = min(remaining, ask_volume)
            if take > 0:
                orders.append(Order(product, int(ask_price), int(take)))
                filled += take
                remaining -= take
        return filled

    def _sell_through(
        self,
        product: str,
        order_depth: OrderDepth,
        min_price: float,
        quantity: int,
        orders: List[Order],
    ) -> int:
        remaining = int(quantity)
        filled = 0
        for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
            bid_volume = abs(int(order_depth.buy_orders[bid_price]))
            if bid_price < min_price or remaining <= 0:
                break
            take = min(remaining, bid_volume)
            if take > 0:
                orders.append(Order(product, int(bid_price), int(-take)))
                filled += take
                remaining -= take
        return filled

    def _best_bid_ask(self, order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def _mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        best_bid, best_ask = self._best_bid_ask(order_depth)
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        if best_bid is not None:
            return float(best_bid)
        if best_ask is not None:
            return float(best_ask)
        return None

    def _load_data(self, trader_data: str) -> Dict:
        if not trader_data:
            return {"ash_mids": [], "pepper_anchor": None}
        try:
            data = json.loads(trader_data)
            if "ash_mids" not in data:
                data["ash_mids"] = []
            if "pepper_anchor" not in data:
                data["pepper_anchor"] = None
            return data
        except Exception:
            return {"ash_mids": [], "pepper_anchor": None}

    def _dump_data(self, data: Dict) -> str:
        # Keep traderData compact.
        compact = {
            "ash_mids": data.get("ash_mids", [])[-self.ASH_HISTORY_LEN :],
            "pepper_anchor": data.get("pepper_anchor"),
        }
        return json.dumps(compact, separators=(",", ":"))