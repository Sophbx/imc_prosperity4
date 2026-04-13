from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import json

class Trader:
    """
    IMC Prosperity 4 - Tutorial Round Algorithm v3
    ===============================================
    Changes from v2:
      - EMERALDS: added inventory skew to center position
      - EMERALDS: TAKE at fair value when carrying inventory (unwind)
      - TOMATOES: multi-level quoting (2 levels) for more fills
      - TOMATOES: non-linear (quadratic) skew for stronger pressure at extremes
      - TOMATOES: slightly more aggressive TAKE edge (0 for unwind side, 1 for build side)
    """

    LIMITS = {"EMERALDS": 80, "TOMATOES": 80}

    # ---- EMERALDS config ----
    EME_SPREAD = 2
    EME_SKEW_KAPPA = 0.03  # Light inventory skew for EMERALDS

    # ---- TOMATOES config ----
    TOMATO_ALPHA = 0.18
    TOMATO_HALF_SPREAD_1 = 3   # Inner level: tighter, smaller size
    TOMATO_HALF_SPREAD_2 = 5   # Outer level: wider, remaining size
    TOMATO_INNER_FRAC = 0.4    # Fraction of capacity at inner level
    TOMATO_SKEW_KAPPA = 0.0015 # Quadratic skew coefficient
    TOMATO_TAKE_EDGE = 1       # Min mispricing to TAKE (build side)
    TOMATO_ENDGAME_TICK = 180000
    TOMATO_ENDGAME_SKEW_MULT = 3

    def bid(self):
        return 15

    def run(self, state: TradingState):
        trader_data = {}
        if state.traderData:
            try:
                trader_data = json.loads(state.traderData)
            except (json.JSONDecodeError, TypeError):
                trader_data = {}

        result = {}

        for product in state.order_depths:
            if product == "EMERALDS":
                result[product] = self.trade_emeralds(state)
            elif product == "TOMATOES":
                result[product], trader_data = self.trade_tomatoes(
                    state, trader_data
                )
            else:
                result[product] = []

        traderData = json.dumps(trader_data)
        conversions = 0
        return result, conversions, traderData

    # ================================================================
    # EMERALDS STRATEGY — with inventory skew + unwind TAKE
    # ================================================================
    def trade_emeralds(self, state: TradingState) -> List[Order]:
        product = "EMERALDS"
        fair_value = 10000
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)
        limit = self.LIMITS[product]

        max_buy = limit - position
        max_sell = limit + position

        orders: List[Order] = []
        buy_vol = 0
        sell_vol = 0

        # Phase 1: TAKE mispriced orders
        # Now also takes AT fair value when we have inventory to unwind
        if order_depth.sell_orders:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                # Buy below fair value always; buy AT fair value only if short (unwind)
                should_take = ask_price < fair_value or (ask_price == fair_value and position < 0)
                if should_take and buy_vol < max_buy:
                    available = -order_depth.sell_orders[ask_price]
                    if position < 0 and ask_price == fair_value:
                        # Only unwind up to flat
                        qty = min(available, max_buy - buy_vol, -position - buy_vol)
                    else:
                        qty = min(available, max_buy - buy_vol)
                    if qty > 0:
                        orders.append(Order(product, ask_price, qty))
                        buy_vol += qty

        if order_depth.buy_orders:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                # Sell above fair value always; sell AT fair value only if long (unwind)
                should_take = bid_price > fair_value or (bid_price == fair_value and position > 0)
                if should_take and sell_vol < max_sell:
                    available = order_depth.buy_orders[bid_price]
                    if position > 0 and bid_price == fair_value:
                        qty = min(available, max_sell - sell_vol, position - sell_vol)
                    else:
                        qty = min(available, max_sell - sell_vol)
                    if qty > 0:
                        orders.append(Order(product, bid_price, -qty))
                        sell_vol += qty

        # Phase 2: MAKE with light inventory skew
        skew = -round(position * self.EME_SKEW_KAPPA)
        mm_bid_price = fair_value - self.EME_SPREAD + skew
        mm_ask_price = fair_value + self.EME_SPREAD + skew

        remaining_buy = max_buy - buy_vol
        remaining_sell = max_sell - sell_vol

        if remaining_buy > 0:
            orders.append(Order(product, mm_bid_price, remaining_buy))
        if remaining_sell > 0:
            orders.append(Order(product, mm_ask_price, -remaining_sell))

        return orders

    # ================================================================
    # TOMATOES STRATEGY — multi-level + quadratic skew
    # ================================================================
    def trade_tomatoes(
        self, state: TradingState, trader_data: dict
    ) -> tuple:
        product = "TOMATOES"
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)
        limit = self.LIMITS[product]

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return [], trader_data

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2

        # ---- EMA fair value ----
        ema_key = "tomato_ema"
        if ema_key in trader_data:
            ema = self.TOMATO_ALPHA * mid_price + (1 - self.TOMATO_ALPHA) * trader_data[ema_key]
        else:
            ema = mid_price

        trader_data[ema_key] = ema
        fair_value = round(ema)

        max_buy = limit - position
        max_sell = limit + position

        orders: List[Order] = []
        buy_vol = 0
        sell_vol = 0

        # ---- Phase 1: TAKE with asymmetric edge ----
        # Unwind side (reduces |position|): TAKE_EDGE = 0 (take at fair value)
        # Build side (increases |position|): TAKE_EDGE = 1 (need real edge)
        for ask_price in sorted(order_depth.sell_orders.keys()):
            # Buying: unwind side if short, build side if long/flat
            edge_needed = 0 if position < 0 else self.TOMATO_TAKE_EDGE
            if ask_price < fair_value - edge_needed and buy_vol < max_buy:
                available = -order_depth.sell_orders[ask_price]
                qty = min(available, max_buy - buy_vol)
                if qty > 0:
                    orders.append(Order(product, ask_price, qty))
                    buy_vol += qty

        for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
            # Selling: unwind side if long, build side if short/flat
            edge_needed = 0 if position > 0 else self.TOMATO_TAKE_EDGE
            if bid_price > fair_value + edge_needed and sell_vol < max_sell:
                available = order_depth.buy_orders[bid_price]
                qty = min(available, max_sell - sell_vol)
                if qty > 0:
                    orders.append(Order(product, bid_price, -qty))
                    sell_vol += qty

        # ---- Phase 2: Multi-level position-skewed market making ----
        # Quadratic skew: stronger at extreme positions
        # skew = -sign(pos) * kappa * pos^2
        kappa = self.TOMATO_SKEW_KAPPA
        if state.timestamp >= self.TOMATO_ENDGAME_TICK:
            kappa *= self.TOMATO_ENDGAME_SKEW_MULT

        skew = -round(kappa * position * abs(position))

        remaining_buy = max_buy - buy_vol
        remaining_sell = max_sell - sell_vol

        # Level 1: inner quotes (tighter spread, smaller size)
        inner_buy = max(1, round(remaining_buy * self.TOMATO_INNER_FRAC))
        inner_sell = max(1, round(remaining_sell * self.TOMATO_INNER_FRAC))

        bid_1 = fair_value - self.TOMATO_HALF_SPREAD_1 + skew
        ask_1 = fair_value + self.TOMATO_HALF_SPREAD_1 + skew

        if remaining_buy > 0:
            orders.append(Order(product, bid_1, min(inner_buy, remaining_buy)))
        if remaining_sell > 0:
            orders.append(Order(product, ask_1, -min(inner_sell, remaining_sell)))

        # Level 2: outer quotes (wider spread, remaining size)
        outer_buy = remaining_buy - min(inner_buy, remaining_buy)
        outer_sell = remaining_sell - min(inner_sell, remaining_sell)

        bid_2 = fair_value - self.TOMATO_HALF_SPREAD_2 + skew
        ask_2 = fair_value + self.TOMATO_HALF_SPREAD_2 + skew

        if outer_buy > 0:
            orders.append(Order(product, bid_2, outer_buy))
        if outer_sell > 0:
            orders.append(Order(product, ask_2, -outer_sell))

        return orders, trader_data
