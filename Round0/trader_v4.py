from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import json

class Trader:
    """
    IMC Prosperity 4 - Tutorial Round Algorithm v4
    ===============================================
    Fixes the mid-simulation dip caused by accumulating large TOMATOES
    positions during sustained price drops.

    Changes from v3:
      - TOMATOES: combined linear+quadratic skew for meaningful skew at all levels
      - TOMATOES: TAKE soft cap — stop aggressive position building beyond ±20
      - TOMATOES: adaptive EMA alpha — faster tracking when position is large
      - TOMATOES: wider outer spread for better adverse selection protection
    """

    LIMITS = {"EMERALDS": 80, "TOMATOES": 80}

    # ---- EMERALDS config ----
    EME_SPREAD = 2
    EME_SKEW_KAPPA = 0.03

    # ---- TOMATOES config ----
    TOMATO_ALPHA_BASE = 0.18       # Base EMA alpha
    TOMATO_ALPHA_MAX = 0.5         # Max alpha when position is extreme
    TOMATO_HALF_SPREAD_1 = 3       # Inner level spread
    TOMATO_HALF_SPREAD_2 = 6       # Outer level spread (wider for protection)
    TOMATO_INNER_FRAC = 0.4        # Fraction at inner level
    TOMATO_LINEAR_KAPPA = 0.08     # Linear skew component
    TOMATO_QUAD_KAPPA = 0.002      # Quadratic skew component
    TOMATO_TAKE_SOFT_LIMIT = 20    # Beyond this, only allow unwind TAKEs
    TOMATO_TAKE_EDGE = 1           # Min edge for build-side TAKE
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
    # EMERALDS STRATEGY
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

        # Phase 1: TAKE (with unwind at fair value)
        if order_depth.sell_orders:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                should_take = ask_price < fair_value or (ask_price == fair_value and position < 0)
                if should_take and buy_vol < max_buy:
                    available = -order_depth.sell_orders[ask_price]
                    if position < 0 and ask_price == fair_value:
                        qty = min(available, max_buy - buy_vol, -position - buy_vol)
                    else:
                        qty = min(available, max_buy - buy_vol)
                    if qty > 0:
                        orders.append(Order(product, ask_price, qty))
                        buy_vol += qty

        if order_depth.buy_orders:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
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

        # Phase 2: MAKE with skew
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
    # TOMATOES STRATEGY
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

        # ---- Adaptive EMA: faster alpha when position is large ----
        # When we have a big position, price is likely moving against us.
        # Speed up EMA to track the move faster and stop chasing.
        abs_pos = abs(position)
        pos_ratio = abs_pos / limit  # 0 to 1
        alpha = self.TOMATO_ALPHA_BASE + (self.TOMATO_ALPHA_MAX - self.TOMATO_ALPHA_BASE) * pos_ratio

        ema_key = "tomato_ema"
        if ema_key in trader_data:
            ema = alpha * mid_price + (1 - alpha) * trader_data[ema_key]
        else:
            ema = mid_price

        trader_data[ema_key] = ema
        fair_value = round(ema)

        max_buy = limit - position
        max_sell = limit + position

        orders: List[Order] = []
        buy_vol = 0
        sell_vol = 0

        # ---- Phase 1: TAKE with soft position cap ----
        # When position exceeds soft limit, only allow unwind TAKEs
        soft_limit = self.TOMATO_TAKE_SOFT_LIMIT

        for ask_price in sorted(order_depth.sell_orders.keys()):
            # Buying: is this building or unwinding?
            is_unwind = position < 0
            is_build = position >= 0

            if is_build and abs_pos >= soft_limit:
                # Already significantly long, skip aggressive buys
                break

            edge_needed = 0 if is_unwind else self.TOMATO_TAKE_EDGE
            if ask_price < fair_value - edge_needed and buy_vol < max_buy:
                available = -order_depth.sell_orders[ask_price]
                qty = min(available, max_buy - buy_vol)
                if qty > 0:
                    orders.append(Order(product, ask_price, qty))
                    buy_vol += qty

        for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
            is_unwind = position > 0
            is_build = position <= 0

            if is_build and abs_pos >= soft_limit:
                # Already significantly short, skip aggressive sells
                break

            edge_needed = 0 if is_unwind else self.TOMATO_TAKE_EDGE
            if bid_price > fair_value + edge_needed and sell_vol < max_sell:
                available = order_depth.buy_orders[bid_price]
                qty = min(available, max_sell - sell_vol)
                if qty > 0:
                    orders.append(Order(product, bid_price, -qty))
                    sell_vol += qty

        # ---- Phase 2: Multi-level market making with linear+quadratic skew ----
        lin_kappa = self.TOMATO_LINEAR_KAPPA
        quad_kappa = self.TOMATO_QUAD_KAPPA
        if state.timestamp >= self.TOMATO_ENDGAME_TICK:
            lin_kappa *= self.TOMATO_ENDGAME_SKEW_MULT
            quad_kappa *= self.TOMATO_ENDGAME_SKEW_MULT

        # Combined skew: linear for moderate positions, quadratic kicks in at extremes
        # pos=20: skew = -(0.08*20 + 0.002*20*20) = -(1.6 + 0.8) = -2.4 -> -2
        # pos=40: skew = -(0.08*40 + 0.002*40*40) = -(3.2 + 3.2) = -6.4 -> -6
        # pos=80: skew = -(0.08*80 + 0.002*80*80) = -(6.4 + 12.8) = -19.2 -> -19
        skew = -round(lin_kappa * position + quad_kappa * position * abs(position))

        remaining_buy = max_buy - buy_vol
        remaining_sell = max_sell - sell_vol

        # Level 1: inner quotes
        inner_buy = max(1, round(remaining_buy * self.TOMATO_INNER_FRAC))
        inner_sell = max(1, round(remaining_sell * self.TOMATO_INNER_FRAC))

        bid_1 = fair_value - self.TOMATO_HALF_SPREAD_1 + skew
        ask_1 = fair_value + self.TOMATO_HALF_SPREAD_1 + skew

        if remaining_buy > 0:
            orders.append(Order(product, bid_1, min(inner_buy, remaining_buy)))
        if remaining_sell > 0:
            orders.append(Order(product, ask_1, -min(inner_sell, remaining_sell)))

        # Level 2: outer quotes
        outer_buy = remaining_buy - min(inner_buy, remaining_buy)
        outer_sell = remaining_sell - min(inner_sell, remaining_sell)

        bid_2 = fair_value - self.TOMATO_HALF_SPREAD_2 + skew
        ask_2 = fair_value + self.TOMATO_HALF_SPREAD_2 + skew

        if outer_buy > 0:
            orders.append(Order(product, bid_2, outer_buy))
        if outer_sell > 0:
            orders.append(Order(product, ask_2, -outer_sell))

        return orders, trader_data
