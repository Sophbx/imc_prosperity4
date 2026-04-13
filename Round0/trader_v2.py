from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import json

class Trader:
    """
    IMC Prosperity 4 - Tutorial Round Algorithm v2
    ===============================================
    Optimized version with:
      - Tighter EMERALDS spread (2 instead of 8)
      - Smoother TOMATOES EMA (alpha=0.18)
      - Stronger inventory skew (kappa=0.12)
      - TAKE edge threshold to filter marginal trades
      - End-of-simulation position flattening
      - Wider TOMATOES half-spread (4 instead of 3)
    """

    LIMITS = {"EMERALDS": 80, "TOMATOES": 80}

    # ---- EMERALDS config ----
    EME_SPREAD = 2  # Half-spread for market making (was 8, now tight)

    # ---- TOMATOES config ----
    TOMATO_ALPHA = 0.18        # EMA smoothing factor (was 0.3)
    TOMATO_HALF_SPREAD = 4     # MM half-spread (was 3)
    TOMATO_SKEW_KAPPA = 0.12   # Inventory skew coefficient (was 0.05)
    TOMATO_TAKE_EDGE = 1       # Min mispricing to TAKE (was 0)
    TOMATO_ENDGAME_TICK = 180000  # Start aggressive flattening
    TOMATO_ENDGAME_SKEW_MULT = 3  # Multiply kappa by this in endgame

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

        # Phase 1: TAKE mispriced orders
        if order_depth.sell_orders:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price < fair_value and buy_vol < max_buy:
                    available = -order_depth.sell_orders[ask_price]
                    qty = min(available, max_buy - buy_vol)
                    if qty > 0:
                        orders.append(Order(product, ask_price, qty))
                        buy_vol += qty

        if order_depth.buy_orders:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price > fair_value and sell_vol < max_sell:
                    available = order_depth.buy_orders[bid_price]
                    qty = min(available, max_sell - sell_vol)
                    if qty > 0:
                        orders.append(Order(product, bid_price, -qty))
                        sell_vol += qty

        # Phase 2: MAKE at tight spread (9998 / 10002)
        mm_bid_price = fair_value - self.EME_SPREAD
        mm_ask_price = fair_value + self.EME_SPREAD

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

        # ---- EMA fair value (smoother alpha=0.18) ----
        ema_key = "tomato_ema"
        if ema_key in trader_data:
            ema = self.TOMATO_ALPHA * mid_price + (1 - self.TOMATO_ALPHA) * trader_data[ema_key]
        else:
            ema = mid_price

        trader_data[ema_key] = ema
        fair_value = round(ema)

        # ---- Position limits ----
        max_buy = limit - position
        max_sell = limit + position

        orders: List[Order] = []
        buy_vol = 0
        sell_vol = 0

        # ---- Phase 1: TAKE with edge threshold ----
        # Only take if mispricing exceeds TAKE_EDGE to avoid adverse selection
        for ask_price in sorted(order_depth.sell_orders.keys()):
            if ask_price < fair_value - self.TOMATO_TAKE_EDGE and buy_vol < max_buy:
                available = -order_depth.sell_orders[ask_price]
                qty = min(available, max_buy - buy_vol)
                if qty > 0:
                    orders.append(Order(product, ask_price, qty))
                    buy_vol += qty

        for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if bid_price > fair_value + self.TOMATO_TAKE_EDGE and sell_vol < max_sell:
                available = order_depth.buy_orders[bid_price]
                qty = min(available, max_sell - sell_vol)
                if qty > 0:
                    orders.append(Order(product, bid_price, -qty))
                    sell_vol += qty

        # ---- Phase 2: Position-skewed market making ----
        # Stronger skew (kappa=0.12) for faster position unwinding
        # In endgame (last ~20K ticks), multiply skew to flatten before sim ends
        kappa = self.TOMATO_SKEW_KAPPA
        if state.timestamp >= self.TOMATO_ENDGAME_TICK:
            kappa *= self.TOMATO_ENDGAME_SKEW_MULT

        skew = -round(position * kappa)

        mm_bid_price = fair_value - self.TOMATO_HALF_SPREAD + skew
        mm_ask_price = fair_value + self.TOMATO_HALF_SPREAD + skew

        remaining_buy = max_buy - buy_vol
        remaining_sell = max_sell - sell_vol

        if remaining_buy > 0:
            orders.append(Order(product, mm_bid_price, remaining_buy))
        if remaining_sell > 0:
            orders.append(Order(product, mm_ask_price, -remaining_sell))

        return orders, trader_data
