from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import json

class Trader:
    """
    IMC Prosperity 4 - Tutorial Round
    Base: original trader.py with EME spread=6
    Focus: improve TOMATOES prediction & inventory management
    """

    LIMITS = {"EMERALDS": 80, "TOMATOES": 80}

    # EMERALDS: spread=6 (sweet spot per testing)
    EME_SPREAD = 6

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
    # EMERALDS — same as original, just spread=6
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

        # Phase 1: TAKE
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

        # Phase 2: MAKE at spread=6
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
    # TOMATOES — same as original baseline for now
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

        # EMA fair value (alpha=0.3, same as original)
        alpha = 0.3
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

        # Phase 1: TAKE
        for ask_price in sorted(order_depth.sell_orders.keys()):
            if ask_price < fair_value and buy_vol < max_buy:
                available = -order_depth.sell_orders[ask_price]
                qty = min(available, max_buy - buy_vol)
                if qty > 0:
                    orders.append(Order(product, ask_price, qty))
                    buy_vol += qty

        for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if bid_price > fair_value and sell_vol < max_sell:
                available = order_depth.buy_orders[bid_price]
                qty = min(available, max_sell - sell_vol)
                if qty > 0:
                    orders.append(Order(product, bid_price, -qty))
                    sell_vol += qty

        # Phase 2: Position-skewed market making (same as original)
        skew = -round(position * 0.05)
        mm_half_spread = 3

        mm_bid_price = fair_value - mm_half_spread + skew
        mm_ask_price = fair_value + mm_half_spread + skew

        remaining_buy = max_buy - buy_vol
        remaining_sell = max_sell - sell_vol

        if remaining_buy > 0:
            orders.append(Order(product, mm_bid_price, remaining_buy))
        if remaining_sell > 0:
            orders.append(Order(product, mm_ask_price, -remaining_sell))

        return orders, trader_data
