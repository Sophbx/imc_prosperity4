from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import json

class Trader:
    """
    IMC Prosperity 4 - Tutorial Round Algorithm
    ============================================

    Products traded:
      - EMERALDS (position limit: 80) — extremely stable, fair value = 10,000
      - TOMATOES (position limit: 80) — volatile, fair value drifts over time

    High-level strategy:
    --------------------
    Both products use a two-phase approach each iteration:
      Phase 1 - TAKE: Aggressively buy any sell orders priced below our fair
                value estimate, and sell to any buy orders priced above it.
                These are "free money" trades against mispriced bot quotes.
      Phase 2 - MAKE: Post resting limit orders (market-making) inside the
                bot spread. Bots needing to trade will prefer our better prices,
                giving us the spread as profit.

    For EMERALDS, fair value is a known constant (10,000).
    For TOMATOES, fair value is tracked dynamically using an EMA.

    Key data insights (from analyzing prices_round_0_day_-1.csv and day_-2.csv):
    ---------------------------------------------------------------------------
    EMERALDS:
      - Mid-price = 10,000 in 96.7% of observations
      - Std deviation of mid-price: 0.72 (near-zero volatility)
      - Bot spread: 9,992 / 10,008 = 16 wide
      - Lag-1 autocorrelation: -0.49 (strong mean reversion)
      => Perfect for static-fair-value market making

    TOMATOES:
      - Day -2 mean mid-price: 5,008 | Day -1 mean: 4,978 (significant drift)
      - Std deviation: ~20 (volatile)
      - Bot spread: 13-14 wide
      - Price trends intra-day (drifts up/down over hundreds of ticks)
      => Requires dynamic fair value tracking + inventory management
    """

    # ================================================================
    # CONFIGURATION
    # ================================================================
    # Position limits from the Tutorial Round rules page
    LIMITS = {"EMERALDS": 80, "TOMATOES": 80}

    def bid(self):
        """
        Required for Algorithmic Trading Round 2 (auction mechanism).
        Ignored in all other rounds including the Tutorial Round.
        """
        return 15

    def run(self, state: TradingState):
        """
        Called once per iteration (1,000 iterations during testing,
        10,000 for the final scored simulation).

        Args:
            state: TradingState containing:
                - order_depths: current order book per product (bot quotes)
                - position: our current holdings per product
                - own_trades: our fills since last iteration
                - market_trades: other participants' fills since last iteration
                - traderData: string we returned last iteration (for persistence)
                - timestamp: current simulation time

        Returns:
            result: Dict[str, List[Order]] — orders to send per product
            conversions: int — conversion requests (0 for tutorial round)
            traderData: str — serialized state for next iteration
        """
        # ---- Restore persistent state ----
        # AWS Lambda is stateless, so class/global variables don't persist.
        # We serialize our state (e.g., EMA values) to a JSON string which
        # the platform passes back to us as state.traderData next iteration.
        trader_data = {}
        if state.traderData:
            try:
                trader_data = json.loads(state.traderData)
            except (json.JSONDecodeError, TypeError):
                trader_data = {}

        result = {}

        # ---- Dispatch to product-specific strategies ----
        for product in state.order_depths:
            if product == "EMERALDS":
                result[product] = self.trade_emeralds(state)
            elif product == "TOMATOES":
                result[product], trader_data = self.trade_tomatoes(
                    state, trader_data
                )
            else:
                # Future-proof: ignore unknown products gracefully
                result[product] = []

        # ---- Serialize state for next iteration ----
        traderData = json.dumps(trader_data)

        # No conversions in the tutorial round (no import/export mechanics)
        conversions = 0

        return result, conversions, traderData

    # ================================================================
    # EMERALDS STRATEGY: Static Fair Value Market Making
    # ================================================================
    def trade_emeralds(self, state: TradingState) -> List[Order]:
        """
        EMERALDS fair value is exactly 10,000 with near-zero volatility.

        Bot order book (typical):
            Bids: 9,992 x14  |  9,990 x29
            Asks: 10,008 x14 |  10,010 x29
            Spread: 16 wide

        Strategy:
          1. TAKE: Buy any asks below 10,000 (cheap emeralds).
                   Sell to any bids above 10,000 (overpriced buyers).
                   (Rare, but captures any anomalies.)
          2. MAKE: Post bid at 9,998 / ask at 10,002.
                   This is 14 tighter than the bot spread of 16.
                   Bots wanting to trade will prefer our prices, giving us
                   ~4 profit per round trip (buy 9,998, sell 10,002).

        Risk: Essentially zero. Fair value deviation is <1. Even if we hold
              max inventory at simulation end, it's valued at ~10,000.

        Position limit enforcement:
          The exchange rejects ALL orders for a product if the total buy
          (or sell) volume would push us past the position limit. So we
          carefully cap our order sizes:
            max_buy  = limit - current_position  (buying this much hits +limit)
            max_sell = limit + current_position  (selling this much hits -limit)
        """
        product = "EMERALDS"
        fair_value = 10000
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)
        limit = self.LIMITS[product]

        # How much more we're allowed to buy/sell
        max_buy = limit - position   # pos=+40 -> can buy 40 more to reach +80
        max_sell = limit + position  # pos=+40 -> can sell 120 to reach -80

        orders: List[Order] = []
        buy_vol = 0   # Tracks total buy volume committed this iteration
        sell_vol = 0  # Tracks total sell volume committed this iteration

        # ---- Phase 1: TAKE mispriced orders ----

        # Buy any sell orders below fair value (cheapest first)
        # sell_orders dict: {price: negative_quantity}
        # We iterate ascending so we grab the cheapest asks first
        if order_depth.sell_orders:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price < fair_value and buy_vol < max_buy:
                    # Negate the negative quantity to get the actual available volume
                    available = -order_depth.sell_orders[ask_price]
                    qty = min(available, max_buy - buy_vol)
                    if qty > 0:
                        # Positive quantity = BUY order
                        orders.append(Order(product, ask_price, qty))
                        buy_vol += qty

        # Sell to any buy orders above fair value (most expensive first)
        # buy_orders dict: {price: positive_quantity}
        if order_depth.buy_orders:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price > fair_value and sell_vol < max_sell:
                    available = order_depth.buy_orders[bid_price]
                    qty = min(available, max_sell - sell_vol)
                    if qty > 0:
                        # Negative quantity = SELL order
                        orders.append(Order(product, bid_price, -qty))
                        sell_vol += qty

        # ---- Phase 2: MAKE (passive market-making inside bot spread) ----
        #
        # We post resting limit orders at prices tighter than the bots:
        #   Bid at 9,998 (vs bot's best bid 9,992 -> 6 better for sellers)
        #   Ask at 10,002 (vs bot's best ask 10,008 -> 6 better for buyers)
        #
        # When a bot wants to sell, it will hit our 9,998 bid first (price-time
        # priority: our price is better). When a bot wants to buy, it will lift
        # our 10,002 offer first. Each pair of trades nets ~4 profit.
        #
        # We use all remaining position capacity for these orders to maximize
        # the volume of spread we capture.

        mm_bid_price = fair_value - 2   # 9,998
        mm_ask_price = fair_value + 2   # 10,002

        remaining_buy = max_buy - buy_vol
        remaining_sell = max_sell - sell_vol

        if remaining_buy > 0:
            orders.append(Order(product, mm_bid_price, remaining_buy))
        if remaining_sell > 0:
            orders.append(Order(product, mm_ask_price, -remaining_sell))

        return orders

    # ================================================================
    # TOMATOES STRATEGY: Dynamic Fair Value Market Making with EMA
    # ================================================================
    def trade_tomatoes(
        self, state: TradingState, trader_data: dict
    ) -> tuple:
        """
        TOMATOES has a shifting fair value that trends over time.

        Data observations:
          - Day-over-day shift of ~30 points (5,008 -> 4,978)
          - Intra-day trends of 15-30 points
          - Lag-1 autocorrelation: -0.42 (mean-reverting short-term,
            but trending at longer horizons)

        Strategy:
          1. TRACK: Estimate fair value using an Exponential Moving Average
             (EMA) of the order book mid-price. Alpha=0.3 balances:
               - Responsiveness to genuine price movements (higher alpha)
               - Smoothing out tick-by-tick noise (lower alpha)

          2. TAKE: Same as EMERALDS but using our EMA estimate as fair value.
             When the price trends, bot quotes may lag behind, creating
             opportunities to buy below or sell above our updated estimate.

          3. MAKE: Market-make at fair_value +/- 3, with an INVENTORY SKEW:
             - When we're LONG: shift both quotes DOWN
               (lower bid = less eager to buy more,
                lower ask = more eager to sell and reduce inventory)
             - When we're SHORT: shift both quotes UP
               (higher bid = more eager to buy and cover,
                higher ask = less eager to sell more)
             - Skew formula: skew = -position * 0.05 (rounded)
               e.g., position +40 -> skew -2 -> bid at FV-5, ask at FV+1

        Risk: Higher than EMERALDS due to price volatility. The inventory
              skew helps by naturally unwinding positions toward zero,
              reducing exposure to adverse price moves.

        Returns:
            (orders, updated_trader_data)
        """
        product = "TOMATOES"
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)
        limit = self.LIMITS[product]

        # ---- Compute current mid-price from the order book ----
        # Mid-price = average of best bid and best ask
        # This is our raw signal for where the market currently is
        if not order_depth.buy_orders or not order_depth.sell_orders:
            # Degenerate case: one-sided book, can't estimate fair value
            return [], trader_data

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2

        # ---- Update EMA fair value estimate ----
        # EMA formula: EMA_new = alpha * price + (1 - alpha) * EMA_old
        #
        # alpha = 0.3 means:
        #   - 30% weight on the current observation
        #   - 70% weight on the historical average
        #   - Effective lookback: ~1/alpha ≈ 3.3 observations
        #   - Responds to trends within a few ticks but smooths out noise
        #
        # We persist the EMA value across iterations via trader_data.
        alpha = 0.3
        ema_key = "tomato_ema"

        if ema_key in trader_data:
            ema = alpha * mid_price + (1 - alpha) * trader_data[ema_key]
        else:
            # First iteration: initialize EMA with current mid-price
            ema = mid_price

        trader_data[ema_key] = ema

        # Round to nearest integer for use as order prices
        # (Prosperity exchange uses integer prices)
        fair_value = round(ema)

        # ---- Position limits ----
        max_buy = limit - position
        max_sell = limit + position

        orders: List[Order] = []
        buy_vol = 0
        sell_vol = 0

        # ---- Phase 1: TAKE mispriced orders ----
        # When TOMATOES trends, the bots' stale quotes create opportunities.
        # E.g., if price dropped to 4,970 but a bot still bids at 4,980,
        # we sell at 4,980 for a 10-point profit.

        # Buy any sell orders below our fair value estimate
        for ask_price in sorted(order_depth.sell_orders.keys()):
            if ask_price < fair_value and buy_vol < max_buy:
                available = -order_depth.sell_orders[ask_price]
                qty = min(available, max_buy - buy_vol)
                if qty > 0:
                    orders.append(Order(product, ask_price, qty))
                    buy_vol += qty

        # Sell to any buy orders above our fair value estimate
        for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if bid_price > fair_value and sell_vol < max_sell:
                available = order_depth.buy_orders[bid_price]
                qty = min(available, max_sell - sell_vol)
                if qty > 0:
                    orders.append(Order(product, bid_price, -qty))
                    sell_vol += qty

        # ---- Phase 2: Position-skewed market making ----
        #
        # INVENTORY SKEW explained:
        #   Without skew, we'd accumulate large positions when the price
        #   trends in one direction (keep buying as price rises, etc.).
        #   The skew shifts our quotes to favor unwinding:
        #
        #   position = +40 (long):
        #     skew = -round(40 * 0.05) = -2
        #     bid  = FV - 3 + (-2) = FV - 5  (less eager to buy more)
        #     ask  = FV + 3 + (-2) = FV + 1  (more eager to sell to unwind)
        #
        #   position = -40 (short):
        #     skew = -round(-40 * 0.05) = +2
        #     bid  = FV - 3 + 2 = FV - 1  (more eager to buy to cover)
        #     ask  = FV + 3 + 2 = FV + 5  (less eager to sell more)
        #
        #   position = 0 (flat):
        #     skew = 0
        #     bid  = FV - 3  |  ask = FV + 3  (symmetric, pure spread capture)

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
