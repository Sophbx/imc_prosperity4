"""
IMC Prosperity 4 - Round 0 (Tutorial)
Wall Mid + Dynamic Overbid/Underbid strategy - v1
==================================================

This is v1 of the Wall Mid market-making strategy, ported from the
Frankfurt Hedgehogs Prosperity 3 runner-up algorithm (see
imc-prosperity-3/FrankfurtHedgehogs_polished.py). It preserves the full
ProductTrader base class framework and the two-phase (TAKE + MAKE)
quoting logic for our two Round 0 products.

Product mapping
---------------
  EMERALDS  <->  RAINFOREST_RESIN  (StaticTrader template)
      - Extremely stable, fair value anchored at 10,000.
      - Bot book: deep wall @ 9990/10010 and noise quotes @ 9992/10008.
      - Handled identically to Resin: wall-mid based TAKE + dynamic
        overbid/underbid MAKE, skipping all Olivia logic (never needed).

  TOMATOES  <->  KELP              (DynamicTrader template)
      - Volatile, dynamic fair value, ~20 std, intra-day drift.
      - Hedgehogs' Kelp does NOT use any EMA/prediction - it simply
        treats the current wall_mid as the best estimate of future price
        ("the best estimate of future price is the current price").
      - We follow that verbatim. No EMA, no skew, no moving averages.

Features explicitly REMOVED from the Hedgehogs source
-----------------------------------------------------
  - Informed trader (Olivia) detection - no informed traders observed in
    Round 0, so check_for_informed(), self.informed_direction,
    informed_bought_ts/informed_sold_ts are all stripped.
  - InkTrader (SQUID_INK) - product not in Round 0.
  - EtfTrader / OptionTrader / CommodityTrader - products not in Round 0.
  - Any options/BS/vega logic.
  - Conversions (always 0 in Round 0).

Risks / edge cases this file handles
-----------------------------------
  - Empty or one-sided order books (wall_mid is None -> skip entirely).
  - Position already at limit (bid()/ask() helpers clamp to
    max_allowed_*_volume, which can be 0).
  - traderData JSON (de)serialization errors (wrapped in try/except).
"""

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import json


# ============================================================
# Module-level constants
# ============================================================
EMERALDS_SYMBOL = 'EMERALDS'
TOMATOES_SYMBOL = 'TOMATOES'

POS_LIMITS = {
    EMERALDS_SYMBOL: 80,
    TOMATOES_SYMBOL: 80,
}


# ============================================================
# ProductTrader base class
# (ported from FrankfurtHedgehogs_polished.py lines ~100-270,
#  with check_for_informed() removed)
# ============================================================
class ProductTrader:
    """
    Base class providing common utilities: order depth parsing, wall
    detection, position-limit clamped bid/ask helpers, and structured
    logging. Subclasses override get_orders().
    """

    def __init__(self, name, state, prints, new_trader_data, product_group=None):
        self.orders: List[Order] = []

        self.name = name
        self.state = state
        self.prints = prints
        self.new_trader_data = new_trader_data
        self.product_group = name if product_group is None else product_group

        self.last_traderData = self.get_last_traderData()

        self.position_limit = POS_LIMITS.get(self.name, 0)
        self.initial_position = self.state.position.get(self.name, 0)
        # expected_position: update this if you expect an already-committed
        # position change (e.g. pre-hedging). Unused in this v1 but kept
        # for parity with the Hedgehogs framework.
        self.expected_position = self.initial_position

        self.mkt_buy_orders, self.mkt_sell_orders = self.get_order_depth()
        self.bid_wall, self.wall_mid, self.ask_wall = self.get_walls()
        self.best_bid, self.best_ask = self.get_best_bid_ask()

        # These get decremented inside bid() / ask() as we commit volume.
        self.max_allowed_buy_volume, self.max_allowed_sell_volume = self.get_max_allowed_volume()
        self.total_mkt_buy_volume, self.total_mkt_sell_volume = self.get_total_market_buy_sell_volume()

    def get_last_traderData(self):
        last_traderData = {}
        try:
            if self.state.traderData != '':
                last_traderData = json.loads(self.state.traderData)
        except Exception:
            self.log("ERROR", 'td')
        return last_traderData

    def get_best_bid_ask(self):
        best_bid = best_ask = None
        try:
            if len(self.mkt_buy_orders) > 0:
                best_bid = max(self.mkt_buy_orders.keys())
            if len(self.mkt_sell_orders) > 0:
                best_ask = min(self.mkt_sell_orders.keys())
        except Exception:
            pass
        return best_bid, best_ask

    def get_walls(self):
        """
        Wall Mid: the core of this strategy.

        bid_wall  = MIN bid price in the book (deepest visible bid level)
        ask_wall  = MAX ask price in the book (deepest visible ask level)
        wall_mid  = (bid_wall + ask_wall) / 2

        Using the walls (instead of best_bid/best_ask) as the fair-value
        reference ignores noise quotes that sit inside the bot market
        maker's wall. See the Hedgehogs README section "What is the Wall
        Mid and why did we use it?" for the motivation.
        """
        bid_wall = wall_mid = ask_wall = None

        try:
            bid_wall = min([x for x, _ in self.mkt_buy_orders.items()])
        except Exception:
            pass

        try:
            ask_wall = max([x for x, _ in self.mkt_sell_orders.items()])
        except Exception:
            pass

        try:
            wall_mid = (bid_wall + ask_wall) / 2
        except Exception:
            pass

        return bid_wall, wall_mid, ask_wall

    def get_total_market_buy_sell_volume(self):
        market_bid_volume = market_ask_volume = 0
        try:
            market_bid_volume = sum([v for p, v in self.mkt_buy_orders.items()])
            market_ask_volume = sum([v for p, v in self.mkt_sell_orders.items()])
        except Exception:
            pass
        return market_bid_volume, market_ask_volume

    def get_max_allowed_volume(self):
        # How much more we're allowed to buy / sell before hitting position limit.
        max_allowed_buy_volume = self.position_limit - self.initial_position
        max_allowed_sell_volume = self.position_limit + self.initial_position
        return max_allowed_buy_volume, max_allowed_sell_volume

    def get_order_depth(self):
        """
        Returns (buy_orders, sell_orders) dicts sorted from best to worst
        price, with volumes stored as positive ints.
        """
        order_depth = None
        buy_orders: Dict[int, int] = {}
        sell_orders: Dict[int, int] = {}

        try:
            order_depth: OrderDepth = self.state.order_depths[self.name]
        except Exception:
            pass

        try:
            buy_orders = {
                bp: abs(bv)
                for bp, bv in sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
            }
        except Exception:
            pass

        try:
            sell_orders = {
                sp: abs(sv)
                for sp, sv in sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
            }
        except Exception:
            pass

        return buy_orders, sell_orders

    def bid(self, price, volume, logging=True):
        """Post a BUY order, clamped to remaining buy capacity."""
        abs_volume = min(abs(int(volume)), self.max_allowed_buy_volume)
        order = Order(self.name, int(price), abs_volume)
        if logging:
            self.log("BUYO", {"p": price, "s": self.name, "v": int(volume)}, product_group='ORDERS')
        self.max_allowed_buy_volume -= abs_volume
        self.orders.append(order)

    def ask(self, price, volume, logging=True):
        """Post a SELL order, clamped to remaining sell capacity."""
        abs_volume = min(abs(int(volume)), self.max_allowed_sell_volume)
        order = Order(self.name, int(price), -abs_volume)
        if logging:
            self.log("SELLO", {"p": price, "s": self.name, "v": int(volume)}, product_group='ORDERS')
        self.max_allowed_sell_volume -= abs_volume
        self.orders.append(order)

    def log(self, kind, message, product_group=None):
        if product_group is None:
            product_group = self.product_group

        if product_group == 'ORDERS':
            group = self.prints.get(product_group, [])
            group.append({kind: message})
        else:
            group = self.prints.get(product_group, {})
            group[kind] = message

        self.prints[product_group] = group

    def get_orders(self):
        # overwrite this in each subclass
        return {}


# ============================================================
# EmeraldsTrader
# (based on Hedgehogs StaticTrader, lines ~274-331)
# ============================================================
class EmeraldsTrader(ProductTrader):
    """
    EMERALDS <-> RAINFOREST_RESIN analogue. True price is essentially
    fixed at 10,000. Bot book shows deep walls at 9990/10010 and noise
    quotes at 9992/10008. Two-phase:
      Phase 1: TAKE mispriced orders using wall_mid as fair value,
               with an unwind exception at wall_mid when already
               on the wrong side.
      Phase 2: MAKE at bid_wall+1 / ask_wall-1, with dynamic
               overbid/underbid to jump ahead of noise quotes as long
               as we keep positive edge vs wall_mid.
    """

    def __init__(self, state, prints, new_trader_data):
        super().__init__(EMERALDS_SYMBOL, state, prints, new_trader_data)

    def get_orders(self):
        if self.wall_mid is not None:

            ##########################################################
            # 1. TAKING (from Hedgehogs StaticTrader, lines ~286-298)
            ##########################################################
            # Buy any ask priced at wall_mid - 1 or cheaper (clear edge).
            # Also buy at wall_mid if we're currently short, to unwind
            # at fair value (flat = zero edge but also zero holding risk).
            for sp, sv in self.mkt_sell_orders.items():
                if sp <= self.wall_mid - 1:
                    self.bid(sp, sv, logging=False)
                elif sp <= self.wall_mid and self.initial_position < 0:
                    volume = min(sv, abs(self.initial_position))
                    self.bid(sp, volume, logging=False)

            # Mirror for the ask side: sell any bid at wall_mid + 1 or
            # better, and also unwind longs at wall_mid.
            for bp, bv in self.mkt_buy_orders.items():
                if bp >= self.wall_mid + 1:
                    self.ask(bp, bv, logging=False)
                elif bp >= self.wall_mid and self.initial_position > 0:
                    volume = min(bv, self.initial_position)
                    self.ask(bp, volume, logging=False)

            ###########################################################
            # 2. MAKING (from Hedgehogs StaticTrader, lines ~303-328)
            ###########################################################
            bid_price = int(self.bid_wall + 1)  # base case
            ask_price = int(self.ask_wall - 1)  # base case

            # OVERBIDDING: If there's a noise quote inside our wall
            # (e.g. bid at 9992 when bid_wall is 9990), overbid it by 1
            # as long as the new bid is still strictly below wall_mid
            # (so we keep positive edge). Otherwise, just match it.
            for bp, bv in self.mkt_buy_orders.items():
                overbidding_price = bp + 1
                if bv > 1 and overbidding_price < self.wall_mid:
                    bid_price = max(bid_price, overbidding_price)
                    break
                elif bp < self.wall_mid:
                    bid_price = max(bid_price, bp)
                    break

            # UNDERBIDDING (mirror of above for asks).
            for sp, sv in self.mkt_sell_orders.items():
                underbidding_price = sp - 1
                if sv > 1 and underbidding_price > self.wall_mid:
                    ask_price = min(ask_price, underbidding_price)
                    break
                elif sp > self.wall_mid:
                    ask_price = min(ask_price, sp)
                    break

            # POST ORDERS - use all remaining capacity.
            self.bid(bid_price, self.max_allowed_buy_volume)
            self.ask(ask_price, self.max_allowed_sell_volume)

        return {self.name: self.orders}


# ============================================================
# TomatoesTrader
# (based on Hedgehogs DynamicTrader, lines ~335-377,
#  with ALL informed-trader / Olivia logic removed)
# ============================================================
class TomatoesTrader(ProductTrader):
    """
    TOMATOES <-> KELP analogue. Volatile, drifting fair value. The
    Hedgehogs approach is strikingly simple: use the current wall_mid
    as the fair-value estimate (no EMA, no regression). The only
    quoting nuance is the "don't cross wall_mid" guard: if bid_wall+1
    would touch wall_mid, back off to bid_wall (same mirrored for asks).
    """

    def __init__(self, state, prints, new_trader_data):
        super().__init__(TOMATOES_SYMBOL, state, prints, new_trader_data)
        # NOTE: the Hedgehogs DynamicTrader additionally calls
        # check_for_informed() here. We skip it entirely - no Olivia
        # in Round 0.

    def get_orders(self):
        if self.wall_mid is not None:

            # ---- BID side ----
            # Default: one tick inside the bid wall.
            bid_price = self.bid_wall + 1
            bid_volume = self.max_allowed_buy_volume

            # Wall-mid guard: if our +1 overbid would hit/cross wall_mid
            # (which happens when the spread is only 2 ticks wide),
            # back off to sitting on the wall itself to preserve edge.
            # Original Hedgehogs line ~356 had an additional SHORT-bias
            # branch from informed-trader logic; stripped here.
            if self.wall_mid - bid_price < 1:
                bid_price = self.bid_wall

            self.bid(bid_price, bid_volume)

            # ---- ASK side (mirror) ----
            ask_price = self.ask_wall - 1
            ask_volume = self.max_allowed_sell_volume

            if ask_price - self.wall_mid < 1:
                ask_price = self.ask_wall

            self.ask(ask_price, ask_volume)

        return {self.name: self.orders}


# ============================================================
# Main Trader - entry point called by the Prosperity runtime
# ============================================================
class Trader:
    """
    Dispatches per-product ProductTrader subclasses, collects their
    orders, and serializes shared state via traderData. Mirrors the
    Hedgehogs top-level Trader.run() pattern (lines ~881-925).
    """

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        new_trader_data: Dict = {}

        # Structured log bucket, same shape as Hedgehogs' `prints`.
        prints = {
            "GENERAL": {
                "TIMESTAMP": state.timestamp,
                "POSITIONS": state.position,
            },
        }

        def export(prints_dict):
            try:
                print(json.dumps(prints_dict))
            except Exception:
                pass

        # Dispatch table: symbol -> ProductTrader subclass.
        product_traders = {
            EMERALDS_SYMBOL: EmeraldsTrader,
            TOMATOES_SYMBOL: TomatoesTrader,
        }

        for symbol, product_trader_cls in product_traders.items():
            if symbol in state.order_depths:
                try:
                    trader = product_trader_cls(state, prints, new_trader_data)
                    result.update(trader.get_orders())
                except Exception:
                    # Never let one product's failure wipe the whole run.
                    pass

        # Serialize trader state for next iteration.
        try:
            final_trader_data = json.dumps(new_trader_data)
        except Exception:
            final_trader_data = ''

        export(prints)

        # Round 0 has no conversion mechanics.
        conversions = 0
        return result, conversions, final_trader_data
