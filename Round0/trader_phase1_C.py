"""
IMC Prosperity 4 - Round 0 (Tutorial)
Phase 1 Agent C: Signal Hunter report
==================================================================

This file is functionally identical to trader_hybrid_v1.py.

Agent C's mandate was to hunt for exploitable bot-behavioural footprints
in the Round 0 tutorial data (Olivia-style patterns, trade-size tells,
cross-product lead-lag, order-book imbalance alpha, volume surges, etc.).

Findings (see Round0/agents/phase1_C_design.md for full details):

  1. trades CSVs have ZERO buyer/seller IDs (all NaN) on both day -1
     and day -2. Any counterparty-fingerprinting strategy (the core of
     Frankfurt Hedgehogs' Olivia exploit) is impossible on this data.

  2. EMERALDS wall_mid is CONSTANT at 10000.0 across every tick of
     both days. All "activity" is just noise inside a 16-wide book.
     There is no directional signal to layer on top of the existing
     wall_mid market-making logic.

  3. Order-book imbalance (bid_vol-ask_vol)/(bid_vol+ask_vol) shows a
     strong-looking effect predicting mid_price (top-of-book mid), but
     it predicts ZERO change in wall_mid. The baseline already uses
     wall_mid as the fair value, so this signal adds no alpha -- it
     just tracks the noise quote that the imbalance IS.

  4. No Olivia signal: trades at running daily highs vs lows are
     roughly symmetric and inconsistent between the two days. No
     bot systematically buys at lows or sells at highs.

  5. No trade-size tell: TOMATOES quantities are uniformly spread
     over {2,3,4,5} on both days; EMERALDS over {3..8}. Nothing
     distinguishes a specific counterparty.

  6. Trade flow has near-zero correlation (<0.02) with future
     wall_mid moves at lags 5/20 ticks on both days.

  7. EMERALDS wall_mid is constant so cross-product lead-lag with
     TOMATOES is mathematically undefined.

Decision: SUBMIT THE BASELINE UNCHANGED. The hunt was thorough and
negative. Fabricating a signal would only add risk.

Preserved from hybrid_v1:
  - EMERALDS: Wall Mid + dynamic overbid/underbid market making.
  - TOMATOES: EMA(alpha=0.3) of wall_mid as fair value, +-3 make
    spread, kappa=0.05 linear inventory skew.
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

# TOMATOES params (back to original trader.py values, proven robust under
# the conservative match mode in backtesting).
TOMATO_EMA_ALPHA = 0.3
TOMATO_HALF_SPREAD = 3
TOMATO_SKEW_KAPPA = 0.05


# ============================================================
# ProductTrader base class (unchanged from trader_wallmid_v1.py)
# ============================================================
class ProductTrader:
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
        self.expected_position = self.initial_position

        self.mkt_buy_orders, self.mkt_sell_orders = self.get_order_depth()
        self.bid_wall, self.wall_mid, self.ask_wall = self.get_walls()
        self.best_bid, self.best_ask = self.get_best_bid_ask()

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
        bid_wall = MIN bid (deepest visible bid level)
        ask_wall = MAX ask (deepest visible ask level)
        wall_mid = (bid_wall + ask_wall) / 2
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
        max_allowed_buy_volume = self.position_limit - self.initial_position
        max_allowed_sell_volume = self.position_limit + self.initial_position
        return max_allowed_buy_volume, max_allowed_sell_volume

    def get_order_depth(self):
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
        abs_volume = min(abs(int(volume)), self.max_allowed_buy_volume)
        order = Order(self.name, int(price), abs_volume)
        if logging:
            self.log("BUYO", {"p": price, "s": self.name, "v": int(volume)}, product_group='ORDERS')
        self.max_allowed_buy_volume -= abs_volume
        self.orders.append(order)

    def ask(self, price, volume, logging=True):
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
        return {}


# ============================================================
# EmeraldsTrader — unchanged from trader_wallmid_v1.py
# (verified to deliver ~3.6x the PnL vs fixed-spread approach)
# ============================================================
class EmeraldsTrader(ProductTrader):
    def __init__(self, state, prints, new_trader_data):
        super().__init__(EMERALDS_SYMBOL, state, prints, new_trader_data)

    def get_orders(self):
        if self.wall_mid is not None:

            # Phase 1: TAKE
            for sp, sv in self.mkt_sell_orders.items():
                if sp <= self.wall_mid - 1:
                    self.bid(sp, sv, logging=False)
                elif sp <= self.wall_mid and self.initial_position < 0:
                    volume = min(sv, abs(self.initial_position))
                    self.bid(sp, volume, logging=False)

            for bp, bv in self.mkt_buy_orders.items():
                if bp >= self.wall_mid + 1:
                    self.ask(bp, bv, logging=False)
                elif bp >= self.wall_mid and self.initial_position > 0:
                    volume = min(bv, self.initial_position)
                    self.ask(bp, volume, logging=False)

            # Phase 2: MAKE with dynamic overbid / underbid
            bid_price = int(self.bid_wall + 1)
            ask_price = int(self.ask_wall - 1)

            for bp, bv in self.mkt_buy_orders.items():
                overbidding_price = bp + 1
                if bv > 1 and overbidding_price < self.wall_mid:
                    bid_price = max(bid_price, overbidding_price)
                    break
                elif bp < self.wall_mid:
                    bid_price = max(bid_price, bp)
                    break

            for sp, sv in self.mkt_sell_orders.items():
                underbidding_price = sp - 1
                if sv > 1 and underbidding_price > self.wall_mid:
                    ask_price = min(ask_price, underbidding_price)
                    break
                elif sp > self.wall_mid:
                    ask_price = min(ask_price, sp)
                    break

            self.bid(bid_price, self.max_allowed_buy_volume)
            self.ask(ask_price, self.max_allowed_sell_volume)

        return {self.name: self.orders}


# ============================================================
# TomatoesTrader — hybrid: Wall Mid fed into EMA + inventory skew
# ============================================================
class TomatoesTrader(ProductTrader):
    """
    Hybrid TOMATOES strategy:
      1. Compute fair value via EMA of wall_mid (more robust input than
         naive best-bid/ask mid, since noise quotes don't bias it).
      2. TAKE only when the order is strictly mispriced beyond wall_mid's
         natural +-1 noise band.
      3. MAKE at fair_value +- 3 with linear inventory skew kappa=0.05
         (original trader.py values, proven robust in conservative
         backtesting).

    Why this hybrid beats pure wall_mid on TOMATOES:
      - Pure wall_mid quoting (bid_wall+1 / ask_wall-1) only makes money
        when the optimistic backtester assumes our passive quotes always
        get filled. Under conservative matching, PnL dropped to zero.
      - The EMA + fixed-spread-around-EMA approach captures real spread
        from bots that cross the book, not just passive fills.
    """

    def __init__(self, state, prints, new_trader_data):
        super().__init__(TOMATOES_SYMBOL, state, prints, new_trader_data)

    def get_orders(self):
        # Degenerate book -> bail.
        if self.wall_mid is None:
            return {self.name: self.orders}

        # ---- Fair value: EMA of wall_mid (persisted via trader_data) ----
        ema_key = 'tomato_ema_wallmid'
        prev_ema = self.last_traderData.get(ema_key, None)
        if prev_ema is None:
            ema = self.wall_mid
        else:
            ema = TOMATO_EMA_ALPHA * self.wall_mid + (1 - TOMATO_EMA_ALPHA) * prev_ema
        self.new_trader_data[ema_key] = ema
        fair_value = round(ema)

        self.log('EMA', round(ema, 2))
        self.log('FV', fair_value)

        # ---- Phase 1: TAKE ----
        # Same rule as original trader.py: buy asks < fair_value,
        # sell bids > fair_value.
        for sp, sv in self.mkt_sell_orders.items():
            if sp < fair_value:
                self.bid(sp, sv, logging=False)
            else:
                break  # prices sorted ascending, no more matches

        for bp, bv in self.mkt_buy_orders.items():
            if bp > fair_value:
                self.ask(bp, bv, logging=False)
            else:
                break  # prices sorted descending, no more matches

        # ---- Phase 2: MAKE with inventory skew ----
        # skew = -round(position * kappa), original formula from trader.py
        skew = -round(self.expected_position * TOMATO_SKEW_KAPPA)

        mm_bid_price = fair_value - TOMATO_HALF_SPREAD + skew
        mm_ask_price = fair_value + TOMATO_HALF_SPREAD + skew

        self.bid(mm_bid_price, self.max_allowed_buy_volume)
        self.ask(mm_ask_price, self.max_allowed_sell_volume)

        return {self.name: self.orders}


# ============================================================
# Main Trader entry point
# ============================================================
class Trader:
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        new_trader_data: Dict = {}

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
                    pass

        try:
            final_trader_data = json.dumps(new_trader_data)
        except Exception:
            final_trader_data = ''

        export(prints)
        conversions = 0
        return result, conversions, final_trader_data
