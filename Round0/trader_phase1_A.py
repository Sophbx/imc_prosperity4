"""
IMC Prosperity 4 - Round 0 (Tutorial)
trader_phase1_A.py - Rigorous Statistician minimalist build
============================================================

Philosophy
----------
Every parameter must be justified by day -1 AND day -2 data. If a feature
is not clearly supported by the EDA, it is removed. Default is to do LESS
than the current best (trader_hybrid_v1.py), not more.

EMERALDS
--------
Kept IDENTICAL to trader_hybrid_v1.py / trader_wallmid_v1.py. The data
says the book is perfectly static on both days:
  - wall spread == 20 in 100% of ticks (bid_wall 9990, ask_wall 10010)
  - wall_mid == 10000 in 100% of ticks
  - noise pair (9992/10008) present in 100% of ticks
  - zero profitable TAKE opportunities vs wall_mid - 1
So EMERALDS PnL is purely a MAKE game, and the wall-mid MAKE approach is
already at cap (14,945) in both match modes. No parameter to tune.

TOMATOES
--------
Key EDA results (computed from /Round0/Data/prices_round_0_day_-{1,2}.csv):

  1. Wall spread is 16 in 93-97% of ticks on both days; wall_spread in
     {15, 16, 17} covers 100% of ticks.
  2. Wall_mid 1-step return std ~= 0.626 on BOTH days (identical).
  3. Wall_mid 1-step return AR(1) slope = -0.18 on day -1 and -0.17 on
     day -2 (consistent mean reversion).
  4. Best EMA predictor of next wall_mid (MSE):
        alpha=0.1 -> 1.399       alpha=0.5 -> 0.431
        alpha=0.2 -> 0.779       alpha=0.8 -> 0.379  <-- best
        alpha=0.3 -> 0.579       alpha=1.0 -> 0.392  (= naive)
     alpha=1.0 ("use current wall_mid") is within 3% of the optimum and
     trader_hybrid_v1.py's alpha=0.3 is SIGNIFICANTLY worse (MSE 0.579,
     ~50% higher than naive). This is because the lag-1 autocorrelation
     is negative: smoothing blends in stale data and biases the fair
     value in the wrong direction.
     -> Decision: drop the EMA entirely. Use raw wall_mid as fair value.
        This is one fewer parameter AND a better predictor.
  5. Profitable TAKE opportunities (ask <= wall_mid - 1 OR bid >=
     wall_mid + 1) exist in ~1.2% of asks and ~0.6% of bids on BOTH days.
     Small but non-zero, and consistent. -> Keep TAKE phase, use raw
     wall_mid as reference.
  6. MM posting at fair_value +- 3 (half_spread=3):
       - NEVER crosses the existing best ask on either day (0/10000).
       - half_spread=2 crosses the book 60 ticks on day -1 and 73 on
         day -2 (we would self-trade or walk into the book). REJECTED.
       - half_spread=3 improves top-of-book 96.3%/96.4% of ticks on day
         -1/-2; half_spread=4 is identical in top-of-book fraction but
         with 33% wider spread and fewer fills.
     -> half_spread = 3 is the minimum SAFE value that also maximizes
        top-of-book presence. Keep.
  7. Inventory skew: the negative lag-1 autocorr means after an adverse
     move we expect reversion, so skewing quotes to unwind inventory is
     data-supported. kappa = 0.05 gives max skew of 4 ticks at
     position=+-80 (=full half-spread shift), which is a conservative
     but non-trivial nudge. Original trader.py value; no stronger choice
     is justified by the data alone.
  8. Order book imbalance (bid_vol - ask_vol) / (bid_vol + ask_vol):
        - 99.9% of ticks have |imbalance| <= 0.2 on both days.
        - Correlation with next-tick wall_mid return is 0.0006 (day -1)
          and 0.0179 (day -2). Essentially zero AND inconsistent.
        - Only 8 ticks (day -1) / 8 ticks (day -2) have |imb|>0.2.
     -> Imbalance feature is REJECTED. No data support.
  9. Trades CSV shows many fills at intermediate prices on TOMATOES,
     confirming there is real flow between the wall and wall_mid, which
     is why MAKE at wall_mid +- 3 can capture fills.
 10. Sharp moves (|ret| > 2*std) occur in 1.13% (day -1) and 0.97%
     (day -2) of ticks -- consistent light tails, no special handling
     needed.

Cross-day consistency report
----------------------------
STABLE (nearly identical day -1 vs day -2):
  - wall_mid return std (0.626 vs 0.626)
  - AR(1) slope (-0.183 vs -0.171)
  - wall spread distribution (93-97% at 16)
  - spread distribution, level counts, total volumes
  - EMA alpha ranking (alpha=0.8 best, alpha=0.3 bad on both days)
  - TAKE opportunity frequencies (1.19/0.60 vs 1.23/0.57)
  - EMERALDS: identically flat on both days
DIFFERS MATERIALLY:
  - TOMATOES absolute mid level (4977 vs 5008) -- this is noise around
    the asset price, the model must NOT depend on absolute level.
  - TOMATOES mid dispersion (std 14.58 vs 10.29) -- intraday drift size
    differs, but volatility STRUCTURE (1-step std) is identical.
  - Imbalance correlation with forward returns (0.0006 vs 0.0179) --
    both near-zero and inconsistent, supports REJECTING imbalance.

Differences from trader_hybrid_v1.py
------------------------------------
  - TOMATOES EMA removed (alpha=0.3 was empirically inferior to naive).
  - TOMATOES fair value rounds wall_mid via int(round(...)) instead of
    running through an EMA; half-tick wall_mids (wall_spread odd, ~3-7%
    of ticks) are rounded to the nearest int, same as hybrid v1 which
    round()s the EMA output.
  - Removed 'tomato_ema_wallmid' trader_data slot.
  - Everything else is byte-for-byte preserved.

Nothing else is added. No imbalance skew, no regime detection, no
dynamic half-spread. The data does not support those features on either
day.
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

# TOMATOES params. Justifications in the module docstring:
#   TOMATO_HALF_SPREAD = 3  -> minimum safe half-spread (hs=2 crosses the
#                              book; hs=4 offers no additional top-of-
#                              book coverage).
#   TOMATO_SKEW_KAPPA  = 0.05 -> 4-tick max skew at +-80, matches the
#                                half-spread; data-supported by negative
#                                lag-1 autocorrelation (inventory bleed
#                                unwinds into reversion).
TOMATO_HALF_SPREAD = 3
TOMATO_SKEW_KAPPA = 0.05


# ============================================================
# Minimal ProductTrader base class
# (same structure as trader_hybrid_v1.py so behavior is directly
#  comparable; nothing removed, nothing added)
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
# EmeraldsTrader - identical to trader_hybrid_v1.py
# (Emeralds is already at cap; data gives no justification to change.)
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
# TomatoesTrader - minimal, data-justified version
# (no EMA, no imbalance, no regime tricks; raw wall_mid is the
#  best available fair-value predictor on both days)
# ============================================================
class TomatoesTrader(ProductTrader):
    """
    TOMATOES: raw wall_mid as fair value.

    Why not EMA (as trader_hybrid_v1.py does with alpha=0.3)?
      - 1-step lag-1 autocorr is NEGATIVE (-0.18) on both days.
      - EMA smoothing blends STALE data with current -- because of the
        negative autocorr, that bias is exactly the WRONG direction.
      - Measured MSE for predicting next wall_mid:
          alpha=0.3 -> 0.579     <- trader_hybrid_v1.py
          alpha=1.0 -> 0.392     <- this file (naive wall_mid)
          alpha=0.8 -> 0.379     <- minor improvement, not worth a param
        alpha=0.3 is ~48% worse than naive on BOTH days.
      - Using raw wall_mid: one fewer parameter, better predictor.

    Phases are the same as trader_hybrid_v1.py:
      1. TAKE any ask strictly below fair, any bid strictly above.
         (1.2% asks, 0.6% bids on both days -- small, real, symmetric.)
      2. MAKE at fair_value +- 3 with linear inventory skew
         kappa=0.05. half_spread=3 is the minimum safe value (hs=2
         crosses the book on 60+ ticks/day).
    """

    def __init__(self, state, prints, new_trader_data):
        super().__init__(TOMATOES_SYMBOL, state, prints, new_trader_data)

    def get_orders(self):
        if self.wall_mid is None:
            return {self.name: self.orders}

        # ---- Fair value: raw wall_mid, rounded to int ----
        # wall_mid is a half-integer when wall_spread is odd (~3-7% of
        # ticks). Python's built-in round() uses banker's rounding; that
        # is unbiased over the long run and matches what trader_hybrid_v1
        # does after the EMA. No smoothing.
        fair_value = int(round(self.wall_mid))

        self.log('WM', self.wall_mid)
        self.log('FV', fair_value)

        # ---- Phase 1: TAKE ----
        # Buy any ask strictly below fair; sell any bid strictly above.
        # Identical rule to trader_hybrid_v1.py; only the fair value is
        # changed (raw wall_mid instead of EMA).
        for sp, sv in self.mkt_sell_orders.items():
            if sp < fair_value:
                self.bid(sp, sv, logging=False)
            else:
                break

        for bp, bv in self.mkt_buy_orders.items():
            if bp > fair_value:
                self.ask(bp, bv, logging=False)
            else:
                break

        # ---- Phase 2: MAKE with inventory skew ----
        # skew = -round(position * kappa) -- negative so long positions
        # lower both quotes (easier to sell, harder to add more longs).
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
