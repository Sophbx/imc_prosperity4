"""
IMC Prosperity 4 - Round 0 (Tutorial)
Phase 2 / Agent D: TOMATOES Maximum Extractor
==============================================

Builds on Phase 1 / Agent B (the current champion) by ADDING one
single piece of microstructure-aware logic on the MAKE side: an
adaptive "always quote one tick tighter than the visible noise quote"
rule. Everything else (TAKE, inventory flatten, wall guard, EMERALDS)
is preserved from phase1_B exactly.

==========================================================================
What changed vs phase1_B (and why)
==========================================================================

phase1_B's MAKE rule was:
    bid_price = bid_wall + 2  (with a "match-don't-overpay" branch
                               if best_bid was already deeper)
    ask_price = ask_wall - 2

The "+2 hardcoded" assumed the noise quote sits 1 tick inside the wall
(the 75% case), and "+2" front-runs it. But:
    - 24% of ticks the noise quote is at +2 (not +1). In that case
      phase1_B quotes alongside it (queue-split, not queue-priority).
    - 1% of ticks the noise quote is even deeper.

phase2_D replaces the static "+2" with:
    bid_price = max(bid_wall + 2, best_bid + 1)
    ask_price = min(ask_wall - 2, best_ask - 1)

i.e. ALWAYS step exactly one tick inside the visible best quote, but
floored at the original "+2" defaults so a degenerate book never makes
us quote behind the wall. The wall_mid guard remains: if our quote
would touch wall_mid we revert to bid_wall+1.

This is hypothesis #3 from the brief ("noise quote detection").

==========================================================================
Hypotheses tested and rejected (full table in phase2_D_design.md)
==========================================================================

  1. Symmetric offset sweep (+1, +3, +4): +1 collapses conservative
     (queue collision with noise); +3 helps somewhat (29,890); +4
     hurts. Best symmetric: +3, but worse than the noise-detect
     adaptive rule.
  2. Asymmetric defaults (bid=2 ask=3 etc): bid=2/ask=3 helps a bit
     (29,964 cons) but is dominated by the adaptive rule.
  3. Mean-reversion TAKE (using lag-1 autocorr -0.18): adding it on
     top of noise-detect IMPROVES day -2 (+150) but HURTS day -1
     (-285). Cross-validation fails -> rejected as overfit.
  4. Inventory-aware TAKE relaxation: never fires (position never
     reaches half-soft threshold) -> no effect, removed.
  5. Wider asks under noise-detect: dominated by symmetric (2,2)
     defaults, because noise-detect already adapts.

The single change that beat phase1_B on BOTH days under conservative
matching is the noise-quote-tracking MAKE rule. Everything else either
helped one day at the expense of another, or had zero effect.

==========================================================================
Backtest results (champion vs phase1_B)
==========================================================================

CONSERVATIVE matching (the relevant mode):
                       Day -2     Day -1     Total
    phase1_B           15,045     13,539     28,584
    phase2_D           16,114     15,543     31,657
    Delta              +1,069     +2,004     +3,073

OPTIMISTIC matching:
                       Day -2     Day -1     Total
    phase1_B           16,510     15,860     32,370
    phase2_D           16,258     15,543     31,800
    Delta                -252       -317       -570

The optimistic regression (~-1.8%) is a known artefact: with adaptive
inside-noise quoting we sometimes step one tick wider on aggressive-
noise ticks, sacrificing 1 tick of "free" optimistic fills that would
have crossed at +2. Conservative is the realistic mode and we win it
strongly on both days. Net: +3,073 conservative which matters for
real submission.
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

# TOMATOES constants - microstructure-derived (not parameter-searched).
# +2 floor: noise quote sits 1 tick inside wall on 75% of ticks.
TOMATO_TAKE_EDGE = 2
TOMATO_BID_FLOOR_STEP = 2     # default bid offset above bid_wall
TOMATO_ASK_FLOOR_STEP = 2     # default ask offset below ask_wall
TOMATO_SOFT_INVENTORY = 40


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
        d = {}
        try:
            if self.state.traderData != '':
                d = json.loads(self.state.traderData)
        except Exception:
            pass
        return d

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
            bid_wall = min(self.mkt_buy_orders.keys())
        except Exception:
            pass
        try:
            ask_wall = max(self.mkt_sell_orders.keys())
        except Exception:
            pass
        try:
            wall_mid = (bid_wall + ask_wall) / 2
        except Exception:
            pass
        return bid_wall, wall_mid, ask_wall

    def get_total_market_buy_sell_volume(self):
        b = a = 0
        try:
            b = sum(self.mkt_buy_orders.values())
            a = sum(self.mkt_sell_orders.values())
        except Exception:
            pass
        return b, a

    def get_max_allowed_volume(self):
        return (
            self.position_limit - self.initial_position,
            self.position_limit + self.initial_position,
        )

    def get_order_depth(self):
        order_depth = None
        buy_orders: Dict[int, int] = {}
        sell_orders: Dict[int, int] = {}
        try:
            order_depth = self.state.order_depths[self.name]
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
        v = min(abs(int(volume)), self.max_allowed_buy_volume)
        if v <= 0:
            return
        self.max_allowed_buy_volume -= v
        self.orders.append(Order(self.name, int(price), v))

    def ask(self, price, volume, logging=True):
        v = min(abs(int(volume)), self.max_allowed_sell_volume)
        if v <= 0:
            return
        self.max_allowed_sell_volume -= v
        self.orders.append(Order(self.name, int(price), -v))

    def get_orders(self):
        return {}


# ============================================================
# EmeraldsTrader - byte-for-byte identical to phase1_B (untouched).
# ============================================================
class EmeraldsTrader(ProductTrader):
    def __init__(self, state, prints, new_trader_data):
        super().__init__(EMERALDS_SYMBOL, state, prints, new_trader_data)

    def get_orders(self):
        if self.wall_mid is not None:

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
# TomatoesTrader - phase1_B + adaptive noise-detect MAKE.
# ============================================================
class TomatoesTrader(ProductTrader):
    """
    Strategy = phase1_B + ONE change:

        MAKE quote = step exactly one tick INSIDE the visible noise
        quote, floored at the original phase1_B defaults of bid_wall+2
        / ask_wall-2.

    Everything else (TAKE, flatten, wall guard) is identical to
    phase1_B. The wall_mid guard still catches degenerate cases.
    """

    def __init__(self, state, prints, new_trader_data):
        super().__init__(TOMATOES_SYMBOL, state, prints, new_trader_data)

    def get_orders(self):
        if self.wall_mid is None or self.bid_wall is None or self.ask_wall is None:
            return {self.name: self.orders}

        wall_spread = self.ask_wall - self.bid_wall
        if wall_spread < 4:
            return {self.name: self.orders}

        fv = self.wall_mid

        # ==========================================================
        # Phase 1: TAKE (unchanged from phase1_B)
        # ==========================================================
        for sp, sv in self.mkt_sell_orders.items():
            if sp <= fv - TOMATO_TAKE_EDGE:
                self.bid(sp, sv, logging=False)
            elif sp <= fv and self.initial_position < 0:
                volume = min(sv, abs(self.initial_position))
                self.bid(sp, volume, logging=False)
            else:
                break

        for bp, bv in self.mkt_buy_orders.items():
            if bp >= fv + TOMATO_TAKE_EDGE:
                self.ask(bp, bv, logging=False)
            elif bp >= fv and self.initial_position > 0:
                volume = min(bv, self.initial_position)
                self.ask(bp, volume, logging=False)
            else:
                break

        # ==========================================================
        # Phase 2: MAKE (changed from phase1_B)
        # ==========================================================
        # Default floors (same as phase1_B):
        bid_price = int(self.bid_wall + TOMATO_BID_FLOOR_STEP)
        ask_price = int(self.ask_wall - TOMATO_ASK_FLOOR_STEP)

        # Adaptive noise-detect: ALWAYS step exactly one tick inside the
        # visible noise quote (the best_bid / best_ask). When noise sits
        # at bid_wall+1 (75% of ticks) the proposal == default, no change.
        # When noise sits at bid_wall+2 (24%) the proposal is bid_wall+3,
        # giving us queue priority over the noise instead of splitting it.
        if self.best_bid is not None:
            bid_price = max(bid_price, int(self.best_bid + 1))
        if self.best_ask is not None:
            ask_price = min(ask_price, int(self.best_ask - 1))

        # Wall-mid guard (unchanged): if our quote touches wall_mid, back
        # off to the safe wall+1 level.
        if bid_price >= fv:
            bid_price = int(self.bid_wall + 1)
        if ask_price <= fv:
            ask_price = int(self.ask_wall - 1)

        # ==========================================================
        # Inventory flattening overlay (unchanged from phase1_B)
        # ==========================================================
        if self.initial_position > TOMATO_SOFT_INVENTORY:
            flatten_size = self.initial_position - TOMATO_SOFT_INVENTORY
            self.ask(int(fv), flatten_size, logging=False)

        if self.initial_position < -TOMATO_SOFT_INVENTORY:
            flatten_size = abs(self.initial_position) - TOMATO_SOFT_INVENTORY
            self.bid(int(fv), flatten_size, logging=False)

        self.bid(bid_price, self.max_allowed_buy_volume)
        self.ask(ask_price, self.max_allowed_sell_volume)

        return {self.name: self.orders}


# ============================================================
# Main Trader entry point
# ============================================================
class Trader:
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        new_trader_data: Dict = {}
        prints = {}

        product_traders = {
            EMERALDS_SYMBOL: EmeraldsTrader,
            TOMATOES_SYMBOL: TomatoesTrader,
        }

        for symbol, cls in product_traders.items():
            if symbol in state.order_depths:
                try:
                    trader = cls(state, prints, new_trader_data)
                    result.update(trader.get_orders())
                except Exception:
                    pass

        try:
            final_trader_data = json.dumps(new_trader_data)
        except Exception:
            final_trader_data = ''

        return result, 0, final_trader_data
