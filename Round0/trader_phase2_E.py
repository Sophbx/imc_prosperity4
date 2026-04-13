"""
IMC Prosperity 4 - Round 0 (Tutorial)
Phase 2 / Agent E: EMERALDS Maximum Extractor
==============================================

Mandate: every Phase-1 trader scored exactly EMERALDS = 14,945 (= 7,182
day -2 + 7,763 day -1, in BOTH match modes). Agent E is the focused
attempt to push past that ceiling by porting Agent B's "quote inside
the wall" insight from TOMATOES to EMERALDS, and by hunting any other
EMERALDS-only edge that the Hedgehogs StaticTrader leaves behind.

==============================================
Findings (full receipts in agents/phase2_E_design.md)
==============================================

1. EMERALDS order book microstructure (cross-validated days -1, -2):
        - bid_wall = 9990, ask_wall = 10010 -> 100% of 20,000 ticks
        - noise quotes = (9992, 10008) -> 100% of ticks
        - intermittent in-book aggressor at 10000 -> ~1.6% of ticks
        - NO order-book level ever lives between the wall and the
          noise quote (no 9991, no 10009), so unlike TOMATOES there
          is NO "virgin edge zone" between noise and wall.
        - EVERY market trade in the trades CSV happens at exactly one
          of {9992, 10000, 10008}. Zero trades at any other price.
          (verified day -1: prices = {9992: 557, 10000: 40, 10008: 552})

2. Backtester fill mechanics (prosperity3bt runner.py):
        - Phase A (in-book sweep): your buy at price P matches against
          asks at price <= P, paying their listed price.
        - Phase B (market-trade fill): your buy at price P matches a
          historical sell trade at price `tp`, paying YOUR price P, IF
          tp < P (strict in 'worse' mode) or tp <= P (in optimistic).

3. Inventory of fills the baseline EmeraldsTrader actually realizes
   (parsed from a baseline backtest's Trade History, both days):
        BUYS:  1087 @ 9993 (+7 each), 511 @ 10000 (+0 each)
        SELLS: 1048 @ 10007 (+7 each), 551 @ 10000 (+0 each)
        Realized PnL = 7*(1087+1048) = 14,945. Inventory ends flat.
        Note 1087 == 557+530 == every historical 9992 sell trade in
        the data. We are at 100% capture of available market trades.

4. Therefore in CONSERVATIVE matching the realized PnL is bounded
   above by:
        sum over historical EMERALDS trades of edge_per_fill
   where edge_per_fill is at most 7 for the 9992/10008 trades (since
   in 'worse' mode quoting at 9992 itself does NOT match equal-price
   historical trades) and 8 for the 10000 trades (which we already
   sweep via the in-book-take phase). The arithmetic gives exactly
   14,945. **There is no conservative-mode play that goes higher.**
   Variants tested empirically all match this bound:
        bid 9994 -> 12,810 (-1 per fill, same fill count)
        bid 9995 -> 10,675 (-2 per fill)
        bid 9996 ->  8,540 (-3 per fill)
        bid 9991 ->      0 (no fills)

5. In OPTIMISTIC matching, equal-price historical trades DO match.
   This means a buy quote at 9992 matches every historical 9992 sell
   trade and pays 9992 = 8 ticks of edge instead of 7. Symmetric for
   sells at 10008. Lifts the optimistic ceiling from 14,945 to about
   14,945 + 2,135 = 17,080. Confirmed empirically.

==============================================
Strategy
==============================================

EMERALDS:
    The new MAKE rule is a SPLIT QUOTE that is provably optimal under
    BOTH match modes:

        - Half of buy capacity is posted at 9993 (the conservative-mode
          maximum-edge price).
        - Half of buy capacity is posted at 9992 (the optimistic-mode
          maximum-edge price -- ties the noise quote and pays +8).
        - Symmetric on the ask side at 10007 and 10008.

    Why this is strictly >= baseline in BOTH modes:
        - In CONSERVATIVE: the 9992 layer never fills (== price excluded),
          so all the action happens at 9993, identical to baseline. The
          9993 layer captures every historical 9992 sell trade that the
          baseline captures. EMERALDS PnL = 14,945, unchanged.
        - In OPTIMISTIC: the 9992 layer fills FIRST against historical
          9992 sells, paying the better price (+8 instead of +7). EME
          PnL ~= 17,080, an honest +2,135.

    The rest of the EMERALDS logic (the in-book TAKE phase, including
    the wall_mid unwind take) is preserved verbatim from phase1_B,
    because the backtest receipts show that those rules are responsible
    for the 511+551 zero-edge inventory rotations at 10000 that are
    necessary to keep position capacity available for the +7/+8
    earning quotes. Removing them is a direct PnL loss.

    A position-limit guard splits the available capacity exactly in
    half so the two layers together never violate the 80-position
    limit (the backtester drops the entire product's orders if the
    sum of long orders + position > limit, so naive 80+80 layering
    is illegal).

TOMATOES:
    UNCHANGED FROM trader_phase1_B.py. Per the agent prompt and per
    the cross-validation evidence in phase1_B_design.md, TomatoesTrader
    is the current champion (13,639 conservative both days). Touching
    it is out of scope and risk-asymmetric.

==============================================
Honest disclosure
==============================================

In CONSERVATIVE mode (the realistic submission mode), EMERALDS is
GENUINELY saturated at 14,945. This trader does NOT magically break
that ceiling -- it ties it. The lift comes only in optimistic mode,
where the same fills are repriced 1 tick better. Anyone who runs this
in conservative mode and reports "no improvement on EMERALDS" is not
finding a bug; they are confirming the theoretical cap derived in
finding (4) above.

If the live engine matches at the 'worse' (conservative) discipline,
EMERALDS PnL will be 14,945 and Phase 2 E reduces to Phase 1 B for
EMERALDS (with TOMATOES still at the phase1_B optimum).

If the live engine is more permissive on equal-price matches (closer
to optimistic), Phase 2 E captures the +2,135 lift on EMERALDS.

Either way, this trader is strictly >= phase1_B on EMERALDS, never
worse.
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

# TOMATOES constants - identical to phase1_B (do not touch).
TOMATO_TAKE_EDGE = 2
TOMATO_NOISE_STEP = 2
TOMATO_SOFT_INVENTORY = 40

# EMERALDS constants - microstructure-derived, not parameter-searched.
EMERALDS_FAIR = 10000           # wall_mid is fixed at 10000 in 100% of ticks
EMERALDS_NOISE_BID = 9992
EMERALDS_NOISE_ASK = 10008
EMERALDS_BID_INSIDE = 9993      # one tick inside noise (conservative-mode optimum)
EMERALDS_ASK_INSIDE = 10007     # mirror
EMERALDS_NOISE_BID_TIE = 9992   # tie noise on bid side (optimistic-mode optimum)
EMERALDS_NOISE_ASK_TIE = 10008  # mirror


# ============================================================
# ProductTrader base class (verbatim from phase1_B)
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
        if abs_volume <= 0:
            return
        order = Order(self.name, int(price), abs_volume)
        if logging:
            self.log("BUYO", {"p": price, "s": self.name, "v": int(volume)}, product_group='ORDERS')
        self.max_allowed_buy_volume -= abs_volume
        self.orders.append(order)

    def ask(self, price, volume, logging=True):
        abs_volume = min(abs(int(volume)), self.max_allowed_sell_volume)
        if abs_volume <= 0:
            return
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
# EmeraldsTrader -- Phase 2 E split-layer evolution.
#
# Key change vs phase1_B EmeraldsTrader:
#   * MAKE phase posts a TWO-LEVEL split (9992 + 9993 / 10007 + 10008)
#     instead of a single dynamic-overbid quote at 9993/10007.
#   * Total volume posted == max_allowed_*_volume (just split in two)
#     so the position-limit guard is satisfied.
#
# All TAKE-phase logic (including the wall_mid unwind take that produces
# the +0-edge inventory rotations) is unchanged.
# ============================================================
class EmeraldsTrader(ProductTrader):
    def __init__(self, state, prints, new_trader_data):
        super().__init__(EMERALDS_SYMBOL, state, prints, new_trader_data)

    def get_orders(self):
        if self.wall_mid is None:
            return {self.name: self.orders}

        # ==========================================================
        # Phase 1: TAKE  (verbatim from phase1_B)
        # ==========================================================
        # Buy any ask strictly inside wall_mid (positive edge), and at
        # wall_mid itself only when we are short (zero-edge unwind that
        # frees buy capacity for future +7/+8 fills).
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

        # ==========================================================
        # Phase 2: MAKE  (split-level, the Phase 2 E innovation)
        # ==========================================================
        # Split remaining buy capacity 50/50 between:
        #   - 9993 (the conservative-mode optimum: +7 edge per fill)
        #   - 9992 (the optimistic-mode optimum:  +8 edge per fill,
        #           but ties the noise bot so under conservative match
        #           it never fills, harmlessly)
        # Symmetric on the ask side.
        cap_buy = self.max_allowed_buy_volume
        cap_sell = self.max_allowed_sell_volume

        # Round half-volume up so we don't drop a single lot.
        bid_inside_size = (cap_buy + 1) // 2   # 9993 layer
        bid_tie_size = cap_buy // 2            # 9992 layer
        ask_inside_size = (cap_sell + 1) // 2  # 10007 layer
        ask_tie_size = cap_sell // 2           # 10008 layer

        # ORDER MATTERS: the matching engine processes orders sequentially
        # and depletes each historical market trade's available quantity
        # as it goes. We must post the OPTIMISTIC-mode optimum (9992 / 10008,
        # +8 edge) BEFORE the conservative-mode workhorse (9993 / 10007,
        # +7 edge), so that under permissive matching the better-priced
        # quote claims the historical trade first. Under conservative
        # matching the 9992/10008 layer simply doesn't fill (== price
        # excluded) and the 9993/10007 layer behind it captures everything
        # exactly as the baseline does.
        self.bid(EMERALDS_NOISE_BID_TIE, bid_tie_size)
        self.bid(EMERALDS_BID_INSIDE, bid_inside_size)
        self.ask(EMERALDS_NOISE_ASK_TIE, ask_tie_size)
        self.ask(EMERALDS_ASK_INSIDE, ask_inside_size)

        return {self.name: self.orders}


# ============================================================
# TomatoesTrader -- BYTE-IDENTICAL to phase1_B. Do not touch.
# ============================================================
class TomatoesTrader(ProductTrader):
    """
    Philosophy:
        - fair_value = wall_mid (current, no smoothing)
        - TAKE when an order is beyond the noise-quote zone
        - MAKE one tick INSIDE the noise quote (queue priority)
        - UNWIND at wall_mid when inventory builds
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

        self.log('WM', fv)
        self.log('BW', self.bid_wall)
        self.log('AW', self.ask_wall)
        self.log('POS', self.initial_position)

        # ----- TAKE -----
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

        # ----- MAKE -----
        bid_price = int(self.bid_wall + TOMATO_NOISE_STEP)
        ask_price = int(self.ask_wall - TOMATO_NOISE_STEP)

        if self.best_bid is not None and self.best_bid >= bid_price:
            bid_price = self.best_bid
            if bid_price >= fv:
                bid_price = int(fv - 1)

        if self.best_ask is not None and self.best_ask <= ask_price:
            ask_price = self.best_ask
            if ask_price <= fv:
                ask_price = int(fv + 1)

        if bid_price >= fv:
            bid_price = int(self.bid_wall + 1)
        if ask_price <= fv:
            ask_price = int(self.ask_wall - 1)

        self.log('MK_B', bid_price)
        self.log('MK_A', ask_price)

        # ----- Inventory flattening overlay -----
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
