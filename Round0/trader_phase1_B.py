"""
IMC Prosperity 4 - Round 0 (Tutorial)
Phase 1 / Agent B: Microstructure Purist (Frankfurt Hedgehogs philosophy)
=========================================================================

This trader is the "Microstructure Purist" response to trader_hybrid_v1.py.

It refuses to use EMAs, regressions, or any time-series model. All fair
value comes from the CURRENT wall_mid - the instantaneous average of
bid_wall and ask_wall - which is the cleanest, noise-free estimate of
the bot market maker's inside price. If a pattern cannot be explained
via order-book walls, queue priority, or adverse-selection mechanics,
this agent does not trade on it.

==========================================================================
Microstructure observed in Round 0 sample data (days -1, -2) for TOMATOES
==========================================================================

    1. The wall (bid_wall = min visible bid, ask_wall = max visible ask) is
       16 ticks wide > 94% of the time. Wall size is ~20 on each side
       (the deep-liquidity bot maker).

    2. Inside the wall, there is always exactly one "noise" quote on each
       side, size ~7-8, sitting 1-2 ticks inside the wall ~99% of rows
       (so best_bid is almost always bid_wall+1 or bid_wall+2). These
       are the tradable entry points.

    3. 100% of observed market trades occur at the noise-quote levels
       (wall_mid +/- 6 or +/- 7). ZERO trades happen at the wall price
       itself - the wall is a structural liquidity floor, not an actual
       execution level.

    4. Trade flow is symmetric within ~10% (day -1: 759 sells vs 674 buys;
       day -2: 707 vs 678). There is no exploitable directional signal
       from noise asymmetry (measured corr of noise-edge-imbalance with
       next-tick wall_mid change was ~0.15, decaying; intrusion events
       resolved in the "expected" direction 35% of the time, i.e. random).

    5. wall_mid is stable - it is unchanged 60% of timesteps and moves
       at most +/- 1 on 98% of timesteps. raw_mid (= (best_bid+best_ask)/2)
       is twice as noisy because noise quotes flip-flop inside the wall.

==========================================================================
Strategy (philosophically defensible in one sentence per rule)
==========================================================================

EMERALDS (completely unchanged from trader_wallmid_v1.py - Hedgehogs Resin):
    Wall is fixed at (9990, 10010), wall_mid = 10000. This is the
    Rainforest-Resin-mode static trader: TAKE any order strictly inside
    wall_mid; MAKE at bid_wall+1 / ask_wall-1 with dynamic overbid of
    noise quotes when doing so still leaves positive edge; unwind long
    positions at wall_mid (the zero-edge exit). Nothing to add.

TOMATOES (microstructure evolution of Hedgehogs Kelp):

    Fair value = wall_mid right now. No smoothing. If wall_mid moves,
    we re-quote instantly - because the *current* wall is the best
    estimate of the *future* wall (no predictive power has been shown
    in the data).

    TAKE rules (new - Hedgehogs Kelp didn't take):
        - Buy any ask <= wall_mid - 2 (>=2 ticks of edge). The noise
          quote lives at wall_mid - 1 on occasion, so requiring <=-2
          avoids tagging its own-noise and guarantees we only take
          when something meaningfully crossed. Volume capped at
          market size.
        - Mirror for asks: sell any bid >= wall_mid + 2.
        - UNWIND TAKE at wall_mid: if we are long and a bid sits at
          wall_mid, hit it (zero-edge exit = free risk capacity).
          Mirror for short.
        - No EMA; no predictive component. The threshold of 2 comes
          from the observed noise-quote-inside-the-wall distance, not
          from backtest curve-fitting.

    MAKE rules (evolved from Hedgehogs Kelp):
        - Default bid at bid_wall + 2; default ask at ask_wall - 2.
          This is ONE TICK INSIDE THE NOISE QUOTE - i.e. queue-priority
          ahead of the noise bot, still leaving ~6 ticks edge vs
          wall_mid. This is the key evolution: Hedgehogs' Kelp sits at
          bid_wall + 1, which is at or behind the noise quote. We step
          in front of it.
        - If the noise quote is unusually deep (inside more than 2
          ticks - an "aggressive" noise day), we do NOT step in front
          of it. Instead we match its price (overbid it by zero),
          maintaining queue position without sacrificing edge further.
        - If our would-be bid price ever touches wall_mid (spread is
          degenerate), we back off to bid_wall, preserving edge.
          (Same wall-mid guard as the original.)

    INVENTORY FLATTENING (new):
        - When |position| > soft_limit (configurable), we add a
          flatten-only quote at wall_mid. It only fills when a
          counterparty is willing to trade through wall_mid - in
          practice the TAKE-unwind above catches most of this; the
          wall_mid quote is a safety net.
        - Hedgehogs' Resin did wall_mid flattening; their Kelp did not.
          We port it to Kelp because the wall is stable enough to treat
          wall_mid as a credible exit.

==========================================================================
Why this edge exists (and persists)
==========================================================================

    * The 16-tick wall and the 2-tick noise behaviour are properties of
      the bot market maker, not of any individual trading team. The edge
      we are extracting is: we front-run the noise bot's queue position
      by one tick, capturing the structural spread that the noise bot
      was earning between its passive fills and the aggressor's price.

    * If the noise bot changes its behaviour (e.g. starts quoting
      tighter), our wall-mid anchor still holds because the WALL (deep
      level) is separate liquidity and is observably more stable. The
      only case where this trader would degrade is if the wall itself
      collapses, at which point wall_mid becomes unreliable and our
      `if self.wall_mid is None / too tight` guards kick in and we
      stop trading.

==========================================================================
Explicit non-uses (philosophical commitments)
==========================================================================

    NO EMA / moving average / rolling window fair value.
    NO regression, no linear model, no learned coefficient.
    NO backtest-tuned thresholds beyond what microstructure prescribes
        (all constants are derived from observed wall/noise distances,
        not parameter-searched).
    NO speculation on directional order flow.
    NO prediction of the next wall_mid.
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

# TOMATOES constants derived directly from observed microstructure.
# These are NOT parameters to search. They are the distances that the
# order book shows us.
TOMATO_TAKE_EDGE = 2         # require at least 2 ticks of edge to TAKE
TOMATO_NOISE_STEP = 2        # target MAKE quote at wall+2 / wall-2 (one inside noise)
TOMATO_SOFT_INVENTORY = 40   # half of position_limit; unwind pressure kicks in above this


# ============================================================
# ProductTrader base class
# (ported from FrankfurtHedgehogs_polished.py / trader_wallmid_v1.py)
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
        """
        bid_wall  = MIN bid price in the book (deepest visible bid level)
        ask_wall  = MAX ask price in the book (deepest visible ask level)
        wall_mid  = (bid_wall + ask_wall) / 2
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
# EmeraldsTrader — UNCHANGED Hedgehogs StaticTrader port.
# This is verified already-optimal (14,945 PnL in both modes).
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

            # Phase 2: MAKE with dynamic overbid/underbid
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
# TomatoesTrader — microstructure purist evolution of Hedgehogs Kelp.
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
        # Guard: if wall is degenerate (rare), bail.
        if wall_spread < 4:
            return {self.name: self.orders}

        fv = self.wall_mid  # fair value

        self.log('WM', fv)
        self.log('BW', self.bid_wall)
        self.log('AW', self.ask_wall)
        self.log('POS', self.initial_position)

        # ==========================================================
        # Phase 1: TAKE
        # ==========================================================
        # Buy any ask <= wall_mid - TAKE_EDGE (at least 2 ticks of edge).
        # Plus an UNWIND TAKE at wall_mid when we're short.
        for sp, sv in self.mkt_sell_orders.items():
            if sp <= fv - TOMATO_TAKE_EDGE:
                self.bid(sp, sv, logging=False)
            elif sp <= fv and self.initial_position < 0:
                # Zero-edge unwind of a short position.
                volume = min(sv, abs(self.initial_position))
                self.bid(sp, volume, logging=False)
            else:
                break  # sell_orders sorted ascending, no more matches possible

        # Mirror: sell any bid >= wall_mid + TAKE_EDGE.
        for bp, bv in self.mkt_buy_orders.items():
            if bp >= fv + TOMATO_TAKE_EDGE:
                self.ask(bp, bv, logging=False)
            elif bp >= fv and self.initial_position > 0:
                volume = min(bv, self.initial_position)
                self.ask(bp, volume, logging=False)
            else:
                break

        # ==========================================================
        # Phase 2: MAKE
        # ==========================================================
        # Default quote: one tick inside the noise quote -> bid_wall+2.
        # This is the key difference from Hedgehogs Kelp (bid_wall+1).
        # Rationale: noise quote sits 1 tick above bid_wall 75% of the
        # time, 2 ticks 24% of the time. Sitting at bid_wall+2 therefore
        # front-runs the noise ~75% of the time and ties with it ~24%.
        bid_price = int(self.bid_wall + TOMATO_NOISE_STEP)
        ask_price = int(self.ask_wall - TOMATO_NOISE_STEP)

        # Adaptive: if best_bid (the noise) has intruded MORE than our
        # default step (i.e. best_bid > bid_wall + TOMATO_NOISE_STEP),
        # we match (don't overpay) - preserves edge.
        if self.best_bid is not None and self.best_bid >= bid_price:
            # Match the noise, don't step ahead (still credible queue
            # position because noise size is small).
            bid_price = self.best_bid
            # But we must preserve edge: never quote at or above wall_mid.
            if bid_price >= fv:
                bid_price = int(fv - 1)

        if self.best_ask is not None and self.best_ask <= ask_price:
            ask_price = self.best_ask
            if ask_price <= fv:
                ask_price = int(fv + 1)

        # Wall-mid guard: never cross.
        if bid_price >= fv:
            bid_price = int(self.bid_wall + 1)
        if ask_price <= fv:
            ask_price = int(self.ask_wall - 1)

        self.log('MK_B', bid_price)
        self.log('MK_A', ask_price)

        # ==========================================================
        # Inventory flattening overlay
        # ==========================================================
        # When we are meaningfully long, add a strong sell quote at
        # wall_mid (zero-edge exit). When short, mirror.
        # This is Hedgehogs Resin behaviour ported to a dynamic product.
        if self.initial_position > TOMATO_SOFT_INVENTORY:
            flatten_size = self.initial_position - TOMATO_SOFT_INVENTORY
            # Try to unwind at wall_mid first (no edge given up vs FV).
            self.ask(int(fv), flatten_size, logging=False)

        if self.initial_position < -TOMATO_SOFT_INVENTORY:
            flatten_size = abs(self.initial_position) - TOMATO_SOFT_INVENTORY
            self.bid(int(fv), flatten_size, logging=False)

        # Post remaining capacity as passive quotes.
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
