"""
IMC Prosperity 4 - Round 0 (Tutorial)
Phase 3 / Synthesis Agent: trader_final_v1
==============================================

Merges the best ideas from the Phase 2 agents:

  * Phase 2 / D (TOMATOES champion, 31,657 conservative):
      - Adaptive noise-quote-tracking MAKE rule
      - bid_price = max(bid_wall + 2, best_bid + 1)  (mirror for asks)

  * Phase 2 / E (EMERALDS ceiling + free optimistic option):
      - Two-level split quote on EMERALDS (9992 + 9993 on bid,
        10008 + 10007 on ask). Under CONSERVATIVE matching the
        9992/10008 layer does not fill (strict-< exclusion) so
        behaviour ties the 14,945 ceiling. Under OPTIMISTIC matching
        the 9992/10008 layer catches every historical noise trade at
        +8 edge (+2,135 lift).
      - Order-submission ORDER is critical: 9992 MUST go BEFORE 9993
        (and 10008 BEFORE 10007) so the optimistic matcher processes
        the better-priced layer first.

  * Phase 2 / F attack report (every actionable defensive guard):
      - 3A  Noise-bot-presence guard on TOMATOES MAKE side.
      - 3B  Pre-clamp combined per-side volume <= position limit.
      - 3C  Half-integer wall_mid handling for flatten quotes.
      - 3D  Error counters replacing bare `except Exception: pass`.
      - 3E  `expected_position` field removed (never used, landmine).
      - 3F  Adverse-selection brake: MEASURED AND SKIPPED. A one-
            tick brake costs -341 conservative PnL with negligible
            optimistic upside - see TomatoesTrader.get_orders() for
            full measurement and reasoning. State is still persisted
            so a future tighter-trigger brake can be re-enabled.

Design notes
------------

  * EMERALDS logic is Phase 2 / E's split-quote, unchanged in effect.
    The TAKE phase of E is preserved byte-for-byte because the
    "zero-edge 10000 unwind" is load-bearing (free inventory rotation
    that keeps buy capacity open for +7/+8 fills - removing it is a
    direct PnL loss per E's measurement).

  * TOMATOES logic starts from Phase 2 / D (the champion) and adds
    only the defensive guards from F. No parameter search, no new
    fitted constants - every change is mechanism-driven.

  * We keep the EMERALDS half-integer case out of scope: EMERALDS
    wall_mid is literally 10000 in 100% of backtest ticks, so the
    half-integer flatten bug does not affect EMERALDS. The fix is
    applied to the TOMATOES flatten quotes where the bug actually
    lives (spread-15 / spread-17 ticks, ~3% combined).

  * The `expected_position` field is deleted. The pre-clamp guard
    (3B) makes it unnecessary, and F flagged it as a future-state
    landmine.
"""

import json
import math
from typing import Dict, List, Optional

from datamodel import Order, OrderDepth, TradingState, UserId


# ============================================================
# Module-level constants
# ============================================================
EMERALDS_SYMBOL = 'EMERALDS'
TOMATOES_SYMBOL = 'TOMATOES'

POS_LIMITS: Dict[str, int] = {
    EMERALDS_SYMBOL: 80,
    TOMATOES_SYMBOL: 80,
}

# --- TOMATOES ---
TOMATO_TAKE_EDGE = 2
TOMATO_BID_FLOOR_STEP = 2      # default bid offset above bid_wall
TOMATO_ASK_FLOOR_STEP = 2      # default ask offset below ask_wall
TOMATO_SOFT_INVENTORY = 40
TOMATO_ADVERSE_BRAKE_TICKS = 1  # how many ticks to widen by one after adverse drift

# --- EMERALDS ---
EMERALDS_BID_INSIDE = 9993      # conservative-mode optimum (+7 edge)
EMERALDS_ASK_INSIDE = 10007     # mirror
EMERALDS_NOISE_BID_TIE = 9992   # optimistic-mode optimum (+8 edge, ties noise)
EMERALDS_NOISE_ASK_TIE = 10008  # mirror


# ============================================================
# Base ProductTrader
# ============================================================
class ProductTrader:
    """
    Common utilities. Subclasses override get_orders().

    Key changes vs phase2_D:
      * `expected_position` field removed (F Rec. 3E).
      * Errors increment an `error_count` field in trader_data rather
        than being silently swallowed (F Rec. 3D).
      * A `_finalise_orders()` helper applies a belt-and-braces
        pre-clamp so combined per-side volume never exceeds the
        position limit (F Rec. 3B).
    """

    def __init__(self, name: str, state: TradingState, prints: dict,
                 new_trader_data: dict, product_group: Optional[str] = None):
        self.orders: List[Order] = []

        self.name = name
        self.state = state
        self.prints = prints
        self.new_trader_data = new_trader_data
        self.product_group = name if product_group is None else product_group

        self.last_traderData = self._get_last_traderData()

        self.position_limit = POS_LIMITS.get(self.name, 0)
        self.initial_position = self.state.position.get(self.name, 0)

        self.mkt_buy_orders, self.mkt_sell_orders = self._get_order_depth()
        self.bid_wall, self.wall_mid, self.ask_wall = self._get_walls()
        self.best_bid, self.best_ask = self._get_best_bid_ask()

        (self.max_allowed_buy_volume,
         self.max_allowed_sell_volume) = self._get_max_allowed_volume()

    # ---------- safe readers (no silent pass) ----------
    def _record_error(self, where: str, exc: Exception) -> None:
        """F Rec. 3D: track exceptions rather than swallowing them."""
        errs = self.new_trader_data.setdefault('errors', {})
        key = f"{where}:{type(exc).__name__}"
        errs[key] = errs.get(key, 0) + 1

    def _get_last_traderData(self) -> dict:
        try:
            if self.state.traderData:
                return json.loads(self.state.traderData)
        except Exception as exc:
            self._record_error('last_traderData', exc)
        return {}

    def _get_order_depth(self):
        buy_orders: Dict[int, int] = {}
        sell_orders: Dict[int, int] = {}
        try:
            order_depth: OrderDepth = self.state.order_depths[self.name]
        except Exception as exc:
            self._record_error('order_depth_get', exc)
            return buy_orders, sell_orders
        try:
            buy_orders = {
                bp: abs(bv)
                for bp, bv in sorted(order_depth.buy_orders.items(),
                                     key=lambda x: x[0], reverse=True)
            }
        except Exception as exc:
            self._record_error('order_depth_buy', exc)
        try:
            sell_orders = {
                sp: abs(sv)
                for sp, sv in sorted(order_depth.sell_orders.items(),
                                     key=lambda x: x[0])
            }
        except Exception as exc:
            self._record_error('order_depth_sell', exc)
        return buy_orders, sell_orders

    def _get_best_bid_ask(self):
        best_bid = best_ask = None
        try:
            if len(self.mkt_buy_orders) > 0:
                best_bid = max(self.mkt_buy_orders.keys())
            if len(self.mkt_sell_orders) > 0:
                best_ask = min(self.mkt_sell_orders.keys())
        except Exception as exc:
            self._record_error('best_bid_ask', exc)
        return best_bid, best_ask

    def _get_walls(self):
        bid_wall = ask_wall = wall_mid = None
        try:
            if len(self.mkt_buy_orders) > 0:
                bid_wall = min(self.mkt_buy_orders.keys())
        except Exception as exc:
            self._record_error('bid_wall', exc)
        try:
            if len(self.mkt_sell_orders) > 0:
                ask_wall = max(self.mkt_sell_orders.keys())
        except Exception as exc:
            self._record_error('ask_wall', exc)
        if bid_wall is not None and ask_wall is not None:
            wall_mid = (bid_wall + ask_wall) / 2
        return bid_wall, wall_mid, ask_wall

    def _get_max_allowed_volume(self):
        return (
            self.position_limit - self.initial_position,
            self.position_limit + self.initial_position,
        )

    # ---------- order helpers ----------
    def bid(self, price, volume) -> None:
        v = min(abs(int(volume)), self.max_allowed_buy_volume)
        if v <= 0:
            return
        self.max_allowed_buy_volume -= v
        self.orders.append(Order(self.name, int(price), v))

    def ask(self, price, volume) -> None:
        v = min(abs(int(volume)), self.max_allowed_sell_volume)
        if v <= 0:
            return
        self.max_allowed_sell_volume -= v
        self.orders.append(Order(self.name, int(price), -v))

    # ---------- F Rec. 3B: pre-clamp guard ----------
    def _finalise_orders(self) -> List[Order]:
        """
        Belt-and-braces guard: if for any reason the total long or
        total short volume on the orders list would breach the
        position limit (which runner.py:137-140 punishes by dropping
        ALL orders for the product silently), clip the LAST orders
        on the offending side in-place until the sum fits.

        We clip tail-first because the earlier orders in the list are
        higher-priority (TAKE first, flatten next, MAKE last). That
        matches F's recommendation: clip MAKE before TAKE.
        """
        pos = self.initial_position
        limit = self.position_limit

        def clip_side(sign: int) -> None:
            # sign=+1 clips positive-qty orders (longs)
            # sign=-1 clips negative-qty orders (shorts)
            if sign > 0:
                budget = limit - pos       # max new long
            else:
                budget = limit + pos       # max new short

            if budget < 0:
                budget = 0

            # Sum current side volume
            total = sum(o.quantity for o in self.orders if (o.quantity > 0 if sign > 0 else o.quantity < 0))
            total = total if sign > 0 else -total
            if total <= budget:
                return

            overflow = total - budget
            # Clip from the end of the list, tail-first
            for i in range(len(self.orders) - 1, -1, -1):
                o = self.orders[i]
                if (sign > 0 and o.quantity > 0) or (sign < 0 and o.quantity < 0):
                    qty = o.quantity if sign > 0 else -o.quantity
                    if qty <= overflow:
                        overflow -= qty
                        self.orders[i] = Order(o.symbol, o.price, 0)
                    else:
                        new_qty = qty - overflow
                        overflow = 0
                        if sign > 0:
                            self.orders[i] = Order(o.symbol, o.price, new_qty)
                        else:
                            self.orders[i] = Order(o.symbol, o.price, -new_qty)
                    if overflow == 0:
                        break

        clip_side(+1)
        clip_side(-1)

        # Drop zero-qty leftovers
        self.orders = [o for o in self.orders if o.quantity != 0]
        return self.orders

    def get_orders(self) -> Dict[str, List[Order]]:
        return {}


# ============================================================
# EmeraldsTrader -- phase2_E split-quote (byte-for-byte effect)
# ============================================================
class EmeraldsTrader(ProductTrader):
    """
    Strategy:
      * TAKE: buy any ask strictly inside wall_mid (positive edge) and
        at wall_mid when short (free unwind). Mirror on asks. Verbatim
        from phase1_B / phase2_E.
      * MAKE: split quote. Bid 9992 BEFORE 9993 (optimistic gets the
        +8 layer first); symmetric asks with 10008 BEFORE 10007.
    """

    def __init__(self, state: TradingState, prints: dict,
                 new_trader_data: dict):
        super().__init__(EMERALDS_SYMBOL, state, prints, new_trader_data)

    def get_orders(self) -> Dict[str, List[Order]]:
        if self.wall_mid is None:
            return {self.name: self._finalise_orders()}

        # ---------- Phase 1: TAKE (unchanged) ----------
        for sp, sv in self.mkt_sell_orders.items():
            if sp <= self.wall_mid - 1:
                self.bid(sp, sv)
            elif sp <= self.wall_mid and self.initial_position < 0:
                volume = min(sv, abs(self.initial_position))
                self.bid(sp, volume)

        for bp, bv in self.mkt_buy_orders.items():
            if bp >= self.wall_mid + 1:
                self.ask(bp, bv)
            elif bp >= self.wall_mid and self.initial_position > 0:
                volume = min(bv, self.initial_position)
                self.ask(bp, volume)

        # ---------- Phase 2: MAKE split quote ----------
        cap_buy = self.max_allowed_buy_volume
        cap_sell = self.max_allowed_sell_volume

        bid_inside_size = (cap_buy + 1) // 2   # 9993 layer
        bid_tie_size = cap_buy // 2            # 9992 layer
        ask_inside_size = (cap_sell + 1) // 2  # 10007 layer
        ask_tie_size = cap_sell // 2           # 10008 layer

        # CRITICAL: 9992 BEFORE 9993 (optimistic optimum first).
        self.bid(EMERALDS_NOISE_BID_TIE, bid_tie_size)
        self.bid(EMERALDS_BID_INSIDE, bid_inside_size)
        # Mirror: 10008 BEFORE 10007.
        self.ask(EMERALDS_NOISE_ASK_TIE, ask_tie_size)
        self.ask(EMERALDS_ASK_INSIDE, ask_inside_size)

        return {self.name: self._finalise_orders()}


# ============================================================
# TomatoesTrader -- phase2_D + F's defensive guards
# ============================================================
class TomatoesTrader(ProductTrader):
    """
    Phase 2 / D adaptive noise-quote MAKE rule, plus:
      3A   noise-bot-presence guard
      3C   half-integer wall_mid flatten fix
      3F   adverse-selection one-tick brake
    Identical TAKE logic and inventory-flatten thresholds.
    """

    def __init__(self, state: TradingState, prints: dict,
                 new_trader_data: dict):
        super().__init__(TOMATOES_SYMBOL, state, prints, new_trader_data)

    # ---------- F Rec. 3A: noise-bot presence guard ----------
    #
    # phase2_D's rule is bid_price = max(bid_wall+2, best_bid+1). If the
    # noise bot DISAPPEARS (best_bid collapses to bid_wall), best_bid+1
    # = bid_wall+1, which is BELOW the default floor bid_wall+2. D would
    # silently fall back to bid_wall+2 and keep quoting alone with no
    # queue priority story at all (F says live PnL collapses in this
    # regime).
    #
    # Guard policy (documented trade-off):
    #   - If best_bid == bid_wall  (no visible noise quote ABOVE wall):
    #       REFUSE to post a MAKE bid this tick. We do not blindly fall
    #       back to bid_wall+1 because (a) in the backtest data this
    #       never happens, so we can't validate that fallback, and
    #       (b) if it DOES happen live it is a regime change we have
    #       no edge in; sitting out is the defensively correct move.
    #       TAKE and flatten still execute.
    #   - Mirror for asks.
    #
    # The brief offers two choices (bw+1 fallback OR refuse). We pick
    # REFUSE because the brief explicitly says "either one OR refuse"
    # and F's risk model ranks a silent wrong quote higher than no
    # quote. In backtest this changes nothing (the guard never fires,
    # see phase2_E analysis: best_bid == 9992 on 100% of TOMATOES
    # ticks with spread>=4).

    # ---------- F Rec. 3C: half-integer wall_mid flatten ----------
    #
    # When wall_spread is 15 or 17 (~3% of ticks combined), wall_mid
    # is a half-integer e.g. 10000.5. `int(wall_mid)` truncates to
    # 10000, which is BELOW fair value. Using `int()` for a flatten
    # ASK posts a sell at 0.5 ticks below FV, costing us. Fix:
    #   * Flatten ASK price: math.ceil(wall_mid)    (round up: >= FV)
    #   * Flatten BID price: math.floor(wall_mid)   (round down: <= FV)
    # Both are favourable rounding.

    def get_orders(self) -> Dict[str, List[Order]]:
        if self.wall_mid is None or self.bid_wall is None or self.ask_wall is None:
            return {self.name: self._finalise_orders()}

        wall_spread = self.ask_wall - self.bid_wall
        if wall_spread < 4:
            return {self.name: self._finalise_orders()}

        fv = self.wall_mid

        # ---------- Phase 1: TAKE (unchanged from D/B) ----------
        for sp, sv in self.mkt_sell_orders.items():
            if sp <= fv - TOMATO_TAKE_EDGE:
                self.bid(sp, sv)
            elif sp <= fv and self.initial_position < 0:
                volume = min(sv, abs(self.initial_position))
                self.bid(sp, volume)
            else:
                break

        for bp, bv in self.mkt_buy_orders.items():
            if bp >= fv + TOMATO_TAKE_EDGE:
                self.ask(bp, bv)
            elif bp >= fv and self.initial_position > 0:
                volume = min(bv, self.initial_position)
                self.ask(bp, volume)
            else:
                break

        # ---------- Phase 2: MAKE (D rule + F guards) ----------
        # Default floors from phase1_B.
        bid_price = int(self.bid_wall + TOMATO_BID_FLOOR_STEP)
        ask_price = int(self.ask_wall - TOMATO_ASK_FLOOR_STEP)

        # Adaptive noise-detect (D's rule).
        if self.best_bid is not None:
            bid_price = max(bid_price, int(self.best_bid + 1))
        if self.best_ask is not None:
            ask_price = min(ask_price, int(self.best_ask - 1))

        # Wall-mid guard (unchanged).
        if bid_price >= fv:
            bid_price = int(self.bid_wall + 1)
        if ask_price <= fv:
            ask_price = int(self.ask_wall - 1)

        # F Rec. 3A: refuse to MAKE this side if no noise visible.
        post_bid = True
        post_ask = True
        if self.best_bid is None or self.best_bid <= self.bid_wall:
            post_bid = False
            self.new_trader_data.setdefault('tom_noise_missing_bid', 0)
            self.new_trader_data['tom_noise_missing_bid'] += 1
        if self.best_ask is None or self.best_ask >= self.ask_wall:
            post_ask = False
            self.new_trader_data.setdefault('tom_noise_missing_ask', 0)
            self.new_trader_data['tom_noise_missing_ask'] += 1

        # F Rec. 3F: adverse-selection one-tick brake.
        #
        # SKIPPED after backtest measurement. We implemented the brake
        # exactly as F recommended (step BUY MAKE out by one tick when
        # wall_mid has drifted down by >=1 tick following a MAKE BUY
        # fill, mirror for asks) and measured its impact:
        #
        #   Config                          Cons      Opt
        #   no brake                        31,671    33,950
        #   with brake                      31,330    33,972
        #   delta (brake on)                -341      +22
        #
        # The brake causes a measurable conservative-mode regression
        # (-341, ~1% of total PnL). F's own wording said "Implement
        # only if you can justify it doesn't create a whipsaw. If
        # unsure, document why you skipped." The backtest shows the
        # brake DOES create whipsaw: after adverse drift we step out
        # and miss the very trades that mean-revert. The optimistic
        # gain is negligible and the conservative loss is real.
        #
        # We still persist prev_wm and prev_pos in traderData so that
        # the next iteration of this trader (or a live rollout with a
        # tighter trigger e.g. dwm <= -2) can re-enable the brake
        # without another state-surgery pass.
        self.new_trader_data['tom_prev_wm'] = fv
        self.new_trader_data['tom_prev_pos'] = self.initial_position

        # ---------- Inventory flattening overlay ----------
        # F Rec. 3C: favourable rounding on half-integer fv.
        bid_flatten_price = int(math.floor(fv))
        ask_flatten_price = int(math.ceil(fv))

        if self.initial_position > TOMATO_SOFT_INVENTORY:
            flatten_size = self.initial_position - TOMATO_SOFT_INVENTORY
            self.ask(ask_flatten_price, flatten_size)

        if self.initial_position < -TOMATO_SOFT_INVENTORY:
            flatten_size = abs(self.initial_position) - TOMATO_SOFT_INVENTORY
            self.bid(bid_flatten_price, flatten_size)

        # ---------- Post MAKE (with guard) ----------
        if post_bid:
            self.bid(bid_price, self.max_allowed_buy_volume)
        if post_ask:
            self.ask(ask_price, self.max_allowed_sell_volume)

        return {self.name: self._finalise_orders()}


# ============================================================
# Main Trader entry point
# ============================================================
class Trader:
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        new_trader_data: Dict = {}

        # Carry forward persistent counters from previous tick so
        # brake / error counters accumulate across ticks.
        try:
            if state.traderData:
                prev = json.loads(state.traderData)
                # Preserve everything we want to persist across ticks.
                for k in ('errors', 'tom_prev_wm', 'tom_prev_pos',
                          'tom_noise_missing_bid', 'tom_noise_missing_ask'):
                    if k in prev:
                        new_trader_data[k] = prev[k]
        except Exception as exc:
            errs = new_trader_data.setdefault('errors', {})
            key = f'run_load:{type(exc).__name__}'
            errs[key] = errs.get(key, 0) + 1

        prints: dict = {}

        product_traders = {
            EMERALDS_SYMBOL: EmeraldsTrader,
            TOMATOES_SYMBOL: TomatoesTrader,
        }

        for symbol, cls in product_traders.items():
            if symbol in state.order_depths:
                try:
                    trader = cls(state, prints, new_trader_data)
                    result.update(trader.get_orders())
                except Exception as exc:
                    errs = new_trader_data.setdefault('errors', {})
                    key = f'{symbol}_run:{type(exc).__name__}'
                    errs[key] = errs.get(key, 0) + 1

        try:
            final_trader_data = json.dumps(new_trader_data)
        except Exception:
            final_trader_data = ''

        return result, 0, final_trader_data
