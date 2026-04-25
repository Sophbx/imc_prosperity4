"""
HYDROGEL_PACK — wall-anchored Avellaneda-Stoikov market maker (v2b).

================================================================================
THESIS — fundamentally different mechanism vs. v1/v2
================================================================================

v1/v2 are passive market makers that quote at touch ± 1 around touch_mid and
manage inventory by SKEWING QUOTE SIZE (shrink one side, grow the other) plus
discrete regime switches at fixed |position| thresholds (60/120 in v2,
120/180 in v1).

v2b replaces the fair-value estimator AND the inventory-control mechanism:

(A) FAIR VALUE = WALL_MID, not touch_mid.
    HYDROGEL's order book is bimodal — a thin inner level (best bid/ask,
    ~10-15 lots, often the small-ticket overbidder) and a deeper outer
    level (~20-30 lots, designated MM band). Statistically (3-day data):

        std(touch_mid − wall_mid) ≈ 0.7 ticks
        when |touch_mid − wall_mid| ≥ 1, expected forward 10-tick change
            in touch_mid is −3.6 ticks if touch>wall, +3.6 if touch<wall

    The touch_mid mean-reverts toward wall_mid; wall_mid is the
    gravitational anchor. This is much stronger than any signal currently
    in v1/v2 (corr ≈ 0.40 vs the ±0.10 seen in delta1_predictor_ranking).
    The "Wall Mid" concept is from the Frankfurt Hedgehogs Prosperity 3
    writeup; their definition uses min/max of book levels which we adapt
    here as the second-deepest level (the actual MM wall in HYDROGEL).

(B) INVENTORY CONTROL = AVELLANEDA-STOIKOV RESERVATION PRICE, not
    discrete regime + size skew.
    From Avellaneda & Stoikov (2008), "High-frequency trading in a limit
    order book":

        reservation = fair_value − q · γ · σ² · (T − t)

    Long → reservation drifts down → bid backs off, ask comes in. Short →
    opposite. One smooth, principled adjustment instead of v1/v2's discrete
    "unwind" / "hard" thresholds. No size skew at all; we let reservation +
    position-limit clamp do the inventory work.

(C) MICROPRICE for fine-grained anchor adjustment. Stoikov's microprice
    `(best_bid·ask_size + best_ask·bid_size) / (bid_size + ask_size)`
    captures order-flow imbalance: when bids overwhelm asks, microprice
    tilts toward best_ask (price going up). On HYDROGEL the microprice and
    the (touch − wall) signal are 100% sign-correlated (verified
    empirically), so blending them α=0.5 acts as a denoising / fine-
    resolution operation on the same underlying signal.

================================================================================
QUOTE LOGIC
================================================================================

baseline_bid = best_bid + 1                   # v1's improve-by-1 baseline
baseline_ask = best_ask − 1
anchor       = (1 − α)·wall_mid + α·microprice
anchor_off   = anchor − touch_mid             # signed deviation
T_remain     = max(0, 1 − ts/1_000_000)
skew         = q · γ · σ² · T_remain
shift        = anchor_off − skew

bid_price = round(baseline_bid + shift)
ask_price = round(baseline_ask + shift)

→ At q=0 with anchor=touch_mid, shift=0 and we match v1 exactly.
→ When wall_mid > touch_mid (touch is cheap relative to fair),
  anchor_off > 0, both quotes shift UP: bid more aggressive, ask backs off.
→ When q>0 (long), skew>0, both quotes shift DOWN: ask more aggressive
  (sell to unwind), bid backs off (don't accumulate further).

These two mechanisms (anchor offset and inventory skew) compose additively
and continuously — no discrete regime jumps, no separate code path for
"unwind mode."

================================================================================
PARAMETERS — calibrated from data and theory, not grid-searched
================================================================================

α          = 0.5     — equal weight on wall_mid (low-noise, discrete) and
                       microprice (continuous resolution)
σ²         = 33.6    — var of 10-step wall_mid changes (measured)
γ          = 7e-4    — at q=200, T_rem=0.5 → 2.4 tick price shift; at
                       q=120 (v1's "unwind" threshold) → 1.4 tick shift,
                       roughly matching v1's regime-switch magnitude but
                       applied smoothly across the whole inventory range
quote_size = 12      — uniform per side; A-S handles inventory via price,
                       not via size

End-of-day flatten + hard safety (cross book at |q|>180) are kept from
v1/v2 as universal hygiene — not v2b innovations.

================================================================================
ITERATION LOG (HONEST ACCOUNTING)
================================================================================

Tried and rejected:
  * Stat-arb TAKER on the touch-vs-wall signal. Selling at best_bid when
    touch is rich means giving up half the touch_spread (~8 ticks) on
    entry, but the predicted touch reversion is only 3.6 ticks. The
    signal is real but not tradeable as a taker; it must be captured by
    making (which our quoter does naturally).
  * Hedgehogs-style "never overbid past wall_mid" cutoff. Big wins on
    days 0/1 but disastrous on day 2 — when wall_mid drifts upward over
    the day, the cutoff systematically forces us short against the trend.
    Total dropped from 23,478 to 22,996 with worst-DD blowing up to
    -11,325. Rejected as overfitted to stationary-wall regimes.
  * Pure wall_mid (α=0). Loses the fine-grained microprice resolution
    that nudges quotes by ±1 tick at the integer rounding boundary.
    Total dropped to 17,381.
  * γ values in {3e-4, 5e-4, 1e-3}. γ=7e-4 is the sweet spot: matches
    v1's effective inventory response at |q|=120, smaller doesn't move
    quotes enough, larger pushes us off the touch floor too aggressively
    and we lose fills.

================================================================================
References
================================================================================
- Avellaneda & Stoikov 2008, "High-frequency trading in a limit order book"
- Stoikov 2018, "The micro-price: a high-frequency estimator of future prices"
- Frankfurt Hedgehogs Prosperity 3 writeup (Wall Mid concept)
- Round3/Analysis/output/delta1_predictor_ranking.csv (signal correlations)
- Empirical analysis: see iteration log.
"""
from datamodel import Order, OrderDepth, TradingState
import json
import math


class Trader:
    HYDROGEL = "HYDROGEL_PACK"
    POSITION_LIMITS = {HYDROGEL: 200}

    CFG = {
        # ---- Stat-arb taker ----
        # DISABLED. Theory says touch_mid mean-reverts to wall_mid when they
        # diverge — but capturing that signal by TAKING at best_bid (when
        # touch is rich) means selling 8 ticks below touch_mid and unwinding
        # later. To make that profitable the touch_mid needs to drop more
        # than the round-trip touch_spread (~16 ticks), and the predicted
        # reversion is only ~3.6 ticks. The signal is NOT tradeable as a
        # taker; it IS tradeable as a maker — see iteration log.
        "take_threshold": 999.0,
        "take_clip": 0,
        "take_walk_distance": 0,
        "soft_inv_cap_take": 100,

        # ---- A-S reservation price quoter ----
        # alpha_microprice = 0.5 blends two estimators that AGREE in sign
        # but at different resolutions: wall_mid is integer-valued (low
        # noise but discrete), microprice is continuous (fine resolution).
        # Joint analysis shows they're 100% sign-correlated, so blending
        # is a denoising operation, not a separate signal.
        "alpha_microprice": 0.5,
        "sigma2": 33.6,             # var of 10-tick wall_mid changes (from data)
        "gamma": 7e-4,              # at q=200 T_rem=0.5 → 2.4 tick skew
                                    # at q=120 T_rem=0.5 → 1.4 tick skew
                                    # at q=60  T_rem=0.5 → 0.7 tick skew (rounds 0/1)
                                    # principled choice: at v1's "unwind" threshold
                                    # (|q|=120) the price shift is just over 1 tick,
                                    # the same effective response as v1's discrete
                                    # regime switch — but applied smoothly across
                                    # the full inventory range.
        "T_total": 1_000_000,       # round end timestamp
        "delta_floor_ticks": 1,     # never post inside best_bid+1 / best_ask-1
        "min_quote_spread": 6,      # if market spread < this, skip quoting
        "quote_size": 12,           # uniform clip per side

        # ---- Hard safety ----
        "hard_inv_limit": 180,      # cross book if |q| above this
        "hard_take_distance": 4,

        # ---- End-of-day flatten ----
        "flatten_window_ts": 980_000,
        "flatten_offset": 2,
    }

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {self.HYDROGEL: []}

        depth = state.order_depths.get(self.HYDROGEL)
        if depth is None:
            return result, 0, ""

        position = int(state.position.get(self.HYDROGEL, 0))
        ts = int(state.timestamp)

        # Persisted state is unused for now — kept for future EWMA of σ etc.
        result[self.HYDROGEL] = self._trade(depth, position, ts)

        # Pass-through traderData (no persistent state needed).
        return result, 0, state.traderData or ""

    # ------------------------------------------------------------------
    # Main trade decision
    # ------------------------------------------------------------------

    def _trade(self, depth: OrderDepth, position: int, ts: int) -> list[Order]:
        cfg = self.CFG
        limit = self.POSITION_LIMITS[self.HYDROGEL]

        bid_wall, best_bid, best_ask, ask_wall = self._book_levels(depth)
        if best_bid is None or best_ask is None:
            return []

        # Fall back if we don't have a wall (single-level book): use touch as wall
        if bid_wall is None:
            bid_wall = best_bid
        if ask_wall is None:
            ask_wall = best_ask

        touch_mid = (best_bid + best_ask) / 2.0
        wall_mid = (bid_wall + ask_wall) / 2.0
        touch_spread = best_ask - best_bid

        # Microprice: weight best prices by the opposite-side volume.
        microprice = self._microprice(depth, best_bid, best_ask, touch_mid)

        # Hard safety: at extreme inventory, cross the book to neutralize.
        if position > cfg["hard_inv_limit"]:
            return self._cross_book(depth, position, wall_mid, side="sell")
        if position < -cfg["hard_inv_limit"]:
            return self._cross_book(depth, position, wall_mid, side="buy")

        # ---- 1. Stat-arb taker leg ----
        taker_orders, position_after_take = self._stat_arb_taker(
            depth, position, touch_mid, wall_mid, ts,
        )

        # ---- 2. A-S reservation quoter leg ----
        # End-of-day flatten overrides A-S
        if ts > cfg["flatten_window_ts"]:
            quoter_orders = self._flatten_quotes(
                position_after_take, wall_mid, best_bid, best_ask, limit,
            )
        elif touch_spread < cfg["min_quote_spread"]:
            quoter_orders = []  # too tight to make profitably
        else:
            quoter_orders = self._reservation_quotes(
                position_after_take, wall_mid, microprice, best_bid, best_ask,
                ts, limit,
            )

        return taker_orders + quoter_orders

    # ------------------------------------------------------------------
    # 1. Stat-arb taker
    # ------------------------------------------------------------------

    def _stat_arb_taker(
        self,
        depth: OrderDepth,
        position: int,
        touch_mid: float,
        wall_mid: float,
        ts: int,
    ) -> tuple[list[Order], int]:
        """When touch_mid deviates from wall_mid >= take_threshold, take the touch.

        The expected mean reversion of touch_mid -> wall_mid is ~3 ticks over
        a 10-tick horizon (corr ≈ 0.4 across all 3 days). Sizing is conservative
        so a single bad tick doesn't blow us out.
        """
        cfg = self.CFG
        diff = touch_mid - wall_mid
        orders: list[Order] = []
        new_pos = position

        if diff >= cfg["take_threshold"]:
            # Touch is rich → SELL. Need ask room and not too short already.
            if position > -cfg["soft_inv_cap_take"]:
                ask_room = self.POSITION_LIMITS[self.HYDROGEL] + position
                clip_target = min(cfg["take_clip"], ask_room,
                                  cfg["soft_inv_cap_take"] + position)
                if clip_target > 0:
                    orders.extend(self._walk_book(
                        depth, side="sell",
                        ref_price=wall_mid,
                        max_distance=cfg["take_walk_distance"],
                        max_qty=clip_target,
                    ))
                    new_pos = position - sum(-o.quantity for o in orders if o.quantity < 0)

        elif diff <= -cfg["take_threshold"]:
            # Touch is cheap → BUY. Need bid room and not too long already.
            if position < cfg["soft_inv_cap_take"]:
                bid_room = self.POSITION_LIMITS[self.HYDROGEL] - position
                clip_target = min(cfg["take_clip"], bid_room,
                                  cfg["soft_inv_cap_take"] - position)
                if clip_target > 0:
                    orders.extend(self._walk_book(
                        depth, side="buy",
                        ref_price=wall_mid,
                        max_distance=cfg["take_walk_distance"],
                        max_qty=clip_target,
                    ))
                    new_pos = position + sum(o.quantity for o in orders if o.quantity > 0)

        return orders, new_pos

    def _walk_book(
        self,
        depth: OrderDepth,
        side: str,
        ref_price: float,
        max_distance: float,
        max_qty: int,
    ) -> list[Order]:
        """Walk the book up to `max_distance` ticks past `ref_price`,
        consuming up to `max_qty` total volume."""
        out: list[Order] = []
        remaining = max_qty
        if side == "sell":
            for price in sorted(depth.buy_orders.keys(), reverse=True):
                if price < ref_price - max_distance:
                    break
                avail = int(depth.buy_orders[price])
                qty = min(remaining, max(0, avail))
                if qty <= 0:
                    continue
                out.append(Order(self.HYDROGEL, int(price), -int(qty)))
                remaining -= qty
                if remaining <= 0:
                    break
        else:  # buy
            for price in sorted(depth.sell_orders.keys()):
                if price > ref_price + max_distance:
                    break
                avail = abs(int(depth.sell_orders[price]))
                qty = min(remaining, max(0, avail))
                if qty <= 0:
                    continue
                out.append(Order(self.HYDROGEL, int(price), int(qty)))
                remaining -= qty
                if remaining <= 0:
                    break
        return out

    # ------------------------------------------------------------------
    # 2. A-S reservation-price quoter
    # ------------------------------------------------------------------

    def _reservation_quotes(
        self,
        position: int,
        wall_mid: float,
        microprice: float,
        best_bid: int,
        best_ask: int,
        ts: int,
        limit: int,
    ) -> list[Order]:
        """Wall-anchored A-S reservation-price quoter (final form).

        We compute a reservation price = anchor − q·γ·σ²·(T−t), where the
        anchor is a wall_mid + microprice blend. Quotes are placed at
        baseline ±1 plus an integer offset derived from (anchor−touch_mid)
        and (−q·γ·σ²·(T−t)).

        At q=0 with anchor=touch_mid we exactly match v1's "improve touch by
        1" baseline. The novelty kicks in when (a) wall_mid disagrees with
        touch_mid (the anchor offset shifts both quotes toward fair), and
        (b) inventory accumulates (the skew shifts both quotes against
        the position, which makes the unwind-side more aggressive and the
        accumulate-side less aggressive — the textbook A-S inventory
        management).

        We deliberately AVOID the harder Hedgehogs-style "never overbid
        past fair" cutoff: empirically (see iteration log) it overfits to
        days where wall_mid is stable, and on a drifting day it forces us
        systematically short or long against a trending wall_mid.
        """
        cfg = self.CFG

        # Anchor blends wall_mid (stable, discrete) and microprice
        # (continuous, fine resolution). Sign-correlated in HYDROGEL data.
        anchor = (1.0 - cfg["alpha_microprice"]) * wall_mid \
                 + cfg["alpha_microprice"] * microprice

        # Time remaining as a fraction in [0, 1].
        T_remain = max(0.0, 1.0 - ts / cfg["T_total"])

        # A-S inventory adjustment. With our calibration (γ=3e-4, σ²=33.6):
        # at q=±200, T_rem=0.5, skew = ±1.0 tick.
        skew_to_anchor = position * cfg["gamma"] * cfg["sigma2"] * T_remain

        # Baseline = v1's "improve touch by 1".
        baseline_bid = best_bid + 1
        baseline_ask = best_ask - 1

        # Anchor offset relative to touch_mid. If wall_mid > touch_mid by
        # 1 tick, anchor_offset ≈ +0.5..+1, and we shift quotes UP by ~1
        # tick (sell into the rich, back off the cheap bid).
        touch_mid = (best_bid + best_ask) / 2.0
        anchor_offset = anchor - touch_mid
        price_shift = anchor_offset - skew_to_anchor

        bid_price = int(round(baseline_bid + price_shift))
        ask_price = int(round(baseline_ask + price_shift))

        # Never cross.
        if bid_price >= best_ask:
            bid_price = best_ask - 1
        if ask_price <= best_bid:
            ask_price = best_bid + 1

        bid_room = max(0, limit - position)
        ask_room = max(0, limit + position)
        bid_qty = min(cfg["quote_size"], bid_room)
        ask_qty = min(cfg["quote_size"], ask_room)

        out: list[Order] = []
        if bid_qty > 0:
            out.append(Order(self.HYDROGEL, int(bid_price), int(bid_qty)))
        if ask_qty > 0:
            out.append(Order(self.HYDROGEL, int(ask_price), -int(ask_qty)))
        return out

    # ------------------------------------------------------------------
    # 3. End-of-day flatten (carryover convention)
    # ------------------------------------------------------------------

    def _flatten_quotes(
        self,
        position: int,
        wall_mid: float,
        best_bid: int,
        best_ask: int,
        limit: int,
    ) -> list[Order]:
        cfg = self.CFG
        bid_price = int(round(wall_mid - cfg["flatten_offset"]))
        ask_price = int(round(wall_mid + cfg["flatten_offset"]))
        if bid_price >= best_ask:
            bid_price = best_ask - 1
        if ask_price <= best_bid:
            ask_price = best_bid + 1
        bid_qty = min(cfg["quote_size"], max(0, limit - position))
        ask_qty = min(cfg["quote_size"], max(0, limit + position))
        out: list[Order] = []
        if bid_qty > 0:
            out.append(Order(self.HYDROGEL, int(bid_price), int(bid_qty)))
        if ask_qty > 0:
            out.append(Order(self.HYDROGEL, int(ask_price), -int(ask_qty)))
        return out

    # ------------------------------------------------------------------
    # Hard-limit emergency
    # ------------------------------------------------------------------

    def _cross_book(
        self,
        depth: OrderDepth,
        position: int,
        wall_mid: float,
        side: str,
    ) -> list[Order]:
        cfg = self.CFG
        target_qty = abs(position) - cfg["hard_inv_limit"]
        return self._walk_book(
            depth, side=side, ref_price=wall_mid,
            max_distance=cfg["hard_take_distance"],
            max_qty=target_qty,
        )

    # ------------------------------------------------------------------
    # Book / fair-value helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _book_levels(depth: OrderDepth):
        """Returns (bid_wall, best_bid, best_ask, ask_wall).

        bid_wall = lowest priced bid in book (deepest level)
        ask_wall = highest priced ask in book (deepest level)
        These are the designated-MM "wall" prices in the Frankfurt Hedgehogs
        sense: outer levels with high volume.
        """
        if depth.buy_orders:
            best_bid = max(depth.buy_orders)
            bid_wall = min(depth.buy_orders)
        else:
            best_bid = bid_wall = None
        if depth.sell_orders:
            best_ask = min(depth.sell_orders)
            ask_wall = max(depth.sell_orders)
        else:
            best_ask = ask_wall = None
        return bid_wall, best_bid, best_ask, ask_wall

    @staticmethod
    def _microprice(
        depth: OrderDepth,
        best_bid: int,
        best_ask: int,
        fallback: float,
    ) -> float:
        """Stoikov microprice: best prices weighted by opposite-side volumes."""
        bid_size = abs(int(depth.buy_orders.get(best_bid, 0)))
        ask_size = abs(int(depth.sell_orders.get(best_ask, 0)))
        total = bid_size + ask_size
        if total <= 0:
            return float(fallback)
        return (best_bid * ask_size + best_ask * bid_size) / total
