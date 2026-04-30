"""
PEBBLES basket snap arb — v8: regime-switching with Snap Hunter mode.

Path:

    v3c     two-sided MM, baseline=1                  ->  +25,434
    v4      two-sided MM, baseline=5                  ->  +53,988
    v6      v4 with XL=2 (absorber-leg hypothesis)    ->  +55,174
    v7      v6 + drop M, XL=10, snap blackout         ->  +68,068
    v8      v7 + Snap Hunter mode on {XS, XL}         ->  +72,987   ★ +7% vs v7

Conceptual change from v7
    v7 treats snap events as risk to be AVOIDED — when |basket_dev|>10,
    skip every leg. v8 keeps that "avoid the unsafe legs" idea but adds an
    independent SNAP HUNTER STATE that becomes active only during those
    snaps and trades the favorable legs aggressively in the right
    direction.

    Two completely independent state machines:

      NORMAL state   (|basket_dev| ≤ 10, ~97% of ticks)
          - v7 logic verbatim: two-sided passive MM at bid+1/ask-1,
            BASELINE = {XS:5, S:5, M:0, L:5, XL:10}, no signal sizing.

      SNAP HUNTER state   (|basket_dev| > 10, ~3% of ticks)
          - Trade ONLY {XS, XL} — the two legs whose post-fill price drift
            is reliable enough during snaps to overcome execution cost.
            (Leg ablation: XS+XL=72,987; +L=72,845; +S=70,849; XL alone=71,943.)
          - Direction is dictated by basket_dev sign:
              dev > 0 (basket overpriced)  → SELL only on XS, XL
              dev < 0 (basket underpriced) → BUY only on XS, XL
          - Quote price = round(implied_fair) where
              implied_fair = mid - basket_dev = the basket-revert target.
            Margin (offset from implied_fair) is irrelevant — sweeps
            showed margin ∈ {0,1,2,3,5} all yield identical PnL because
            the matching engine fills at OUR price regardless.
          - Quote size = full available room to position cap (POS_LIMIT
            ± current position). Snap is rare; fill at full depth.

Why XS and XL, why not the others
    Per-fill 5-tick drift in NORMAL mode (analysis from v7's diagnostic):
        XS  +0.38   neutral, slightly favorable
        S   −0.93   slightly bleeds
        M   −5.55   ★ bleeds heavily — already dropped from BASELINE in v7
        L   −0.27   neutral
        XL  +6.34   ★ very favorable
    In SNAP mode the dynamics are different (we're on the directional
    side of the basket reversion, not just collecting random spread). The
    leg ablation showed:
        XS+XL ........ 72,987   ★
        XS+L+XL ...... 72,845
        XL only ...... 71,943
        L+XL ......... 71,801
        XS+S+L+XL .... 70,849   (S in snap mode is mildly destructive)
        S+XL ......... 69,947   (S meaningfully bad in snap mode)
        XS+L ......... 68,970   (no XL = barely better than v7)
        v7 (skip all). 68,068

    {XS, XL} is the cleanest cut — the two legs whose snap-mode bias
    points the same direction as basket_dev sign without S or L's noise
    eroding the edge.

Critical implementation detail
    Snap mode skips legs where BASELINE[leg] == 0. Specifically: M is
    NEVER traded, even in snap mode, because normal mode can't unwind
    accumulated M positions (its baseline is 0). An earlier v8 attempt
    that traded M in snap mode underperformed v7 because M positions
    accumulated and bled mark-to-market.

What this still doesn't do
    - Doesn't detect snaps before they happen (no leading indicator).
    - Doesn't model order-book queue priority.
    - Doesn't have post-snap residual-position closeout (relies on normal
      mode's two-sided baseline to organically unwind any snap-mode
      inventory).
"""

from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple


class Trader:
    SUFFIXES = ["XS", "S", "M", "L", "XL"]
    SYMS = [f"PEBBLES_{s}" for s in SUFFIXES]
    POS_LIMIT = 10
    BASKET_FAIR = 50_000.0

    # Normal-mode parameters (= v7)
    BASELINE = {"XS": 5, "S": 5, "M": 0, "L": 5, "XL": 10}
    OFFSET   = {"XS": 1, "S": 1, "M": 1, "L": 1, "XL": 1}

    # Snap Hunter mode parameters
    SNAP_THR = 10                                # |basket_dev| > this => snap mode
    SNAP_LEGS = frozenset({"XS", "XL"})          # only trade these legs in snap mode

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}

        # 1. read all 5 books
        books: Dict[str, Tuple[int, int]] = {}
        mids: Dict[str, float] = {}
        for sym in self.SYMS:
            d = state.order_depths.get(sym)
            if d is None or not d.buy_orders or not d.sell_orders:
                return {}, 0, ""
            bb = max(d.buy_orders); ba = min(d.sell_orders)
            if bb >= ba:
                return {}, 0, ""
            books[sym] = (bb, ba)
            mids[sym] = (bb + ba) / 2.0

        basket_dev = sum(mids.values()) - self.BASKET_FAIR

        # 2. STATE DISPATCH
        if abs(basket_dev) > self.SNAP_THR:
            return self._snap_hunter(state, books, mids, basket_dev)
        return self._normal_mm(state, books)

    # ------------------------------------------------------------------
    # SNAP HUNTER state — directional, one-sided, on whitelisted legs only
    # ------------------------------------------------------------------
    def _snap_hunter(self, state, books, mids, basket_dev):
        result = {}
        for sym, suf in zip(self.SYMS, self.SUFFIXES):
            if suf not in self.SNAP_LEGS:
                continue
            bb, ba = books[sym]
            position = int(state.position.get(sym, 0))
            implied_fair = mids[sym] - basket_dev   # leg's post-revert target

            orders: List[Order] = []
            if basket_dev > 0:
                # Basket overpriced -> legs about to drop -> SELL only.
                ask_price = int(round(implied_fair))
                if ask_price <= bb:
                    ask_price = bb + 1
                sell_room = self.POS_LIMIT + position
                if sell_room > 0 and ask_price < ba:
                    orders.append(Order(sym, ask_price, -sell_room))
            else:
                # Basket underpriced -> legs about to rise -> BUY only.
                bid_price = int(round(implied_fair))
                if bid_price >= ba:
                    bid_price = ba - 1
                buy_room = self.POS_LIMIT - position
                if buy_room > 0 and bid_price > bb:
                    orders.append(Order(sym, bid_price, +buy_room))

            if orders:
                result[sym] = orders
        return result, 0, ""

    # ------------------------------------------------------------------
    # NORMAL state — v7 verbatim, two-sided passive MM
    # ------------------------------------------------------------------
    def _normal_mm(self, state, books):
        result = {}
        for sym, suf in zip(self.SYMS, self.SUFFIXES):
            bb, ba = books[sym]
            position = int(state.position.get(sym, 0))
            baseline = self.BASELINE[suf]
            offset   = self.OFFSET[suf]
            if baseline == 0:
                continue

            buy_room = self.POS_LIMIT - position
            sell_room = self.POS_LIMIT + position
            buy_qty  = min(baseline if buy_room  > 0 else 0, buy_room)
            sell_qty = min(baseline if sell_room > 0 else 0, sell_room)

            orders: List[Order] = []
            if buy_qty > 0:
                bid_price = bb + offset
                if bid_price >= ba:
                    bid_price = ba - 1
                if bid_price > bb:
                    orders.append(Order(sym, bid_price, +buy_qty))
            if sell_qty > 0:
                ask_price = ba - offset
                if ask_price <= bb:
                    ask_price = bb + 1
                if ask_price < ba:
                    orders.append(Order(sym, ask_price, -sell_qty))
            if orders:
                result[sym] = orders
        return result, 0, ""
