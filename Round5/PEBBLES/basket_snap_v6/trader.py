"""
PEBBLES basket snap arb — v6: XL-asymmetric tuning of v4.

Path so far (folders v1/v2/v3a/v3b pruned earlier):

    v3c (baseline=1)        ->  +25,434
    v4  (baseline=5)        ->  +53,988   (+112% vs v3c)
    v6  (XL=2 / others=5)   ->  +55,174   (+2.2% vs v4)

Where v6 comes from
    Round5/Analysis section 4.5 (added by user) showed:

    - XS, S, M, L are mutually independent at the residual level
      (correlations within +/-0.07 after EWMA-detrending).
    - Each of XS/S/M/L is anti-correlated with XL at ~ -0.50.
    - XL is the "absorber leg" - it moves to compensate the others
      to keep the basket sum close to 50,000.
    - At snap events, XL has IQR 44 ticks vs ~20 for the other four,
      and is the largest single-leg mover in 35% of events vs 17%
      for each of the others.

    Implication: XL faces 2x the adverse-selection risk of any other
    leg when we quote passively at bid+1 / ask-1. v4 quotes XL with
    the same BASELINE_SIZE=5 as the rest, which over-exposes us on
    XL fills.

    A small sweep (sweep.py / sweep_results.csv) over
        XL_BASELINE in {0, 1, 2, 3, 5}
        XL_OFFSET   in {1, 2, 3}
    found:
        XL_BASELINE=2, XL_OFFSET=1  ->  55,174  ★
        XL_BASELINE=3, XL_OFFSET=1  ->  55,123
        XL_BASELINE=5, XL_OFFSET=1  ->  53,988  (= v4)
    Wider offsets always hurt (consistent with v3a/v3b sweep).

    The asymmetry helps modestly but consistently: cutting XL baseline
    from 5 to 2 trims adverse selection on the absorber leg without
    sacrificing material spread capture.

What v6 changes vs v4
    BASELINE_SIZE: scalar 5  ->  per-leg dict {XS,S,M,L: 5, XL: 2}
    OFFSET:        scalar 1  ->  per-leg dict (all 1, same as v4)

Everything else is identical to v4.
"""

from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple


class Trader:
    SUFFIXES = ["XS", "S", "M", "L", "XL"]
    SYMS = [f"PEBBLES_{s}" for s in SUFFIXES]
    POS_LIMIT = 10
    BASKET_FAIR = 50_000.0

    K = 1.0                         # irrelevant at high baseline (see v4 sweep)
    MAX_TRADE_SIZE = 10

    # Per-leg config: XL gets a smaller passive baseline because it's the
    # absorber leg with 2x the snap-event noise.
    BASELINE = {"XS": 5, "S": 5, "M": 5, "L": 5, "XL": 2}
    OFFSET   = {"XS": 1, "S": 1, "M": 1, "L": 1, "XL": 1}

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}

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

        for sym, suf in zip(self.SYMS, self.SUFFIXES):
            bb, ba = books[sym]
            position = int(state.position.get(sym, 0))
            baseline = self.BASELINE[suf]
            offset   = self.OFFSET[suf]

            target = int(round(-self.K * basket_dev))
            target = max(-self.POS_LIMIT, min(self.POS_LIMIT, target))

            buy_room = self.POS_LIMIT - position
            sell_room = self.POS_LIMIT + position
            buy_signal_qty  = max(0, target - position)
            sell_signal_qty = max(0, position - target)
            buy_qty  = max(baseline if buy_room  > 0 else 0, buy_signal_qty)
            sell_qty = max(baseline if sell_room > 0 else 0, sell_signal_qty)
            buy_qty  = min(buy_qty,  self.MAX_TRADE_SIZE, buy_room)
            sell_qty = min(sell_qty, self.MAX_TRADE_SIZE, sell_room)

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
