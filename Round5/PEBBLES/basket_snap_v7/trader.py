"""
PEBBLES basket snap arb — v7: drop the bleeder + snap blackout.

Path:

    v3c (b=1)         ->  +25,434
    v4  (b=5)         ->  +53,988   (+112% vs v3c)
    v6  (XL=2)        ->  +55,174   (+2% vs v4)
    v7  (this)        ->  +68,068   (+23% vs v6, +27% vs v4) ★

How v7 was found
    Stepped back, treated the trader as an unknown. Three diagnostics
    against the raw data forced the design:

    1. Bot trade flow ALL happens at quiet ticks (|basket_dev| < 3).
       Median |dev| at fill = 0.5; only 1.86% of fills are at snap events.
       So PEBBLES profit = bot's spread payment - our adverse-selection cost.
       The basket-snap signal is structurally untradeable in this matching
       engine because the spread (5-8 ticks/leg) > snap deviation (max 18).

    2. v6 already captures 100% of bot trades (3,220/3,220). The whole
       "be more aggressive on entry / catch more flow" lever is exhausted.
       The remaining lever is reducing adverse selection.

    3. Per-fill drift analysis (mid 5 ticks ahead, signed by side):
            XS:  +0.38   (neutral)
            S:   -0.93   (slight bleed, but trades are critical to the mix)
            M:   -5.55   ← BIG bleeder
            L:   -0.27   (neutral)
            XL:  +6.34   ← big winner

       v6 sized XL DOWN (b=2) because of the absorber-leg snap-noise
       hypothesis. But that snap noise only applies to 1.86% of fills.
       At the other 98.14% of fills, XL is favorably selected and v6 was
       throttling its best leg.

What v7 changes
    a) Skip ALL quotes when |basket_dev| > 10 — avoid the small but
       expensive adverse-selection slice at snap events.
    b) BASELINE per leg is now data-driven from the per-fill drift table:
            XS  -> 5      (neutral, keep at default)
            S   -> 5      (slight bleed, but dropping it costs ~47k in
                          basket structure — keep)
            M   -> 0      ★ drop entirely; the worst bleeder
            L   -> 5      (neutral, keep)
            XL  -> 10     ★ max it out; the most favorable leg post-blackout
    c) Removed K (signal-driven inventory) — confirmed irrelevant at any
       baseline; pure two-sided MM.

Sweep evidence (Kevin BT, all numbers reproducible):
    M=0, others=5, XL=10                    -> 68,068  ★
    M=0, S=0, others=5, XL=10               -> 21,431  (S critical)
    M=0, L=0, others=5, XL=10               -> 62,693
    ONLY XL=10                              -> 19,520  (need diversification)
    XS=10,S=10,M=0,L=10,XL=10               -> 68,068  (saturates ≥5)
    M=0, XL=10, others=7                    -> 68,068  (saturates ≥5)

Why it works
    PEBBLES PnL = sum over legs of [bot_spread_paid - adverse_selection_loss].
    M's spread income (~+7k) was almost exactly cancelled by adverse
    selection (~−7k); zero net contribution but lots of risk.
    Dropping M removes both, and the basket-constraint structure transfers
    M's portion of the bot flow onto XS/S/L/XL which still capture spread.
    XL's drift advantage on quiet-tick fills accumulates with the larger
    baseline. Blackout shuts off the snap-tick adverse-selection events
    that were costing ~6k/3 days.
"""

from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple


class Trader:
    SUFFIXES = ["XS", "S", "M", "L", "XL"]
    SYMS = [f"PEBBLES_{s}" for s in SUFFIXES]
    POS_LIMIT = 10
    BASKET_FAIR = 50_000.0

    BASELINE = {"XS": 5, "S": 5, "M": 0, "L": 5, "XL": 10}
    OFFSET   = {"XS": 1, "S": 1, "M": 1, "L": 1, "XL": 1}
    BLACKOUT_THR = 10  # skip all quotes when |basket_dev| > this

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

        # Snap blackout — skip everything during big basket deviations
        if abs(basket_dev) > self.BLACKOUT_THR:
            return {}, 0, ""

        for sym, suf in zip(self.SYMS, self.SUFFIXES):
            bb, ba = books[sym]
            position = int(state.position.get(sym, 0))
            baseline = self.BASELINE[suf]
            offset   = self.OFFSET[suf]
            if baseline == 0:
                continue  # leg fully disabled (M)

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
