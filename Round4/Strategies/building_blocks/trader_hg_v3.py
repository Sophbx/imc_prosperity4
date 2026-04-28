"""
HG-only trader, R4 v3.

Diff from v1: free-take threshold is now mid-relative, not anchored on 10000.

Why: v1's bug was `bb >= fair_eff + 8` with fair_eff drifting around 10000
even when mid was 10016. The threshold became 10007 while bid was 10008 —
"free take" fired, sold at bid, lost half-spread. Anchor on mid instead so
the take only fires on a genuine dislocation: bid above where our resting
ask would naturally sit, OR ask below where our resting bid would.

Concretely: with HALF_SPREAD=8 and a 2-tick safety buffer, the take fires
only when bb >= mid + 10 (for sells) or ba <= mid - 10 (for buys). In a
normal 16-wide market, bb = mid - 8 and ba = mid + 8 by construction, so
neither condition ever fires — the branch is now a true backstop for
crossed/anomalous books, not a routine spread-bleeder.

This keeps the safety net for genuine outliers without introducing the
asymmetric short bleed we saw in the test run.
"""
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List


class Trader:
    LIMITS = {"HYDROGEL_PACK": 200}

    HYDRO_FAIR = 10000.0
    HYDRO_SIGMA = 32.0
    HYDRO_HALF_SPREAD = 8
    HYDRO_LIMIT = 200
    HYDRO_BASE_SIZE = 30
    HYDRO_INV_TILT = 0.04
    HYDRO_Z_TILT = 2.0
    HYDRO_FREE_TAKE_BUFFER = 2

    def run(self, state: TradingState):
        out: Dict[str, List[Order]] = {}
        depth = state.order_depths.get("HYDROGEL_PACK")
        if depth is None or not depth.buy_orders or not depth.sell_orders:
            return out, 0, ""

        bb = max(depth.buy_orders)
        ba = min(depth.sell_orders)
        if bb >= ba:
            return out, 0, ""

        pos = int(state.position.get("HYDROGEL_PACK", 0))
        mid = (bb + ba) / 2.0

        bid_px = bb + 1
        ask_px = ba - 1

        z = (mid - self.HYDRO_FAIR) / self.HYDRO_SIGMA
        skew = -self.HYDRO_INV_TILT * pos - self.HYDRO_Z_TILT * z
        bid_px = int(round(bid_px + skew))
        ask_px = int(round(ask_px + skew))

        bid_px = min(bid_px, ba - 1)
        ask_px = max(ask_px, bb + 1)
        if bid_px >= ask_px:
            bid_px = ask_px - 1

        bid_size = min(self.HYDRO_BASE_SIZE, max(0, self.HYDRO_LIMIT - pos))
        ask_size = min(self.HYDRO_BASE_SIZE, max(0, self.HYDRO_LIMIT + pos))

        orders: List[Order] = []
        if bid_size > 0:
            orders.append(Order("HYDROGEL_PACK", bid_px, +bid_size))
        if ask_size > 0:
            orders.append(Order("HYDROGEL_PACK", ask_px, -ask_size))

        take_sell_thresh = mid + self.HYDRO_HALF_SPREAD + self.HYDRO_FREE_TAKE_BUFFER
        take_buy_thresh = mid - self.HYDRO_HALF_SPREAD - self.HYDRO_FREE_TAKE_BUFFER
        if ba <= take_buy_thresh:
            sz = min(-depth.sell_orders[ba], max(0, self.HYDRO_LIMIT - pos))
            if sz > 0:
                orders.append(Order("HYDROGEL_PACK", ba, +sz))
        if bb >= take_sell_thresh:
            sz = min(depth.buy_orders[bb], max(0, self.HYDRO_LIMIT + pos))
            if sz > 0:
                orders.append(Order("HYDROGEL_PACK", bb, -sz))

        out["HYDROGEL_PACK"] = orders
        return out, 0, ""
