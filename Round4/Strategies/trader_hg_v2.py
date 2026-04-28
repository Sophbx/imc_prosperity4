"""
HG-only trader, R4 v2.

Diff from v1: free-take branch removed entirely.

Why: v1's free-take condition was `bb >= fair_eff + 8` with fair_eff anchored
on 10000. Whenever mid drifts above 10000 (which is ~50% of the time), the
condition fired and made us sell AT THE BID — i.e. cross the spread to sell
8 below mid, the opposite of what we want. Diagnostic from the 521391
test run showed many sells filling at bid_1 with mid 8 above, bleeding
~$8/share each. Net: ended day -84 short with $1,583 PnL on a 1000-iter
test run, vs the backtest predicting ~$1,950.

The core MM (passive bid_1+1 / ask_1-1 with z and inventory tilt) doesn't
need a free-take branch in this market; the inside is stable Mark-14 quotes
and there's no genuine dislocation to capture. Drop it.
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

        out["HYDROGEL_PACK"] = orders
        return out, 0, ""
