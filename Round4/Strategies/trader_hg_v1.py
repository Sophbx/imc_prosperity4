"""
HG-only trader, R4 v1.

Strategy: Be Mark 14, with a soft mean-reversion + inventory tilt.

Findings from R4 data (3 days, 360k snapshots):
  - HG mid mean-reverts around 10000, daily std ~32, p5..p95 = -63..+48.
  - Spread = 16 in 92.5% of all snapshots. L1 size ~12 per side.
  - 100% of Mark 14's HG fills are at bid_1 / ask_1 (he IS the inside quote).
  - 100% of Mark 38's HG fills are at the opposite top-of-book (he crosses
    the spread). He pays Mark 14 ~8 / share, ~1300 lots / day.
  - The R3 swing-trader strategy was idle 82.5% of the time and crossed
    the spread on every fill it did do.

Plan:
  - Quote 1 tick INSIDE bid_1 / ask_1 (price priority over Mark 14, since
    the wiki confirms price-time priority and we don't know the FIFO
    ordering at same price).
  - Soft tilt the quote price by inventory and z-score so we lean against
    rich/cheap and against accumulated inventory, instead of binary mode
    flipping.
  - Free-money take: still cross when the inside is grossly mispriced
    relative to fair (rare; sanity backstop).
  - No swing-flip state machine.

Other products are intentionally NOT traded here so we can isolate HG PnL
and compare cleanly to the R3 live HG result of $19,551.
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
    HYDRO_FREE_TAKE_THRESH = 8

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

        fair_eff = self.HYDRO_FAIR + skew
        if ba <= fair_eff - self.HYDRO_FREE_TAKE_THRESH:
            sz = min(-depth.sell_orders[ba], max(0, self.HYDRO_LIMIT - pos))
            if sz > 0:
                orders.append(Order("HYDROGEL_PACK", ba, +sz))
        if bb >= fair_eff + self.HYDRO_FREE_TAKE_THRESH:
            sz = min(depth.buy_orders[bb], max(0, self.HYDRO_LIMIT + pos))
            if sz > 0:
                orders.append(Order("HYDROGEL_PACK", bb, -sz))

        out["HYDROGEL_PACK"] = orders
        return out, 0, ""
