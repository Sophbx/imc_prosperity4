"""
v9 — multi-category two-sided MM, per-leg baseline driven by drift analysis.

Idea
    The signal that drove v7→v8 (per-fill 5-tick drift) was applied to
    PEBBLES. But the same diagnostic on ALL 50 products reveals 7 "XL-like"
    favorable legs we've never traded, scattered across GALAXY_SOUNDS,
    SLEEP_POD, MICROCHIP, ROBOT, PANEL — and 9 "M-like" bleeders that we'd
    want to drop. v9 activates two-sided MM on every leg with non-bleeding
    drift, sized by the strength of the favorable bias.

Drift-based BASELINE rule
    drift > +2       → BASELINE = 10  (max out)
    drift in [0, +2] → BASELINE = 5   (standard MM)
    drift in [-1, 0] → BASELINE = 5   (spread income covers small drift)
    drift < -1       → BASELINE = 0   (drop entirely — bleeder)

PEBBLES retains v8 Snap Hunter mode (regime-switching on basket dev).
Other categories don't have a basket constraint — pure MM only.

Snap Hunter only triggers on PEBBLES, gated by PEBBLES basket sum.
"""

from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple


class Trader:
    POS_LIMIT = 10
    PEBBLES_FAIR = 50_000.0
    PEBBLES_LEGS = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
    PEBBLES_SNAP_THR = 10
    PEBBLES_SNAP_LEGS = frozenset({"PEBBLES_XS", "PEBBLES_XL"})

    # Per-leg BASELINE derived from per-fill 5-tick drift on R5 data.
    # Drift > +2: BASELINE=10. Drift in [-1, +2]: BASELINE=5. Drift < -1: BASELINE=0.
    BASELINE = {
        # PEBBLES (kept = v8)
        "PEBBLES_XS":              5,
        "PEBBLES_S":               5,
        "PEBBLES_M":               0,   # bleeder (drift -3.99)
        "PEBBLES_L":               5,
        "PEBBLES_XL":             10,   # favorable (drift +4.08)

        # GALAXY_SOUNDS — all 5 ok, BLACK_HOLES & SOLAR_WINDS slightly favorable
        "GALAXY_SOUNDS_DARK_MATTER":      5,
        "GALAXY_SOUNDS_BLACK_HOLES":      5,
        "GALAXY_SOUNDS_PLANETARY_RINGS":  5,
        "GALAXY_SOUNDS_SOLAR_WINDS":      5,
        "GALAXY_SOUNDS_SOLAR_FLAMES":     5,

        # SLEEP_POD — all 5 ok, COTTON slightly favorable
        "SLEEP_POD_SUEDE":         5,
        "SLEEP_POD_LAMB_WOOL":     5,
        "SLEEP_POD_POLYESTER":     5,
        "SLEEP_POD_NYLON":         5,
        "SLEEP_POD_COTTON":        5,

        # MICROCHIP — SQUARE max, RECTANGLE drop
        "MICROCHIP_CIRCLE":        5,
        "MICROCHIP_OVAL":          5,
        "MICROCHIP_SQUARE":       10,   # favorable (drift +3.37)
        "MICROCHIP_RECTANGLE":     0,   # bleeder (drift -1.23)
        "MICROCHIP_TRIANGLE":      5,

        # ROBOT — LAUNDRY favorable, MOPPING & DISHES drop
        "ROBOT_VACUUMING":         5,
        "ROBOT_MOPPING":           0,   # bleeder (drift -1.52)
        "ROBOT_DISHES":            0,   # bleeder (drift -1.40)
        "ROBOT_LAUNDRY":           5,
        "ROBOT_IRONING":           5,

        # UV_VISOR — three bleeders, only ORANGE & AMBER survive
        "UV_VISOR_YELLOW":         0,   # bleeder (drift -1.74)
        "UV_VISOR_AMBER":          5,
        "UV_VISOR_ORANGE":         5,
        "UV_VISOR_RED":            0,   # bleeder (drift -1.92)
        "UV_VISOR_MAGENTA":        0,   # bleeder (drift -1.39)

        # TRANSLATOR — VOID_BLUE drop
        "TRANSLATOR_SPACE_GRAY":   5,
        "TRANSLATOR_ASTRO_BLACK":  5,
        "TRANSLATOR_ECLIPSE_CHARCOAL": 5,
        "TRANSLATOR_GRAPHITE_MIST":    5,
        "TRANSLATOR_VOID_BLUE":    0,   # bleeder (drift -1.14)

        # PANEL — 1X2 drop, 4X4 favorable
        "PANEL_1X2":               0,   # bleeder (drift -1.40)
        "PANEL_2X2":               5,
        "PANEL_1X4":               5,
        "PANEL_2X4":               5,
        "PANEL_4X4":               5,

        # OXYGEN_SHAKE — all neutral, all keep at 5
        "OXYGEN_SHAKE_MORNING_BREATH": 5,
        "OXYGEN_SHAKE_EVENING_BREATH": 5,
        "OXYGEN_SHAKE_MINT":            5,
        "OXYGEN_SHAKE_CHOCOLATE":       5,
        "OXYGEN_SHAKE_GARLIC":          5,

        # SNACKPACK — all neutral, all keep at 5 (cointegration explored separately)
        "SNACKPACK_CHOCOLATE":     5,
        "SNACKPACK_VANILLA":       5,
        "SNACKPACK_PISTACHIO":     5,
        "SNACKPACK_STRAWBERRY":    5,
        "SNACKPACK_RASPBERRY":     5,
    }

    OFFSET = 1  # uniform across all legs (proven optimal in v3a/v3b/v6 sweeps)

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}

        # --- 1. PEBBLES regime-switching block (v8 logic verbatim) ---
        peb_books, peb_mids = {}, {}
        peb_ok = True
        for sym in self.PEBBLES_LEGS:
            d = state.order_depths.get(sym)
            if d is None or not d.buy_orders or not d.sell_orders:
                peb_ok = False; break
            bb = max(d.buy_orders); ba = min(d.sell_orders)
            if bb >= ba:
                peb_ok = False; break
            peb_books[sym] = (bb, ba)
            peb_mids[sym] = (bb + ba) / 2.0

        if peb_ok:
            basket_dev = sum(peb_mids.values()) - self.PEBBLES_FAIR
            if abs(basket_dev) > self.PEBBLES_SNAP_THR:
                # Snap Hunter mode on PEBBLES
                for sym in self.PEBBLES_LEGS:
                    if sym not in self.PEBBLES_SNAP_LEGS:
                        continue
                    bb, ba = peb_books[sym]
                    pos = int(state.position.get(sym, 0))
                    implied_fair = peb_mids[sym] - basket_dev
                    orders = []
                    if basket_dev > 0:
                        ask_price = int(round(implied_fair))
                        if ask_price <= bb: ask_price = bb + 1
                        sell_room = self.POS_LIMIT + pos
                        if sell_room > 0 and ask_price < ba:
                            orders.append(Order(sym, ask_price, -sell_room))
                    else:
                        bid_price = int(round(implied_fair))
                        if bid_price >= ba: bid_price = ba - 1
                        buy_room = self.POS_LIMIT - pos
                        if buy_room > 0 and bid_price > bb:
                            orders.append(Order(sym, bid_price, +buy_room))
                    if orders: result[sym] = orders
                # PEBBLES in snap mode → other PEBBLES legs do nothing this tick.
                # Other categories continue normally below.
            else:
                # Normal MM on all PEBBLES legs
                for sym in self.PEBBLES_LEGS:
                    self._mm_one(state, sym, peb_books[sym], result)

        # --- 2. All non-PEBBLES legs: standard two-sided MM ---
        for sym, baseline in self.BASELINE.items():
            if sym.startswith("PEBBLES_"):
                continue   # handled above
            if baseline == 0:
                continue
            d = state.order_depths.get(sym)
            if d is None or not d.buy_orders or not d.sell_orders:
                continue
            bb = max(d.buy_orders); ba = min(d.sell_orders)
            if bb >= ba:
                continue
            self._mm_one(state, sym, (bb, ba), result)

        return result, 0, ""

    def _mm_one(self, state, sym, book, result):
        bb, ba = book
        pos = int(state.position.get(sym, 0))
        baseline = self.BASELINE.get(sym, 0)
        if baseline == 0:
            return
        buy_room = self.POS_LIMIT - pos
        sell_room = self.POS_LIMIT + pos
        buy_qty = min(baseline if buy_room > 0 else 0, buy_room)
        sell_qty = min(baseline if sell_room > 0 else 0, sell_room)
        orders = []
        if buy_qty > 0:
            bid_price = bb + self.OFFSET
            if bid_price >= ba: bid_price = ba - 1
            if bid_price > bb:
                orders.append(Order(sym, bid_price, +buy_qty))
        if sell_qty > 0:
            ask_price = ba - self.OFFSET
            if ask_price <= bb: ask_price = bb + 1
            if ask_price < ba:
                orders.append(Order(sym, ask_price, -sell_qty))
        if orders:
            result[sym] = orders
