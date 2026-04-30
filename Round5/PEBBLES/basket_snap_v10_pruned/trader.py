"""
v10 — v9 with the empirically-confirmed bleeders dropped.

v9 used per-fill 5-tick drift to predict which legs to disable. Backtesting
revealed several legs with positive drift that nonetheless lost money in
actual MM (drift is a noisy proxy — it captures direction but not the full
spread/adverse-selection economics).

v10 drops the seven non-PEBBLES legs whose v9 actual-PnL was negative:

    SLEEP_POD_LAMB_WOOL          -23,920
    GALAXY_SOUNDS_SOLAR_FLAMES    -8,237
    TRANSLATOR_SPACE_GRAY         -5,722
    PANEL_4X4                     -4,729
    ROBOT_VACUUMING               -3,061
    TRANSLATOR_GRAPHITE_MIST      -1,349
    OXYGEN_SHAKE_MINT               -558

Total drag from these in v9: ~47,576 ticks. v10 expects to recover most of
that (could be slightly less because some of these legs might have positive
spread income offset by adverse selection — dropping loses both).

PEBBLES legs are NOT touched (v8 logic preserved exactly).
"""

from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple


class Trader:
    POS_LIMIT = 10
    PEBBLES_FAIR = 50_000.0
    PEBBLES_LEGS = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
    PEBBLES_SNAP_THR = 10
    PEBBLES_SNAP_LEGS = frozenset({"PEBBLES_XS", "PEBBLES_XL"})

    BASELINE = {
        # PEBBLES (= v8)
        "PEBBLES_XS":              5,
        "PEBBLES_S":               5,
        "PEBBLES_M":               0,
        "PEBBLES_L":               5,
        "PEBBLES_XL":             10,

        # GALAXY_SOUNDS — drop SOLAR_FLAMES (loser in v9 BT)
        "GALAXY_SOUNDS_DARK_MATTER":      5,
        "GALAXY_SOUNDS_BLACK_HOLES":      5,
        "GALAXY_SOUNDS_PLANETARY_RINGS":  5,
        "GALAXY_SOUNDS_SOLAR_WINDS":      5,
        "GALAXY_SOUNDS_SOLAR_FLAMES":     0,   # ← v9 BT bleeder

        # SLEEP_POD — drop LAMB_WOOL (biggest single bleeder)
        "SLEEP_POD_SUEDE":         5,
        "SLEEP_POD_LAMB_WOOL":     0,   # ← v9 BT bleeder
        "SLEEP_POD_POLYESTER":     5,
        "SLEEP_POD_NYLON":         5,
        "SLEEP_POD_COTTON":        5,

        # MICROCHIP
        "MICROCHIP_CIRCLE":        5,
        "MICROCHIP_OVAL":          5,
        "MICROCHIP_SQUARE":       10,
        "MICROCHIP_RECTANGLE":     0,
        "MICROCHIP_TRIANGLE":      5,

        # ROBOT — drop VACUUMING (loser in v9 BT)
        "ROBOT_VACUUMING":         0,   # ← v9 BT bleeder
        "ROBOT_MOPPING":           0,
        "ROBOT_DISHES":            0,
        "ROBOT_LAUNDRY":           5,
        "ROBOT_IRONING":           5,

        # UV_VISOR
        "UV_VISOR_YELLOW":         0,
        "UV_VISOR_AMBER":          5,
        "UV_VISOR_ORANGE":         5,
        "UV_VISOR_RED":            0,
        "UV_VISOR_MAGENTA":        0,

        # TRANSLATOR — drop SPACE_GRAY and GRAPHITE_MIST (both v9 BT bleeders)
        "TRANSLATOR_SPACE_GRAY":   0,   # ← v9 BT bleeder
        "TRANSLATOR_ASTRO_BLACK":  5,
        "TRANSLATOR_ECLIPSE_CHARCOAL": 5,
        "TRANSLATOR_GRAPHITE_MIST":    0,   # ← v9 BT bleeder
        "TRANSLATOR_VOID_BLUE":    0,

        # PANEL — drop 4X4 (counterintuitive — drift +1.22 but BT loser)
        "PANEL_1X2":               0,
        "PANEL_2X2":               5,
        "PANEL_1X4":               5,
        "PANEL_2X4":               5,
        "PANEL_4X4":               0,   # ← v9 BT bleeder despite drift +1.22

        # OXYGEN_SHAKE — drop MINT (small but real bleeder)
        "OXYGEN_SHAKE_MORNING_BREATH": 5,
        "OXYGEN_SHAKE_EVENING_BREATH": 5,
        "OXYGEN_SHAKE_MINT":            0,   # ← v9 BT bleeder
        "OXYGEN_SHAKE_CHOCOLATE":       5,
        "OXYGEN_SHAKE_GARLIC":          5,

        # SNACKPACK
        "SNACKPACK_CHOCOLATE":     5,
        "SNACKPACK_VANILLA":       5,
        "SNACKPACK_PISTACHIO":     5,
        "SNACKPACK_STRAWBERRY":    5,
        "SNACKPACK_RASPBERRY":     5,
    }

    OFFSET = 1

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}

        # PEBBLES regime-switching block (v8)
        peb_books, peb_mids = {}, {}
        peb_ok = True
        for sym in self.PEBBLES_LEGS:
            d = state.order_depths.get(sym)
            if d is None or not d.buy_orders or not d.sell_orders:
                peb_ok = False; break
            bb = max(d.buy_orders); ba = min(d.sell_orders)
            if bb >= ba: peb_ok = False; break
            peb_books[sym] = (bb, ba)
            peb_mids[sym] = (bb + ba) / 2.0

        if peb_ok:
            basket_dev = sum(peb_mids.values()) - self.PEBBLES_FAIR
            if abs(basket_dev) > self.PEBBLES_SNAP_THR:
                for sym in self.PEBBLES_LEGS:
                    if sym not in self.PEBBLES_SNAP_LEGS: continue
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
            else:
                for sym in self.PEBBLES_LEGS:
                    self._mm_one(state, sym, peb_books[sym], result)

        # Non-PEBBLES MM
        for sym, baseline in self.BASELINE.items():
            if sym.startswith("PEBBLES_"): continue
            if baseline == 0: continue
            d = state.order_depths.get(sym)
            if d is None or not d.buy_orders or not d.sell_orders: continue
            bb = max(d.buy_orders); ba = min(d.sell_orders)
            if bb >= ba: continue
            self._mm_one(state, sym, (bb, ba), result)

        return result, 0, ""

    def _mm_one(self, state, sym, book, result):
        bb, ba = book
        pos = int(state.position.get(sym, 0))
        baseline = self.BASELINE.get(sym, 0)
        if baseline == 0: return
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
