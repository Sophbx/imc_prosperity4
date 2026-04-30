"""
v12 = v10 multi-cat MM + teammate's directional/relative-value logic on the
5 products where it materially beats MM:

    MICROCHIP_CIRCLE     teammate +30,638  vs  v10 +10,381  (+20,257)
    MICROCHIP_TRIANGLE   teammate +14,606  vs  v10 +12,214  (+2,392)
    ROBOT_LAUNDRY        teammate +17,256  vs  v10  +6,187  (+11,069)
    SNACKPACK_RASPBERRY  teammate +21,282  vs  v10 +15,397  (+5,885)
    SNACKPACK_VANILLA    teammate  +8,489  vs  v10  +2,772  (+5,717)
                                                                       --------
                                                  Predicted gross gain  +45,320

Teammate's strategy was tested standalone — it also trades ROBOT_DISHES and
ROBOT_MOPPING with directional momentum and LOSES on both (-12,834 and
-20,595). v10 already drops these (BASELINE=0) — we keep that drop.

Teammate's reference-value strategies for VAN and RASP use SNACKPACK_CHOCOLATE
and SNACKPACK_PISTACHIO as reference mids without trading them. v10's MM on
CHOC and PIS coexists fine (different products).

Mechanism summary on the 5 swapped products:

    MICROCHIP_CIRCLE       hold-momentum (no flatten):
        signal = SMA(100) - SMA(200), entry edge 30 ticks
        target = ±POS_LIMIT on |signal| > entry_edge, else hold
    MICROCHIP_TRIANGLE     mean-reversion with flatten:
        fair = SMA(1000), entry edge 100 (vs touch), exit edge 30 (vs mid)
    ROBOT_LAUNDRY          relative-value vs ROBOT_DISHES:
        fair = 19841 - mid(DISHES), entry edge 250, exit edge 60
    SNACKPACK_RASPBERRY    relative-value vs SNACKPACK_PISTACHIO:
        fair = 19574 - mid(PIST), entry edge 80, exit edge 20
    SNACKPACK_VANILLA      relative-value vs SNACKPACK_CHOCOLATE:
        fair = 19941 - mid(CHOC), entry edge 50, exit edge 12

All 5 use aggressive crossing (lift ask / hit bid) with full POS_LIMIT
sizing — directional bets, not market making.
"""

from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple
from collections import defaultdict, deque


class Trader:
    POS_LIMIT = 10
    PEBBLES_FAIR = 50_000.0
    PEBBLES_LEGS = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
    PEBBLES_SNAP_THR = 10
    PEBBLES_SNAP_LEGS = frozenset({"PEBBLES_XS", "PEBBLES_XL"})

    # Products handed to teammate's directional logic (not MM'd)
    DIRECTIONAL_PRODUCTS = frozenset({
        "MICROCHIP_CIRCLE",
        "MICROCHIP_TRIANGLE",
        "ROBOT_LAUNDRY",
        "SNACKPACK_RASPBERRY",
        "SNACKPACK_VANILLA",
    })

    # Teammate strategy params (verbatim from their trader_round5_mc_rb_sp.py)
    HOLD_MOMENTUM = {
        "MICROCHIP_CIRCLE": (100, 200, 30),  # short_w, long_w, entry_edge
    }
    MEAN_REVERSION = {
        "MICROCHIP_TRIANGLE": (1000, 100, 30),  # window, entry, exit
    }
    ROBOT_LAUNDRY_REF_SUM = 19841
    ROBOT_LAUNDRY_EDGE    = 250
    ROBOT_LAUNDRY_EXIT    = 60
    VANILLA_REF_SUM   = 19941
    VANILLA_EDGE      = 50
    VANILLA_EXIT      = 12
    RASPBERRY_REF_SUM = 19574
    RASPBERRY_EDGE    = 80
    RASPBERRY_EXIT    = 20

    MAX_HISTORY = 1200

    # v10 BASELINE map — ONLY the 5 swapped products set to 0; everything else unchanged
    BASELINE = {
        "PEBBLES_XS": 5, "PEBBLES_S": 5, "PEBBLES_M": 0, "PEBBLES_L": 5, "PEBBLES_XL": 10,
        "GALAXY_SOUNDS_DARK_MATTER": 5, "GALAXY_SOUNDS_BLACK_HOLES": 5,
        "GALAXY_SOUNDS_PLANETARY_RINGS": 5, "GALAXY_SOUNDS_SOLAR_WINDS": 5,
        "GALAXY_SOUNDS_SOLAR_FLAMES": 0,
        "SLEEP_POD_SUEDE": 5, "SLEEP_POD_LAMB_WOOL": 0, "SLEEP_POD_POLYESTER": 5,
        "SLEEP_POD_NYLON": 5, "SLEEP_POD_COTTON": 5,
        "MICROCHIP_CIRCLE":   0,    # ← v10 had 5 — handed to directional layer
        "MICROCHIP_OVAL":     5,
        "MICROCHIP_SQUARE":   10,
        "MICROCHIP_RECTANGLE":0,
        "MICROCHIP_TRIANGLE": 0,    # ← v10 had 5 — handed to directional layer
        "ROBOT_VACUUMING":    0, "ROBOT_MOPPING": 0, "ROBOT_DISHES": 0,
        "ROBOT_LAUNDRY":      0,    # ← v10 had 5 — handed to directional layer
        "ROBOT_IRONING":      5,
        "UV_VISOR_YELLOW": 0, "UV_VISOR_AMBER": 5, "UV_VISOR_ORANGE": 5,
        "UV_VISOR_RED": 0, "UV_VISOR_MAGENTA": 0,
        "TRANSLATOR_SPACE_GRAY": 0, "TRANSLATOR_ASTRO_BLACK": 5,
        "TRANSLATOR_ECLIPSE_CHARCOAL": 5, "TRANSLATOR_GRAPHITE_MIST": 0,
        "TRANSLATOR_VOID_BLUE": 0,
        "PANEL_1X2": 0, "PANEL_2X2": 5, "PANEL_1X4": 5, "PANEL_2X4": 5, "PANEL_4X4": 0,
        "OXYGEN_SHAKE_MORNING_BREATH": 5, "OXYGEN_SHAKE_EVENING_BREATH": 5,
        "OXYGEN_SHAKE_MINT": 0, "OXYGEN_SHAKE_CHOCOLATE": 5, "OXYGEN_SHAKE_GARLIC": 5,
        "SNACKPACK_CHOCOLATE":  5,  # KEPT (used as reference + still profitable to MM)
        "SNACKPACK_VANILLA":    0,  # ← v10 had 5 — handed to directional layer
        "SNACKPACK_PISTACHIO":  5,  # KEPT (reference + profitable MM)
        "SNACKPACK_STRAWBERRY": 5,
        "SNACKPACK_RASPBERRY":  0,  # ← v10 had 5 — handed to directional layer
    }
    OFFSET = 1

    def __init__(self):
        self.mid_history = defaultdict(lambda: deque(maxlen=self.MAX_HISTORY))

    # ========== utilities ==========
    def _best_bid_ask(self, depth):
        if not depth.buy_orders or not depth.sell_orders:
            return None
        bb = max(depth.buy_orders); ba = min(depth.sell_orders)
        return bb, depth.buy_orders[bb], ba, -depth.sell_orders[ba]

    def _rolling_mean(self, product, window):
        h = self.mid_history[product]
        if len(h) < window: return None
        return sum(list(h)[-window:]) / window

    def _add_buy(self, orders, vp, sym, price, qty):
        cap = self.POS_LIMIT - vp.get(sym, 0)
        q = min(qty, cap)
        if q > 0:
            orders.setdefault(sym, []).append(Order(sym, price, +q))
            vp[sym] = vp.get(sym, 0) + q

    def _add_sell(self, orders, vp, sym, price, qty):
        cap = self.POS_LIMIT + vp.get(sym, 0)
        q = min(qty, cap)
        if q > 0:
            orders.setdefault(sym, []).append(Order(sym, price, -q))
            vp[sym] = vp.get(sym, 0) - q

    def _rebalance(self, orders, vp, sym, target, bb, bv, ba, av):
        delta = target - vp.get(sym, 0)
        if delta > 0:
            self._add_buy(orders, vp, sym, ba, min(delta, av))
        elif delta < 0:
            self._add_sell(orders, vp, sym, bb, min(-delta, bv))

    # ========== Teammate's directional strategies (verbatim) ==========
    def _trade_hold_momentum(self, orders, vp, quote):
        for sym, (sw, lw, edge) in self.HOLD_MOMENTUM.items():
            if sym not in quote: continue
            sm = self._rolling_mean(sym, sw)
            lm = self._rolling_mean(sym, lw)
            if sm is None or lm is None: continue
            sig = sm - lm
            bb, bv, ba, av = quote[sym]
            cur = vp.get(sym, 0)
            if sig > edge: tgt = self.POS_LIMIT
            elif sig < -edge: tgt = -self.POS_LIMIT
            else: tgt = cur
            self._rebalance(orders, vp, sym, tgt, bb, bv, ba, av)

    def _trade_mean_reversion(self, orders, vp, quote, mids):
        for sym, (window, ent, exi) in self.MEAN_REVERSION.items():
            if sym not in quote or sym not in mids: continue
            fair = self._rolling_mean(sym, window)
            if fair is None: continue
            bb, bv, ba, av = quote[sym]
            mid = mids[sym]
            cur = vp.get(sym, 0)
            if ba < fair - ent: tgt = self.POS_LIMIT
            elif bb > fair + ent: tgt = -self.POS_LIMIT
            elif abs(mid - fair) < exi: tgt = 0
            else: tgt = cur
            self._rebalance(orders, vp, sym, tgt, bb, bv, ba, av)

    def _trade_reference_value(self, orders, vp, quote, mids,
                               sym, ref_sym, ref_sum, ent, exi):
        if sym not in quote or ref_sym not in mids or sym not in mids: return
        fair = ref_sum - mids[ref_sym]
        bb, bv, ba, av = quote[sym]
        mid = mids[sym]
        cur = vp.get(sym, 0)
        if ba < fair - ent: tgt = self.POS_LIMIT
        elif bb > fair + ent: tgt = -self.POS_LIMIT
        elif abs(mid - fair) < exi: tgt = 0
        else: tgt = cur
        self._rebalance(orders, vp, sym, tgt, bb, bv, ba, av)

    # ========== v10's MM ==========
    def _mm_one(self, state, sym, book, result):
        bb, ba = book
        pos = int(state.position.get(sym, 0))
        baseline = self.BASELINE.get(sym, 0)
        if baseline == 0: return
        bq = min(baseline if self.POS_LIMIT - pos > 0 else 0, self.POS_LIMIT - pos)
        sq = min(baseline if self.POS_LIMIT + pos > 0 else 0, self.POS_LIMIT + pos)
        orders = []
        if bq > 0:
            p = bb + self.OFFSET
            if p >= ba: p = ba - 1
            if p > bb: orders.append(Order(sym, p, +bq))
        if sq > 0:
            p = ba - self.OFFSET
            if p <= bb: p = bb + 1
            if p < ba: orders.append(Order(sym, p, -sq))
        if orders:
            result[sym] = orders

    # ========== main ==========
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        vp = dict(state.position)  # virtual position tracker for directional layer

        # --- Build quote/mid maps for everything ---
        quote = {}
        mids = {}
        for sym, depth in state.order_depths.items():
            best = self._best_bid_ask(depth)
            if best is None: continue
            bb, bv, ba, av = best
            if bb >= ba: continue
            quote[sym] = (bb, bv, ba, av)
            mids[sym] = (bb + ba) / 2.0

        # --- v10 PEBBLES Snap Hunter (unchanged) ---
        peb_books = {s: (quote[s][0], quote[s][2]) for s in self.PEBBLES_LEGS if s in quote}
        peb_mids  = {s: mids[s] for s in self.PEBBLES_LEGS if s in mids}
        in_pebbles_snap = False
        if len(peb_mids) == 5:
            basket_dev = sum(peb_mids.values()) - self.PEBBLES_FAIR
            if abs(basket_dev) > self.PEBBLES_SNAP_THR:
                in_pebbles_snap = True
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

        # --- Teammate's directional layer on the 5 swapped products ---
        self._trade_hold_momentum(result, vp, quote)         # CIRCLE
        self._trade_mean_reversion(result, vp, quote, mids)  # TRIANGLE
        self._trade_reference_value(result, vp, quote, mids,
            sym="ROBOT_LAUNDRY", ref_sym="ROBOT_DISHES",
            ref_sum=self.ROBOT_LAUNDRY_REF_SUM,
            ent=self.ROBOT_LAUNDRY_EDGE, exi=self.ROBOT_LAUNDRY_EXIT)
        self._trade_reference_value(result, vp, quote, mids,
            sym="SNACKPACK_VANILLA", ref_sym="SNACKPACK_CHOCOLATE",
            ref_sum=self.VANILLA_REF_SUM,
            ent=self.VANILLA_EDGE, exi=self.VANILLA_EXIT)
        self._trade_reference_value(result, vp, quote, mids,
            sym="SNACKPACK_RASPBERRY", ref_sym="SNACKPACK_PISTACHIO",
            ref_sum=self.RASPBERRY_REF_SUM,
            ent=self.RASPBERRY_EDGE, exi=self.RASPBERRY_EXIT)

        # --- v10 MM on everything not in DIRECTIONAL_PRODUCTS ---
        if len(peb_mids) == 5 and not in_pebbles_snap:
            for sym in self.PEBBLES_LEGS:
                if sym in quote:
                    self._mm_one(state, sym, (quote[sym][0], quote[sym][2]), result)

        for sym, baseline in self.BASELINE.items():
            if sym.startswith("PEBBLES_"): continue
            if sym in self.DIRECTIONAL_PRODUCTS: continue
            if baseline == 0: continue
            if sym not in quote: continue
            self._mm_one(state, sym, (quote[sym][0], quote[sym][2]), result)

        # --- Update mid history for directional layer ---
        for sym in self.DIRECTIONAL_PRODUCTS | {"ROBOT_DISHES", "SNACKPACK_CHOCOLATE", "SNACKPACK_PISTACHIO"}:
            if sym in mids:
                self.mid_history[sym].append(mids[sym])

        return result, 0, ""
