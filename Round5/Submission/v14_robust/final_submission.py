"""
v14_robust = v12 + 3 ROBUST additions (subset of v14_research's 7 alphas).

The agent's v14_research found 7 alphas (+261,850 in BT vs v12). Tightness audit
on (mid_A + mid_B) per-day means revealed that 4 of the 7 pairs DRIFT across
days more than 1× std — they're not true cointegrations, just data-mined edges
that happened to profit on D+2/D+3/D+4. Those are dropped for OOS robustness:

    Pair                              std    D2-D4 drift   ratio   verdict
    --------------------------------------------------------------------------
    KEEP:
      ROBOT_VACUUMING+DISHES          433       197       0.5σ    ROBUST
      MINT+EVE_BREATH                 542       687       1.3σ    SAFE (entry=700 absorbs)
      TRANSLATOR_VOID_BLUE momentum    -         -         -      single-product
    DROP (overfit risk):
      PANEL_1X2+2X2                   613      1027       1.7σ    drift > std
      UV_VISOR_YELLOW+RED             675       893       1.3σ    drift > std
      SLEEP_POD_LAMB_WOOL+NYLON       798      1332       1.7σ    drift > std
      MICROCHIP_RECTANGLE+TRIANGLE   1298      2893       2.2σ    extreme drift, NOT cointegrated

For reference, the teammate's deployed pairs in v12 are all robust:
      VAN+CHOC                         76       155       2.0σ    LOW std baseline
      RASP+PIST                       180       325       1.8σ    LOW std baseline
      LAUNDRY+DISHES                  439       441       1.0σ    on the edge

Three additions kept:

  1. ROBOT_VACUUMING relative-value vs ROBOT_DISHES
     fair = 19185 - mid(DISHES), entry 100, exit 10
     measured BT: +40,489  [+15,160 / +14,527 / +10,802]
     Tightest cointegration: drift only 197 across days vs std 433.

  2. OXYGEN_SHAKE_MINT relative-value vs OXYGEN_SHAKE_EVENING_BREATH
     fair = 19110 - mid(EVE_BREATH), entry 700, exit 100
     measured BT: +35,005  [+15,895 / +9,950 / +9,160]
     Wide entry edge (700) is engineered to absorb daily mean drift.

  3. TRANSLATOR_VOID_BLUE hold-momentum (no flatten)
     short_w=200, long_w=400, entry_edge=50
     measured BT: +25,696  [+17,217 / +6,051 / +2,428]
     Single-product momentum — pair-independent. Signal decays across days
     (some OOS risk) but doesn't depend on a stale cointegration constant.

Predicted v14_robust BT total: 504,693 + 40,489 + 35,005 + 25,696 ≈ +605,883
(any small interaction effect to be measured by direct BT)
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

    # Products handed to teammate's directional logic + our v14_robust additions (not MM'd)
    DIRECTIONAL_PRODUCTS = frozenset({
        "MICROCHIP_CIRCLE",
        "MICROCHIP_TRIANGLE",
        "ROBOT_LAUNDRY",
        "SNACKPACK_RASPBERRY",
        "SNACKPACK_VANILLA",
        # v14_robust additions (only the 3 robust ones from v14_research's 7)
        "ROBOT_VACUUMING",
        "OXYGEN_SHAKE_MINT",
        "TRANSLATOR_VOID_BLUE",
    })

    # Teammate strategy params (verbatim from their trader_round5_mc_rb_sp.py)
    HOLD_MOMENTUM = {
        "MICROCHIP_CIRCLE": (100, 200, 30),  # short_w, long_w, entry_edge
        # v14: TRANSLATOR_VOID_BLUE hold-momentum
        "TRANSLATOR_VOID_BLUE": (200, 400, 50),
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

    # v14_robust RV additions (subset of v14_research's 7 — only tight pairs kept)
    VACUUMING_REF_SUM = 19185
    VACUUMING_EDGE    = 100
    VACUUMING_EXIT    = 10
    MINT_REF_SUM      = 19110
    MINT_EDGE         = 700
    MINT_EXIT         = 100

    MAX_HISTORY = 1200

    # v10 BASELINE map — ONLY the 5 swapped products set to 0; everything else unchanged
    BASELINE = {
        "PEBBLES_XS": 5, "PEBBLES_S": 5, "PEBBLES_M": 0, "PEBBLES_L": 5, "PEBBLES_XL": 10,
        "GALAXY_SOUNDS_DARK_MATTER": 5, "GALAXY_SOUNDS_BLACK_HOLES": 5,
        "GALAXY_SOUNDS_PLANETARY_RINGS": 5, "GALAXY_SOUNDS_SOLAR_WINDS": 5,
        "GALAXY_SOUNDS_SOLAR_FLAMES": 0,
        "SLEEP_POD_SUEDE": 5, "SLEEP_POD_LAMB_WOOL": 0, "SLEEP_POD_POLYESTER": 5,
        "SLEEP_POD_NYLON": 5, "SLEEP_POD_COTTON": 5,
        "MICROCHIP_CIRCLE":   0,
        "MICROCHIP_OVAL":     5,
        "MICROCHIP_SQUARE":   10,
        "MICROCHIP_RECTANGLE":0,
        "MICROCHIP_TRIANGLE": 0,
        "ROBOT_VACUUMING":    0,    # ← v14: handed to RV layer
        "ROBOT_MOPPING": 0, "ROBOT_DISHES": 0,
        "ROBOT_LAUNDRY":      0,
        "ROBOT_IRONING":      5,
        "UV_VISOR_YELLOW": 0, "UV_VISOR_AMBER": 5, "UV_VISOR_ORANGE": 5,
        "UV_VISOR_RED": 0, "UV_VISOR_MAGENTA": 0,
        "TRANSLATOR_SPACE_GRAY": 0, "TRANSLATOR_ASTRO_BLACK": 5,
        "TRANSLATOR_ECLIPSE_CHARCOAL": 5, "TRANSLATOR_GRAPHITE_MIST": 0,
        "TRANSLATOR_VOID_BLUE": 0,    # ← v14: handed to MOM layer
        "PANEL_1X2": 0,               # ← v14: handed to RV layer
        "PANEL_2X2": 5, "PANEL_1X4": 5, "PANEL_2X4": 5, "PANEL_4X4": 0,
        "OXYGEN_SHAKE_MORNING_BREATH": 5, "OXYGEN_SHAKE_EVENING_BREATH": 5,
        "OXYGEN_SHAKE_MINT": 0,       # ← v14: handed to RV layer
        "OXYGEN_SHAKE_CHOCOLATE": 5, "OXYGEN_SHAKE_GARLIC": 5,
        "SNACKPACK_CHOCOLATE":  5,
        "SNACKPACK_VANILLA":    0,
        "SNACKPACK_PISTACHIO":  5,
        "SNACKPACK_STRAWBERRY": 5,
        "SNACKPACK_RASPBERRY":  0,
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

        # --- Teammate's directional layer + v14 momentum on TRANSLATOR_VOID_BLUE ---
        self._trade_hold_momentum(result, vp, quote)         # CIRCLE + VOID_BLUE
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
        # --- v14_robust RV additions (only the 2 robust pairs) ---
        self._trade_reference_value(result, vp, quote, mids,
            sym="ROBOT_VACUUMING", ref_sym="ROBOT_DISHES",
            ref_sum=self.VACUUMING_REF_SUM,
            ent=self.VACUUMING_EDGE, exi=self.VACUUMING_EXIT)
        self._trade_reference_value(result, vp, quote, mids,
            sym="OXYGEN_SHAKE_MINT", ref_sym="OXYGEN_SHAKE_EVENING_BREATH",
            ref_sum=self.MINT_REF_SUM,
            ent=self.MINT_EDGE, exi=self.MINT_EXIT)

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
        history_syms = (self.DIRECTIONAL_PRODUCTS
            | {"ROBOT_DISHES", "SNACKPACK_CHOCOLATE", "SNACKPACK_PISTACHIO",
               "OXYGEN_SHAKE_EVENING_BREATH"})
        for sym in history_syms:
            if sym in mids:
                self.mid_history[sym].append(mids[sym])

        return result, 0, ""
