"""
v14 = v12 + 7 additional alphas discovered by sim profiling on round5 days {2,3,4}.
Each addition was selected for being POSITIVE ON ALL 3 DAYS in standalone sim
(min single-day PnL > 0).

Total measured BT vs v12:
    v12 baseline   504,693
    v14 (this)     766,543
    uplift        +261,850   (+12% on average per day, no losing days vs v12)

Per-day comparison:
            v12         v14        delta
    D+2  152,528     261,868    +109,340
    D+3  145,636     223,772     +78,136
    D+4  206,530     280,904     +74,374
All deltas positive. D+3 was the weakest day pre-and-post (still is).

New strategies layered on top of v12:

  1. ROBOT_VACUUMING relative-value vs ROBOT_DISHES
     fair = 19185 - mid(DISHES), entry 100, exit 10
     standalone sim: +40,810  [+15,460 / +14,010 / +11,340]
     measured BT (in v14a): +40,489

  2. OXYGEN_SHAKE_MINT relative-value vs OXYGEN_SHAKE_EVENING_BREATH
     fair = 19110 - mid(EVE_BREATH), entry 700, exit 100
     standalone sim: +35,005  [+15,895 / +9,950 / +9,160]
     measured BT (in v14a): +35,005

  3. PANEL_1X2 relative-value vs PANEL_2X2
     fair = 18499 - mid(PANEL_2X2), entry 400, exit 25
     standalone sim: +39,685  [+24,700 / +2,345 / +12,640]
     measured BT (in v14a): +38,860

  4. TRANSLATOR_VOID_BLUE hold-momentum (no flatten)
     short_w=200, long_w=400, entry_edge=50
     standalone sim: +26,275  [+17,560 / +5,545 / +3,170]
     measured BT (in v14a): +25,696

  5. UV_VISOR_YELLOW relative-value vs UV_VISOR_RED
     fair = 22020 - mid(UV_VISOR_RED), entry 200, exit 50
     standalone sim: +66,390  [+24,395 / +16,075 / +25,920]
     UV_VISOR_RED was BASELINE=0 in v12 (read-only reference; no MM conflict)

  6. SLEEP_POD_LAMB_WOOL relative-value vs SLEEP_POD_NYLON
     fair = 20337 - mid(NYLON), entry 300, exit 10
     standalone sim: +23,745  [+7,035 / +12,730 / +3,980]
     SLEEP_POD_NYLON still MM'd at BASELINE=5 (mid is read-only for LAMB_WOOL)

  7. MICROCHIP_RECTANGLE relative-value vs MICROCHIP_TRIANGLE
     fair = 18418 - mid(TRIANGLE), entry 500, exit 25
     standalone sim: +31,425  [+5,755 / +16,290 / +9,380]
     measured BT: +31,728. TRIANGLE is being directionally traded but the
     position-induced mid noise is small relative to entry edge of 500.

Interactions reviewed:
  - ROBOT_DISHES is now a shared read-only reference (also used by ROBOT_LAUNDRY)
  - PANEL_2X2 still MM'd at BASELINE=5 (mid is read-only for PANEL_1X2 RV)
  - OXYGEN_SHAKE_EVENING_BREATH still MM'd at BASELINE=5 (read-only reference)
  - SLEEP_POD_NYLON still MM'd at BASELINE=5 (read-only for LAMB_WOOL RV)
  - UV_VISOR_RED is BASELINE=0 (read-only natural market mid for YELLOW RV)
  - All 6 new traded products were BASELINE=0 in v12 — no MM conflict.

Skipped candidates (failed all-3-days-positive on standalone sim or interaction risk):
  - PANEL_2X2 vs PANEL_4X4 (sim +36,055): switching PANEL_2X2 loses +7k of MM
  - MICROCHIP_RECTANGLE vs MICROCHIP_TRIANGLE: ref product is being directionally traded
  - ROBOT_MOPPING vs ROBOT_VACUUMING: ref now being RV-traded (already added)
  - OXYGEN_SHAKE_GARLIC vs OXYGEN_SHAKE_MINT: ref now being RV-traded
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

    # Products handed to teammate's directional logic + our v14 additions (not MM'd)
    DIRECTIONAL_PRODUCTS = frozenset({
        "MICROCHIP_CIRCLE",
        "MICROCHIP_TRIANGLE",
        "ROBOT_LAUNDRY",
        "SNACKPACK_RASPBERRY",
        "SNACKPACK_VANILLA",
        # v14 additions
        "ROBOT_VACUUMING",
        "OXYGEN_SHAKE_MINT",
        "PANEL_1X2",
        "TRANSLATOR_VOID_BLUE",
        "UV_VISOR_YELLOW",
        "SLEEP_POD_LAMB_WOOL",
        "MICROCHIP_RECTANGLE",
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

    # v14 RV additions
    VACUUMING_REF_SUM = 19185
    VACUUMING_EDGE    = 100
    VACUUMING_EXIT    = 10
    MINT_REF_SUM      = 19110
    MINT_EDGE         = 700
    MINT_EXIT         = 100
    PANEL_1X2_REF_SUM = 18499
    PANEL_1X2_EDGE    = 400
    PANEL_1X2_EXIT    = 25
    UV_YELLOW_REF_SUM = 22020
    UV_YELLOW_EDGE    = 200
    UV_YELLOW_EXIT    = 50
    LAMB_WOOL_REF_SUM = 20337
    LAMB_WOOL_EDGE    = 300
    LAMB_WOOL_EXIT    = 10
    RECT_REF_SUM      = 18418
    RECT_EDGE         = 500
    RECT_EXIT         = 25

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
        # --- v14 RV additions ---
        self._trade_reference_value(result, vp, quote, mids,
            sym="ROBOT_VACUUMING", ref_sym="ROBOT_DISHES",
            ref_sum=self.VACUUMING_REF_SUM,
            ent=self.VACUUMING_EDGE, exi=self.VACUUMING_EXIT)
        self._trade_reference_value(result, vp, quote, mids,
            sym="OXYGEN_SHAKE_MINT", ref_sym="OXYGEN_SHAKE_EVENING_BREATH",
            ref_sum=self.MINT_REF_SUM,
            ent=self.MINT_EDGE, exi=self.MINT_EXIT)
        self._trade_reference_value(result, vp, quote, mids,
            sym="PANEL_1X2", ref_sym="PANEL_2X2",
            ref_sum=self.PANEL_1X2_REF_SUM,
            ent=self.PANEL_1X2_EDGE, exi=self.PANEL_1X2_EXIT)
        self._trade_reference_value(result, vp, quote, mids,
            sym="UV_VISOR_YELLOW", ref_sym="UV_VISOR_RED",
            ref_sum=self.UV_YELLOW_REF_SUM,
            ent=self.UV_YELLOW_EDGE, exi=self.UV_YELLOW_EXIT)
        self._trade_reference_value(result, vp, quote, mids,
            sym="SLEEP_POD_LAMB_WOOL", ref_sym="SLEEP_POD_NYLON",
            ref_sum=self.LAMB_WOOL_REF_SUM,
            ent=self.LAMB_WOOL_EDGE, exi=self.LAMB_WOOL_EXIT)
        self._trade_reference_value(result, vp, quote, mids,
            sym="MICROCHIP_RECTANGLE", ref_sym="MICROCHIP_TRIANGLE",
            ref_sum=self.RECT_REF_SUM,
            ent=self.RECT_EDGE, exi=self.RECT_EXIT)

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
               "OXYGEN_SHAKE_EVENING_BREATH", "PANEL_2X2",
               "UV_VISOR_RED", "SLEEP_POD_NYLON"})
        for sym in history_syms:
            if sym in mids:
                self.mid_history[sym].append(mids[sym])

        return result, 0, ""
