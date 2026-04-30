"""
trader_superwoman_v1.py — same as Superman Final EXCEPT HG.

HG idea: "Mark 22 sniper + Mark 38 cumulative fade".

Why
---
From our 3-day Mark analysis on HG:
  - Mark 22 buys HG (n=11): fwd_50k = -13.09  ← STRONG bearish signal
  - Mark 22 sells HG (n=8):  fwd_50k =  -5.19  ← bearish too
  Both Mark 22 directions on HG predict price DROP.
  Sample is small (only ~6 events / day total) but signal magnitude is huge.

  - Mark 38 buys HG: fwd_50k = -0.59  (his trades are mildly anti-predictive)
  - Mark 38 sells HG: fwd_50k = +0.55

How
---
HG handler differs from Superman:
  1. Default behavior — minimal: just quote bid_1+1 / ask_1-1 with size 20,
     similar to Superman but WITHOUT the EMA / Mark 14/38 alpha overlay.
  2. **Mark 22 sniper override**: when Mark 22 has just traded HG (any side),
     trigger a "short" sniper position — cross spread to sell 80 lots,
     hold for SNIPER_HOLD ticks, then unwind passively.
  3. **Mark 38 cumulative fade**: track his net flow over rolling 100k window.
     If |flow| > 30, lean OPPOSITE direction by skewing fair.

If the Mark 22 signal holds: 6 events × ~$10/share × 80 lots ≈ $5K/day extra.

State (mem):
  hg_ema, vev4000_ve_history, vev4000_ve_signal:  Superman carryovers
  m22_hg_short_until: int — timestamp until which we hold the sniper short
  m38_recent_flow: float — EWMA of Mark 38 net buy flow
"""
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Optional, Tuple
import json
import math


class Trader:
    HG_SYM = "HYDROGEL_PACK"
    VE_SYM = "VELVETFRUIT_EXTRACT"
    HG_LIMIT = 200
    VE_LIMIT = 200
    VEV_LIMIT = 300

    # ---- HG Superwoman: Mark 22 sniper + Mark 38 fade ----
    HG_BASE_QUOTE_SIZE = 20
    HG_BASE_HALF_SPREAD = 7
    HG_INV_SKEW = 0.05
    M22_SNIPE_QTY = 80
    M22_HOLD_TICKS = 50_000
    M38_FADE_DECAY = 0.998
    M38_FADE_WEIGHT = 0.05

    # ---- Swing windows (per-strike, from superman_final / v13) ----
    SWING_ENTRY_START = 350_000
    SWING_ENTRY_END = 880_000
    SWING_EXIT_START = 920_000
    SWING_EXIT_FORCE = 970_000
    VE_SWING_QUOTE_SIZE = 80

    VEV5300_ENTRY_START = 500_000
    VEV5300_ENTRY_END = 700_000
    VEV5300_EXIT_START = 900_000
    VEV5300_EXIT_FORCE = 970_000
    VE_ENTRY_START = 500_000
    VE_ENTRY_END = 700_000
    VE_EXIT_START = 900_000
    VE_EXIT_FORCE = 970_000

    SWING_VEV_STRIKES = [5100, 5200, 5300, 5400]
    VEV_SWING_QUOTE_SIZE = 80

    # ---- VEV_4000 v4 ----
    V4_QUOTE_SIZE = 60
    V4_INV_TILT = 0.0
    V4_BRAKE_THRESH = 100
    V4_BRAKE_TILT = 0.04
    V4_MIN_HALF_SPREAD = 8
    V4_VE_HISTORY_LEN = 10
    V4_VE_TRIGGER_MOVE = 3
    V4_VE_TRIGGER_LOOKBACK = 5
    V4_SNIPE_EDGE_MIN = 3
    V4_SNIPE_MAX_QTY = 60
    V4_SIGNAL_DECAY = 0.9995
    V4_SIGNAL_PER_LOT = 0.17
    V4_SIGNAL_MAX = 5.0

    # ---- VEV_5500 BS ----
    BS_TAKE_EDGE = 1.0
    BS_INV_SKEW = 0.08
    BASE_IV = 0.23
    T_DAYS = 4.0
    BS_VEV_STRIKES = [5500]

    # ==================================================================
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        try:
            mem = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            mem = {}

        result: Dict[str, List[Order]] = {}

        hg_orders = self._trade_hg(state, mem)
        if hg_orders: result[self.HG_SYM] = hg_orders

        ve_orders = self._trade_ve_swing(state)
        if ve_orders: result[self.VE_SYM] = ve_orders

        v4_orders = self._trade_vev4000_v4(state, mem)
        if v4_orders: result["VEV_4000"] = v4_orders

        for strike in self.SWING_VEV_STRIKES:
            sym = f"VEV_{strike}"
            orders = self._trade_vev_swing(state, sym)
            if orders: result[sym] = orders

        ve_mid = self._mid(state.order_depths.get(self.VE_SYM))
        if ve_mid is not None:
            for strike in self.BS_VEV_STRIKES:
                sym = f"VEV_{strike}"
                orders = self._trade_vev_bs(state, sym, strike, ve_mid)
                if orders: result[sym] = orders

        return result, 0, json.dumps(mem)

    # ==================================================================
    # HG — Superwoman (Mark 22 sniper + Mark 38 fade)
    # ==================================================================
    def _trade_hg(self, state: TradingState, mem: Dict) -> List[Order]:
        depth = state.order_depths.get(self.HG_SYM)
        if not self._book_ok(depth): return []
        bb = max(depth.buy_orders); ba = min(depth.sell_orders)
        if bb >= ba: return []
        mid = (bb + ba) / 2.0
        pos = int(state.position.get(self.HG_SYM, 0))
        ts = int(state.timestamp)

        # State
        m22_hold_until = int(mem.get("m22_hg_hold_until", -1))
        m38_flow = float(mem.get("m38_hg_flow", 0.0))

        # ── update Mark 22 sniper detection ──
        for t in state.market_trades.get(self.HG_SYM, []) or []:
            if t.buyer == "Mark 22" or t.seller == "Mark 22":
                # Mark 22 traded HG — fire sniper, hold until ts + HOLD
                m22_hold_until = ts + self.M22_HOLD_TICKS
        mem["m22_hg_hold_until"] = m22_hold_until

        # ── update Mark 38 cumulative net flow (EWMA) ──
        m38_flow *= self.M38_FADE_DECAY
        for t in state.market_trades.get(self.HG_SYM, []) or []:
            if t.buyer == "Mark 38":
                m38_flow += t.quantity
            elif t.seller == "Mark 38":
                m38_flow -= t.quantity
        mem["m38_hg_flow"] = m38_flow

        orders: List[Order] = []

        # ── Mode A: Mark 22 SNIPER active ──
        if ts < m22_hold_until:
            # Want to be SHORT (-100). Get there via cross-spread.
            target = -self.M22_SNIPE_QTY
            if pos > target:
                qty = min(pos - target, self.HG_LIMIT + pos)
                avail = int(depth.buy_orders[bb])
                qty = min(qty, avail)
                if qty > 0:
                    orders.append(Order(self.HG_SYM, bb, -qty))
            return orders

        # ── Mode B: passive MM with Mark 38 fade ──
        # If Mark 38 has been net buying (flow > 0), expect drop → fair DOWN
        # If Mark 38 has been net selling (flow < 0), expect rise → fair UP
        m38_skew = -self.M38_FADE_WEIGHT * m38_flow
        # Inventory skew (bring inventory back toward 0)
        inv_skew = -self.HG_INV_SKEW * pos
        fair_eff = mid + m38_skew + inv_skew

        bid_px = int(round(fair_eff - self.HG_BASE_HALF_SPREAD))
        ask_px = int(round(fair_eff + self.HG_BASE_HALF_SPREAD))
        bid_px = min(bid_px, ba - 1)
        ask_px = max(ask_px, bb + 1)
        if bid_px >= ask_px:
            bid_px = ask_px - 1

        bid_size = max(0, min(self.HG_BASE_QUOTE_SIZE, self.HG_LIMIT - pos))
        ask_size = max(0, min(self.HG_BASE_QUOTE_SIZE, self.HG_LIMIT + pos))

        if bid_size > 0:
            orders.append(Order(self.HG_SYM, bid_px, +bid_size))
        if ask_size > 0:
            orders.append(Order(self.HG_SYM, ask_px, -ask_size))
        return orders

    # ==================================================================
    # All other handlers — same as Superman Final
    # ==================================================================
    def _trade_ve_swing(self, state: TradingState) -> List[Order]:
        depth = state.order_depths.get(self.VE_SYM)
        if not self._book_ok(depth): return []
        bb = max(depth.buy_orders); ba = min(depth.sell_orders)
        if bb >= ba: return []
        pos = int(state.position.get(self.VE_SYM, 0))
        ts = int(state.timestamp)
        orders: List[Order] = []
        if ts < self.VE_ENTRY_START: pass
        elif ts < self.VE_ENTRY_END:
            room = max(0, self.VE_LIMIT - pos)
            if room > 0:
                size = min(self.VE_SWING_QUOTE_SIZE, room)
                orders.append(Order(self.VE_SYM, bb + 1, +size))
        elif ts < self.VE_EXIT_START: pass
        elif ts < self.VE_EXIT_FORCE:
            if pos > 0:
                size = min(self.VE_SWING_QUOTE_SIZE, pos)
                orders.append(Order(self.VE_SYM, ba - 1, -size))
        else:
            if pos > 0:
                avail = int(depth.buy_orders[bb])
                size = min(pos, avail)
                if size > 0: orders.append(Order(self.VE_SYM, bb, -size))
            elif pos < 0:
                avail = -int(depth.sell_orders[ba])
                size = min(-pos, avail)
                if size > 0: orders.append(Order(self.VE_SYM, ba, +size))
        return orders

    def _trade_vev_swing(self, state: TradingState, sym: str) -> List[Order]:
        depth = state.order_depths.get(sym)
        if not self._book_ok(depth): return []
        bb = max(depth.buy_orders); ba = min(depth.sell_orders)
        if bb >= ba: return []
        pos = int(state.position.get(sym, 0))
        ts = int(state.timestamp)

        if sym in ("VEV_5300", "VEV_5400"):
            entry_start, entry_end, exit_start, exit_force = (
                self.VEV5300_ENTRY_START, self.VEV5300_ENTRY_END,
                self.VEV5300_EXIT_START, self.VEV5300_EXIT_FORCE,
            )
        else:
            entry_start, entry_end, exit_start, exit_force = (
                self.SWING_ENTRY_START, self.SWING_ENTRY_END,
                self.SWING_EXIT_START, self.SWING_EXIT_FORCE,
            )

        orders: List[Order] = []
        if ts < entry_start: pass
        elif ts < entry_end:
            room = max(0, self.VEV_LIMIT - pos)
            if room > 0:
                size = min(self.VEV_SWING_QUOTE_SIZE, room)
                orders.append(Order(sym, bb + 1, +size))
        elif ts < exit_start: pass
        elif ts < exit_force:
            if pos > 0:
                size = min(self.VEV_SWING_QUOTE_SIZE, pos)
                orders.append(Order(sym, ba - 1, -size))
        else:
            if pos > 0:
                avail = int(depth.buy_orders[bb])
                size = min(pos, avail)
                if size > 0: orders.append(Order(sym, bb, -size))
            elif pos < 0:
                avail = -int(depth.sell_orders[ba])
                size = min(-pos, avail)
                if size > 0: orders.append(Order(sym, ba, +size))
        return orders

    def _trade_vev4000_v4(self, state: TradingState, mem: Dict) -> List[Order]:
        sym = "VEV_4000"
        depth = state.order_depths.get(sym)
        if not self._book_ok(depth): return []
        bb = max(depth.buy_orders); ba = min(depth.sell_orders)
        if bb >= ba: return []
        ve_book = state.order_depths.get(self.VE_SYM)
        ve_hist = list(mem.get("vev4000_ve_history", []))
        ve_sig = float(mem.get("vev4000_ve_signal", 0.0))
        ve_mid = self._mid(ve_book)
        if ve_mid is not None:
            ve_hist.append(ve_mid)
            if len(ve_hist) > self.V4_VE_HISTORY_LEN:
                ve_hist = ve_hist[-self.V4_VE_HISTORY_LEN:]
        ve_sig *= self.V4_SIGNAL_DECAY
        for t in state.market_trades.get(self.VE_SYM, []) or []:
            if t.buyer == "Mark 67": ve_sig += self.V4_SIGNAL_PER_LOT * t.quantity
            elif t.seller == "Mark 49": ve_sig += self.V4_SIGNAL_PER_LOT * t.quantity
        ve_sig = max(-self.V4_SIGNAL_MAX, min(self.V4_SIGNAL_MAX, ve_sig))
        mem["vev4000_ve_history"] = ve_hist
        mem["vev4000_ve_signal"] = ve_sig

        pos = int(state.position.get(sym, 0))
        orders: List[Order] = []
        if ve_mid is not None and len(ve_hist) >= self.V4_VE_TRIGGER_LOOKBACK + 1:
            recent = ve_hist[-1] - ve_hist[-1 - self.V4_VE_TRIGGER_LOOKBACK]
            if abs(recent) >= self.V4_VE_TRIGGER_MOVE:
                vev_fair = ve_mid - 4000.0
                if bb >= vev_fair + self.V4_SNIPE_EDGE_MIN:
                    avail = int(depth.buy_orders[bb])
                    qty = min(avail, self.V4_SNIPE_MAX_QTY, max(0, self.VEV_LIMIT + pos))
                    if qty > 0: orders.append(Order(sym, bb, -qty))
                if ba <= vev_fair - self.V4_SNIPE_EDGE_MIN:
                    avail = -int(depth.sell_orders[ba])
                    qty = min(avail, self.V4_SNIPE_MAX_QTY, max(0, self.VEV_LIMIT - pos))
                    if qty > 0: orders.append(Order(sym, ba, +qty))
        existing_buy = sum(o.quantity for o in orders if o.quantity > 0)
        existing_sell = sum(-o.quantity for o in orders if o.quantity < 0)
        bid_room = max(0, self.VEV_LIMIT - pos - existing_buy)
        ask_room = max(0, self.VEV_LIMIT + pos - existing_sell)
        bid_px = bb + 1; ask_px = ba - 1
        if abs(pos) <= self.V4_BRAKE_THRESH:
            inv_skew = -self.V4_INV_TILT * pos
        else:
            excess = abs(pos) - self.V4_BRAKE_THRESH
            sgn = 1 if pos > 0 else -1
            inv_skew = -self.V4_BRAKE_TILT * excess * sgn
        total_skew = inv_skew + ve_sig
        bid_px = int(round(bid_px + total_skew))
        ask_px = int(round(ask_px + total_skew))
        natural_mid = (bb + ba) / 2.0
        bid_px = min(bid_px, int(natural_mid - self.V4_MIN_HALF_SPREAD))
        ask_px = max(ask_px, int(natural_mid + self.V4_MIN_HALF_SPREAD))
        bid_px = min(bid_px, ba - 1); ask_px = max(ask_px, bb + 1)
        if bid_px >= ask_px: bid_px = ask_px - 1
        bid_size = min(self.V4_QUOTE_SIZE, bid_room)
        ask_size = min(self.V4_QUOTE_SIZE, ask_room)
        if bid_size > 0: orders.append(Order(sym, bid_px, +bid_size))
        if ask_size > 0: orders.append(Order(sym, ask_px, -ask_size))
        return orders

    def _trade_vev_bs(self, state: TradingState, sym: str, strike: int, ve_mid: float) -> List[Order]:
        depth = state.order_depths.get(sym)
        if not self._book_ok(depth): return []
        bb = max(depth.buy_orders); ba = min(depth.sell_orders)
        pos = int(state.position.get(sym, 0))
        theo = self._bs_call(ve_mid, strike, self.BASE_IV, self.T_DAYS)
        fair = theo - self.BS_INV_SKEW * pos
        orders: List[Order] = []
        cur_pos = pos
        if ba <= fair - self.BS_TAKE_EDGE:
            avail = -depth.sell_orders[ba]
            qty = max(0, min(avail, self.VEV_LIMIT - cur_pos))
            if qty > 0: orders.append(Order(sym, ba, qty)); cur_pos += qty
        if bb >= fair + self.BS_TAKE_EDGE:
            avail = depth.buy_orders[bb]
            qty = max(0, min(avail, self.VEV_LIMIT + cur_pos))
            if qty > 0: orders.append(Order(sym, bb, -qty))
        return orders

    @staticmethod
    def _book_ok(book) -> bool:
        return book is not None and book.buy_orders and book.sell_orders
    @staticmethod
    def _mid(book) -> Optional[float]:
        if book is None or not book.buy_orders or not book.sell_orders: return None
        return (max(book.buy_orders) + min(book.sell_orders)) / 2.0
    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    def _bs_call(self, s: float, k: float, sigma: float, t_days: float) -> float:
        if s <= 0 or k <= 0: return 0.0
        total_vol = sigma * math.sqrt(t_days / 365.0)
        if total_vol <= 1e-9: return max(s - k, 0.0)
        d1 = (math.log(s / k) + 0.5 * total_vol * total_vol) / total_vol
        d2 = d1 - total_vol
        return s * self._norm_cdf(d1) - k * self._norm_cdf(d2)
