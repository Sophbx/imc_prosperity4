"""
trader_batman.py — Superman Final but with R3 v4's HG swing-trader.

Same as trader_superman_final.py EXCEPT:
  HG: replaced teammate's MM (HP_ALPHA + EMA + take/make) with R3 v4's
      z-score state-machine swing trader (mode flat/short/long, 3.35σ
      reversal flip, etc.). On R4 dataset 3-day backtest, R3 v4 HG
      makes $66,104 vs teammate's $17,746 → +$48,358 over 3 days.

Trade-off vs Superman Final:
  + Bigger expected PnL (Day 2 in R4 dataset gave HG $32K alone for v4)
  − Path stability worse (state machine sits idle ~80% of time)
  − Day-to-day variance higher (depends on swing pattern existing)
  − Does NOT use Round 4 counterparty info on HG (v4 was R3 code, no Mark IDs)

Source of HG logic: Round3/trader_v4.py (= live R3 submission 484304.py).
This is the IDENTICAL state machine that delivered $19,551 HG on the
live R3 day (= R4 dataset Day 3, also $19,551 in R4 backtest).

Per-product strategy (chosen by per-product 3-day backtest winner):

  HG          → teammate's MM   (HP_ALPHA, EMA, take + make)
  VE          → my swing v1     (5-phase intraday swing)
  VEV_4000    → my v4           (passive MM + VE directional signal)
  VEV_4500    → SKIP            (流量死区)
  VEV_5000    → SKIP            (BS over-prices, structural loss)
  VEV_5100    → my swing
  VEV_5200    → my swing
  VEV_5300    → my swing
  VEV_5400    → my swing
  VEV_5500    → teammate's BS take/hit
  VEV_6000    → SKIP            (mid pin at 0.5, no swing alpha)
  VEV_6500    → SKIP

Per-product PnL (3 days):
  HG       $17,746   VE      $21,067   VEV_4000 $8,553   VEV_5100 $6,333
  VEV_5200 $13,031   VEV_5300 $8,694   VEV_5400 $1,461   VEV_5500   $613

Key insight: per-strike swing windows.
  VEV_5100, VEV_5200: prefer WIDE entry (350-880k), later exit start (920k)
  VE, VEV_5300, VEV_5400: prefer narrow v1 windows (500-700, 900-970)

Architecture
------------
- All state persisted via JSON-encoded traderData (Lambda-safe).
- Each product handler is independent — no cross-coupling besides the
  shared market_trades / order_depths reads.
- Position limits enforced per product before any order emit.
- Defensive empty-book handling on every product.

State carried in traderData:
  hg_ema:                EMA of HG mid (teammate's HG fair calc)
  ve_swing_unused:       (no state — swing is timestamp-based)
  vev4000_ve_history:    last N VE mids (for VEV_4000 v4 sniping)
  vev4000_ve_signal:     accumulated bullish flow signal for VEV_4000 v4
"""
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Optional, Tuple
import json
import math


class Trader:
    # ------------------------------------------------------------------
    # Symbols & limits
    # ------------------------------------------------------------------
    HG_SYM = "HYDROGEL_PACK"
    VE_SYM = "VELVETFRUIT_EXTRACT"
    HG_LIMIT = 200
    VE_LIMIT = 200
    VEV_LIMIT = 300

    # ------------------------------------------------------------------
    # HG — R3 v4 swing trader (state machine, no counterparty info)
    # ------------------------------------------------------------------
    HYDRO_FAIR = 10000.0
    HYDRO_SIGMA = 32.0
    HYDRO_ENTRY_Z_1 = 0.85
    HYDRO_ENTRY_Z_2 = 1.25
    HYDRO_ENTRY_Z_3 = 1.65
    HYDRO_SIZE_1 = 120
    HYDRO_SIZE_2 = 160
    HYDRO_SIZE_3 = 200
    HYDRO_REVERSAL_Z = 3.35
    HYDRO_REBOUND_Z = 1.20
    HYDRO_MAX_TRADE = 40
    HYDRO_COOLDOWN = 1200
    HYDRO_RESET_Z = 0.35

    # ------------------------------------------------------------------
    # VE — my swing v1 phases
    # ------------------------------------------------------------------
    # v8 per-product windows (tuned individually):
    # VE / 5100 / 5200 / 5400 — wider entry, later exit start (v6d)
    SWING_ENTRY_START = 350_000
    SWING_ENTRY_END = 880_000
    SWING_EXIT_START = 920_000
    SWING_EXIT_FORCE = 970_000
    VE_SWING_QUOTE_SIZE = 80

    # VEV_5300 — keep v1 windows (longer entry hurt this strike)
    VEV5300_ENTRY_START = 500_000
    VEV5300_ENTRY_END = 700_000
    VEV5300_EXIT_START = 900_000
    VEV5300_EXIT_FORCE = 970_000

    # VE — try v1 windows (v6d hurt VE by $1,171)
    VE_ENTRY_START = 500_000
    VE_ENTRY_END = 700_000
    VE_EXIT_START = 900_000
    VE_EXIT_FORCE = 970_000

    # ------------------------------------------------------------------
    # VEV_4000 — my v4 (passive MM + VE signal)
    # ------------------------------------------------------------------
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
    V4_SIGNAL_DECAY = 0.999
    V4_SIGNAL_PER_LOT = 0.5
    V4_SIGNAL_MAX = 10.0

    # ------------------------------------------------------------------
    # VEV swing strikes
    # ------------------------------------------------------------------
    SWING_VEV_STRIKES = [5100, 5200, 5300, 5400]
    VEV_SWING_QUOTE_SIZE = 80

    # ------------------------------------------------------------------
    # VEV_5500 — teammate's BS take/hit
    # ------------------------------------------------------------------
    BS_TAKE_EDGE = 1.0
    BS_INV_SKEW = 0.08
    BASE_IV = 0.23
    T_DAYS = 4.0
    BS_VEV_STRIKES = [5500]

    # ==================================================================
    # Main run
    # ==================================================================
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        try:
            mem = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            mem = {}

        result: Dict[str, List[Order]] = {}

        # 1. HG — teammate's MM
        hg_orders = self._trade_hg(state, mem)
        if hg_orders:
            result[self.HG_SYM] = hg_orders

        # 2. VE — my swing
        ve_orders = self._trade_ve_swing(state)
        if ve_orders:
            result[self.VE_SYM] = ve_orders

        # 3. VEV_4000 — my v4 (passive MM + VE signal)
        v4_orders = self._trade_vev4000_v4(state, mem)
        if v4_orders:
            result["VEV_4000"] = v4_orders

        # 4. VEV swing strikes (5100, 5200, 5300, 5400)
        for strike in self.SWING_VEV_STRIKES:
            sym = f"VEV_{strike}"
            orders = self._trade_vev_swing(state, sym)
            if orders:
                result[sym] = orders

        # 5. VEV BS strikes (5500)
        ve_mid = self._mid(state.order_depths.get(self.VE_SYM))
        if ve_mid is not None:
            for strike in self.BS_VEV_STRIKES:
                sym = f"VEV_{strike}"
                orders = self._trade_vev_bs(state, sym, strike, ve_mid)
                if orders:
                    result[sym] = orders

        return result, 0, json.dumps(mem)

    # ==================================================================
    # HG — R3 v4 swing-trader (state machine: flat / short / long)
    # ==================================================================
    def _hydro_desired_short_size(self, z: float) -> int:
        if z >= self.HYDRO_ENTRY_Z_3:
            return self.HYDRO_SIZE_3
        if z >= self.HYDRO_ENTRY_Z_2:
            return self.HYDRO_SIZE_2
        if z >= self.HYDRO_ENTRY_Z_1:
            return self.HYDRO_SIZE_1
        return 0

    def _hydro_target(self, mid: float, timestamp: int, hd: Dict) -> int:
        mode = hd.get("mode", "flat")
        extreme = hd.get("extreme")
        cooldown_until = int(hd.get("cooldown_until", -1))
        needs_reset = bool(hd.get("needs_reset", False))

        z = (mid - self.HYDRO_FAIR) / self.HYDRO_SIGMA
        reversal = self.HYDRO_REVERSAL_Z * self.HYDRO_SIGMA
        rebound = self.HYDRO_REBOUND_Z * self.HYDRO_SIGMA
        reset_band = self.HYDRO_RESET_Z * self.HYDRO_SIGMA
        reset_low = self.HYDRO_FAIR - reset_band
        reset_high = self.HYDRO_FAIR + reset_band

        if mode == "flat":
            hd["extreme"] = None
            hd["short_size"] = 0
            if needs_reset:
                if reset_low <= mid <= reset_high:
                    hd["needs_reset"] = False
                else:
                    return 0
            if timestamp < cooldown_until:
                return 0
            short_size = self._hydro_desired_short_size(z)
            if short_size > 0:
                hd["mode"] = "short"
                hd["extreme"] = mid
                hd["short_size"] = short_size
                return -short_size
            return 0

        if mode == "short":
            high = max(float(extreme or mid), mid)
            hd["extreme"] = high
            old_short_size = int(hd.get("short_size", self.HYDRO_SIZE_1))
            desired_short_size = self._hydro_desired_short_size(z)
            short_size = max(old_short_size, desired_short_size)
            if short_size <= 0:
                short_size = self.HYDRO_SIZE_1
            short_size = min(short_size, self.HG_LIMIT)
            hd["short_size"] = short_size
            if mid <= high - reversal:
                hd["mode"] = "long"
                hd["extreme"] = mid
                hd["short_size"] = 0
                return self.HG_LIMIT
            return -short_size

        if mode == "long":
            low = min(float(extreme or mid), mid)
            hd["extreme"] = low
            if mid >= low + rebound:
                hd["mode"] = "flat"
                hd["extreme"] = None
                hd["cooldown_until"] = timestamp + self.HYDRO_COOLDOWN
                hd["needs_reset"] = True
                hd["short_size"] = 0
                return 0
            return self.HG_LIMIT

        # Unknown mode — recover to flat.
        hd["mode"] = "flat"
        hd["extreme"] = None
        hd["needs_reset"] = False
        hd["short_size"] = 0
        return 0

    def _trade_hg(self, state: TradingState, mem: Dict) -> List[Order]:
        depth = state.order_depths.get(self.HG_SYM)
        if not self._book_ok(depth):
            return []
        bb = max(depth.buy_orders); ba = min(depth.sell_orders)
        if bb is None or ba is None:
            return []

        mid = (bb + ba) / 2.0
        position = int(state.position.get(self.HG_SYM, 0))

        # State-machine state in mem
        hd = mem.setdefault("hydro", {
            "mode": "flat",
            "extreme": None,
            "cooldown_until": -1,
            "needs_reset": False,
            "short_size": 0,
        })
        target = self._hydro_target(mid, int(state.timestamp), hd)

        buy_cap = max(0, self.HG_LIMIT - position)
        sell_cap = max(0, self.HG_LIMIT + position)

        orders: List[Order] = []
        if target > position and buy_cap > 0:
            visible = max(0, -depth.sell_orders.get(ba, 0))
            qty = min(target - position, buy_cap, self.HYDRO_MAX_TRADE, visible)
            if qty > 0:
                orders.append(Order(self.HG_SYM, ba, qty))
        elif target < position and sell_cap > 0:
            visible = max(0, depth.buy_orders.get(bb, 0))
            qty = min(position - target, sell_cap, self.HYDRO_MAX_TRADE, visible)
            if qty > 0:
                orders.append(Order(self.HG_SYM, bb, -qty))
        return orders

    # NOTE: Below dead — kept only because older code paths reference it.
    def _trade_hg_OLD(self, state, mem):
        depth = state.order_depths.get(self.HG_SYM)
        bb = max(depth.buy_orders); ba = min(depth.sell_orders)
        cur_pos = 0
        orders: List[Order] = []
        if False:
            qty = 0
            if qty > 0:
                orders.append(Order(self.HG_SYM, sell_px, -qty))
        return orders

    # ==================================================================
    # VE — my swing v1
    # ==================================================================
    def _trade_ve_swing(self, state: TradingState) -> List[Order]:
        depth = state.order_depths.get(self.VE_SYM)
        if not self._book_ok(depth):
            return []
        bb = max(depth.buy_orders); ba = min(depth.sell_orders)
        if bb >= ba: return []
        pos = int(state.position.get(self.VE_SYM, 0))
        ts = int(state.timestamp)

        orders: List[Order] = []
        if ts < self.VE_ENTRY_START:
            pass
        elif ts < self.VE_ENTRY_END:
            room = max(0, self.VE_LIMIT - pos)
            if room > 0:
                size = min(self.VE_SWING_QUOTE_SIZE, room)
                orders.append(Order(self.VE_SYM, bb + 1, +size))
        elif ts < self.VE_EXIT_START:
            pass
        elif ts < self.VE_EXIT_FORCE:
            if pos > 0:
                size = min(self.VE_SWING_QUOTE_SIZE, pos)
                orders.append(Order(self.VE_SYM, ba - 1, -size))
        else:
            if pos > 0:
                avail = int(depth.buy_orders[bb])
                size = min(pos, avail)
                if size > 0:
                    orders.append(Order(self.VE_SYM, bb, -size))
            elif pos < 0:
                avail = -int(depth.sell_orders[ba])
                size = min(-pos, avail)
                if size > 0:
                    orders.append(Order(self.VE_SYM, ba, +size))
        return orders

    # ==================================================================
    # VEV_4000 — my v4 (passive MM + VE signal)
    # ==================================================================
    def _trade_vev4000_v4(self, state: TradingState, mem: Dict) -> List[Order]:
        sym = "VEV_4000"
        depth = state.order_depths.get(sym)
        if not self._book_ok(depth):
            return []
        bb = max(depth.buy_orders); ba = min(depth.sell_orders)
        if bb >= ba: return []

        ve_book = state.order_depths.get(self.VE_SYM)
        ve_hist: List[float] = list(mem.get("vev4000_ve_history", []))
        ve_sig: float = float(mem.get("vev4000_ve_signal", 0.0))

        ve_mid = self._mid(ve_book)
        if ve_mid is not None:
            ve_hist.append(ve_mid)
            if len(ve_hist) > self.V4_VE_HISTORY_LEN:
                ve_hist = ve_hist[-self.V4_VE_HISTORY_LEN:]

        # Update VE signal
        ve_sig *= self.V4_SIGNAL_DECAY
        for t in state.market_trades.get(self.VE_SYM, []) or []:
            if t.buyer == "Mark 67":
                ve_sig += self.V4_SIGNAL_PER_LOT * t.quantity
            elif t.seller == "Mark 49":
                ve_sig += self.V4_SIGNAL_PER_LOT * t.quantity
        ve_sig = max(-self.V4_SIGNAL_MAX, min(self.V4_SIGNAL_MAX, ve_sig))

        mem["vev4000_ve_history"] = ve_hist
        mem["vev4000_ve_signal"] = ve_sig

        pos = int(state.position.get(sym, 0))
        orders: List[Order] = []

        # Layer 3: stale-quote snipe
        if ve_mid is not None and len(ve_hist) >= self.V4_VE_TRIGGER_LOOKBACK + 1:
            recent = ve_hist[-1] - ve_hist[-1 - self.V4_VE_TRIGGER_LOOKBACK]
            if abs(recent) >= self.V4_VE_TRIGGER_MOVE:
                vev_fair = ve_mid - 4000.0
                if bb >= vev_fair + self.V4_SNIPE_EDGE_MIN:
                    avail = int(depth.buy_orders[bb])
                    qty = min(avail, self.V4_SNIPE_MAX_QTY, max(0, self.VEV_LIMIT + pos))
                    if qty > 0:
                        orders.append(Order(sym, bb, -qty))
                if ba <= vev_fair - self.V4_SNIPE_EDGE_MIN:
                    avail = -int(depth.sell_orders[ba])
                    qty = min(avail, self.V4_SNIPE_MAX_QTY, max(0, self.VEV_LIMIT - pos))
                    if qty > 0:
                        orders.append(Order(sym, ba, +qty))

        # Layer 1: passive MM with inv skew + VE signal
        existing_buy = sum(o.quantity for o in orders if o.quantity > 0)
        existing_sell = sum(-o.quantity for o in orders if o.quantity < 0)
        bid_room = max(0, self.VEV_LIMIT - pos - existing_buy)
        ask_room = max(0, self.VEV_LIMIT + pos - existing_sell)

        bid_px = bb + 1
        ask_px = ba - 1

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
        bid_px = min(bid_px, ba - 1)
        ask_px = max(ask_px, bb + 1)
        if bid_px >= ask_px:
            bid_px = ask_px - 1

        bid_size = min(self.V4_QUOTE_SIZE, bid_room)
        ask_size = min(self.V4_QUOTE_SIZE, ask_room)
        if bid_size > 0:
            orders.append(Order(sym, bid_px, +bid_size))
        if ask_size > 0:
            orders.append(Order(sym, ask_px, -ask_size))
        return orders

    # ==================================================================
    # VEV swing (5100, 5200, 5300, 5400)
    # ==================================================================
    def _trade_vev_swing(self, state: TradingState, sym: str) -> List[Order]:
        depth = state.order_depths.get(sym)
        if not self._book_ok(depth):
            return []
        bb = max(depth.buy_orders); ba = min(depth.sell_orders)
        if bb >= ba: return []
        pos = int(state.position.get(sym, 0))
        ts = int(state.timestamp)

        # v13: per-strike windows; VEV_5300 and VEV_5400 use v1 (narrow) windows
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
        if ts < entry_start:
            pass
        elif ts < entry_end:
            room = max(0, self.VEV_LIMIT - pos)
            if room > 0:
                size = min(self.VEV_SWING_QUOTE_SIZE, room)
                orders.append(Order(sym, bb + 1, +size))
        elif ts < exit_start:
            pass
        elif ts < exit_force:
            if pos > 0:
                size = min(self.VEV_SWING_QUOTE_SIZE, pos)
                orders.append(Order(sym, ba - 1, -size))
        else:
            if pos > 0:
                avail = int(depth.buy_orders[bb])
                size = min(pos, avail)
                if size > 0:
                    orders.append(Order(sym, bb, -size))
            elif pos < 0:
                avail = -int(depth.sell_orders[ba])
                size = min(-pos, avail)
                if size > 0:
                    orders.append(Order(sym, ba, +size))
        return orders

    # ==================================================================
    # VEV_5500 — teammate's BS take/hit
    # ==================================================================
    def _trade_vev_bs(
        self, state: TradingState, sym: str, strike: int, ve_mid: float
    ) -> List[Order]:
        depth = state.order_depths.get(sym)
        if not self._book_ok(depth):
            return []
        bb = max(depth.buy_orders); ba = min(depth.sell_orders)
        pos = int(state.position.get(sym, 0))

        theo = self._bs_call(ve_mid, strike, self.BASE_IV, self.T_DAYS)
        fair = theo - self.BS_INV_SKEW * pos

        orders: List[Order] = []
        cur_pos = pos
        if ba <= fair - self.BS_TAKE_EDGE:
            avail = -depth.sell_orders[ba]
            qty = max(0, min(avail, self.VEV_LIMIT - cur_pos))
            if qty > 0:
                orders.append(Order(sym, ba, qty))
                cur_pos += qty
        if bb >= fair + self.BS_TAKE_EDGE:
            avail = depth.buy_orders[bb]
            qty = max(0, min(avail, self.VEV_LIMIT + cur_pos))
            if qty > 0:
                orders.append(Order(sym, bb, -qty))
        return orders

    # ==================================================================
    # Helpers
    # ==================================================================
    @staticmethod
    def _book_ok(book) -> bool:
        return book is not None and book.buy_orders and book.sell_orders

    @staticmethod
    def _mid(book) -> Optional[float]:
        if book is None or not book.buy_orders or not book.sell_orders:
            return None
        return (max(book.buy_orders) + min(book.sell_orders)) / 2.0

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _bs_call(self, s: float, k: float, sigma: float, t_days: float) -> float:
        if s <= 0 or k <= 0:
            return 0.0
        total_vol = sigma * math.sqrt(t_days / 365.0)
        if total_vol <= 1e-9:
            return max(s - k, 0.0)
        d1 = (math.log(s / k) + 0.5 * total_vol * total_vol) / total_vol
        d2 = d1 - total_vol
        return s * self._norm_cdf(d1) - k * self._norm_cdf(d2)
