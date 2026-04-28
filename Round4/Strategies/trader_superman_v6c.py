"""
trader_superman_v6c.py — entry 400k + exit 940k

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

Expected backtest PnL (sum of per-product winners, 3 days):
  HG: $17,746 | VE: $21,067 | VEV_4000: $8,553 | VEV_5100: $3,361
  VEV_5200: $7,972 | VEV_5300: $8,694 | VEV_5400: $1,461 | VEV_5500: $613
  Total ≈ $69,500 / 3 days = ~$23,200/day

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
    # HG — teammate's MM (HP_ALPHA + EMA + take + make)
    # ------------------------------------------------------------------
    HP_ALPHA = {"Mark 14": 1.5, "Mark 38": -1.5}
    HG_EMA_ALPHA = 0.08
    HG_TAKE_EDGE = 2.5
    HG_QUOTE_EDGE = 7
    HG_QUOTE_SIZE = 20
    HG_INV_SKEW = 0.15

    # ------------------------------------------------------------------
    # VE — my swing v1 phases
    # ------------------------------------------------------------------
    SWING_ENTRY_START = 400_000
    SWING_ENTRY_END = 700_000
    SWING_EXIT_START = 940_000
    SWING_EXIT_FORCE = 970_000
    VE_SWING_QUOTE_SIZE = 80

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
    V4_SIGNAL_DECAY = 0.9995
    V4_SIGNAL_PER_LOT = 0.17
    V4_SIGNAL_MAX = 5.0

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
    # HG — teammate's strategy
    # ==================================================================
    def _trade_hg(self, state: TradingState, mem: Dict) -> List[Order]:
        depth = state.order_depths.get(self.HG_SYM)
        if not self._book_ok(depth):
            return []

        bb = max(depth.buy_orders); ba = min(depth.sell_orders)
        mid = (bb + ba) / 2.0
        pos = int(state.position.get(self.HG_SYM, 0))

        # EMA
        ema = mem.get("hg_ema")
        ema = mid if ema is None else self.HG_EMA_ALPHA * mid + (1 - self.HG_EMA_ALPHA) * ema
        mem["hg_ema"] = ema

        # Mark alpha signal
        sig = 0.0
        for t in state.market_trades.get(self.HG_SYM, []) or []:
            if t.buyer in self.HP_ALPHA:
                sig += self.HP_ALPHA[t.buyer]
            if t.seller in self.HP_ALPHA:
                sig -= self.HP_ALPHA[t.seller]
        sig = max(-3.0, min(3.0, sig))

        fair = 0.65 * mid + 0.35 * ema + sig - self.HG_INV_SKEW * pos

        orders: List[Order] = []
        cur_pos = pos

        # Take crossed asks
        for ask in sorted(depth.sell_orders.keys()):
            if ask <= fair - self.HG_TAKE_EDGE:
                avail = -depth.sell_orders[ask]
                qty = max(0, min(avail, self.HG_LIMIT - cur_pos))
                if qty > 0:
                    orders.append(Order(self.HG_SYM, ask, qty))
                    cur_pos += qty
        # Hit expensive bids
        for bid in sorted(depth.buy_orders.keys(), reverse=True):
            if bid >= fair + self.HG_TAKE_EDGE:
                avail = depth.buy_orders[bid]
                qty = max(0, min(avail, self.HG_LIMIT + cur_pos))
                if qty > 0:
                    orders.append(Order(self.HG_SYM, bid, -qty))
                    cur_pos -= qty

        # Passive quotes
        buy_px = min(bb + 1, int(math.floor(fair - self.HG_QUOTE_EDGE)))
        sell_px = max(ba - 1, int(math.ceil(fair + self.HG_QUOTE_EDGE)))
        if buy_px < ba:
            qty = max(0, min(self.HG_QUOTE_SIZE, self.HG_LIMIT - cur_pos))
            if qty > 0:
                orders.append(Order(self.HG_SYM, buy_px, qty))
        if sell_px > bb:
            qty = max(0, min(self.HG_QUOTE_SIZE, self.HG_LIMIT + cur_pos))
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
        if ts < self.SWING_ENTRY_START:
            pass
        elif ts < self.SWING_ENTRY_END:
            room = max(0, self.VE_LIMIT - pos)
            if room > 0:
                size = min(self.VE_SWING_QUOTE_SIZE, room)
                orders.append(Order(self.VE_SYM, bb + 1, +size))
        elif ts < self.SWING_EXIT_START:
            pass
        elif ts < self.SWING_EXIT_FORCE:
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

        orders: List[Order] = []
        if ts < self.SWING_ENTRY_START:
            pass
        elif ts < self.SWING_ENTRY_END:
            room = max(0, self.VEV_LIMIT - pos)
            if room > 0:
                size = min(self.VEV_SWING_QUOTE_SIZE, room)
                orders.append(Order(sym, bb + 1, +size))
        elif ts < self.SWING_EXIT_START:
            pass
        elif ts < self.SWING_EXIT_FORCE:
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
