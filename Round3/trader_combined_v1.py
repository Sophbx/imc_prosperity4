"""
Combined Round 3 trader.

Components:
  1. HYDROGEL_PACK         passive MM (50/60/120 bands)
  2. VELVETFRUIT_EXTRACT   passive MM (30/40/80 bands, scaled to ~5-tick spread)
  3. VEV_4000              passive ITM MM (75/90/180 bands, 300 limit)
  4. VEV_5000-5400         smile-anchored MM + cautious taker (edge=2) on fit strikes

Skipped:
  - VEV_4500   spread is wide but ZERO bot trade volume on day 0 (no liquidity)
  - VEV_5500-VEV_6500   spread too tight (<=1) to make money
"""
from math import erf, exp, log, pi, sqrt

from datamodel import Order, OrderDepth, TradingState


# =========================================================================
# Black-Scholes math (inline, no scipy/numpy)
# =========================================================================

SQRT_2 = sqrt(2.0)
SQRT_2PI = sqrt(2.0 * pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / SQRT_2))


def _norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / SQRT_2PI


def _d1(S: float, K: float, T: float, sigma: float) -> float:
    return (log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt(T))


def bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = _d1(S, K, T, sigma)
    d2 = d1 - sigma * sqrt(T)
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)


def bs_call_vega(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    return S * _norm_pdf(_d1(S, K, T, sigma)) * sqrt(T)


def implied_vol_call(price: float, S: float, K: float, T: float,
                     sigma_init: float = 0.25,
                     max_iter: int = 30, tol: float = 1e-6) -> float:
    if T <= 0 or price <= 0 or S <= 0 or K <= 0:
        return float("nan")
    intrinsic = max(S - K, 0.0)
    if price < intrinsic - tol:
        return float("nan")
    sigma = sigma_init
    for _ in range(max_iter):
        f = bs_call_price(S, K, T, sigma) - price
        if abs(f) < tol:
            return sigma
        v = bs_call_vega(S, K, T, sigma)
        if v < 1e-10:
            return float("nan")
        sigma = sigma - f / v
        if sigma <= 1e-4 or sigma > 5.0:
            return float("nan")
    return float("nan")


def _solve_3x3(M, y):
    a, b, c = M[0]
    d, e, f = M[1]
    g, h, i = M[2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-12:
        return None
    y0, y1, y2 = y
    det_a = y0 * (e * i - f * h) - b * (y1 * i - f * y2) + c * (y1 * h - e * y2)
    det_b = a * (y1 * i - f * y2) - y0 * (d * i - f * g) + c * (d * y2 - y1 * g)
    det_c = a * (e * y2 - y1 * h) - b * (d * y2 - y1 * g) + y0 * (d * h - e * g)
    return det_a / det, det_b / det, det_c / det


def fit_parabola(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    s0 = float(n)
    s1 = sum(xs)
    s2 = sum(x * x for x in xs)
    s3 = sum(x * x * x for x in xs)
    s4 = sum(x * x * x * x for x in xs)
    t0 = sum(ys)
    t1 = sum(x * y for x, y in zip(xs, ys))
    t2 = sum(x * x * y for x, y in zip(xs, ys))
    M = [[s4, s3, s2], [s3, s2, s1], [s2, s1, s0]]
    return _solve_3x3(M, [t2, t1, t0])


# =========================================================================
# Constants
# =========================================================================

HYDROGEL = "HYDROGEL_PACK"
VELVET = "VELVETFRUIT_EXTRACT"
VOUCHER_STRIKES_SMILE = [5000, 5100, 5200, 5300, 5400]    # quoted with smile fit
VOUCHER_STRIKES_FIT = [5000, 5100, 5200, 5300]            # used to fit the smile
VOUCHER_STRIKES_ITM = [4000]                               # delta-1 ITM, no smile

ALL_VOUCHERS_SMILE = [f"VEV_{K}" for K in VOUCHER_STRIKES_SMILE]
ALL_VOUCHERS_ITM = [f"VEV_{K}" for K in VOUCHER_STRIKES_ITM]

# Position limits
LIMIT_HYDROGEL = 200
LIMIT_VELVET = 200
LIMIT_VOUCHER = 300

# Smile / option time-to-expiry
TIMESTAMPS_PER_DAY = 1_000_000
YEAR_DAYS = 365
TTE_DAYS_AT_DAY = {0: 8, 1: 7, 2: 6, 3: 5}

# Voucher smile-MM constants
VOUCHER_MAX_POSITION = 250
VOUCHER_QUOTE_SIZE = 30
VOUCHER_SOFT_INVENTORY_LIMIT = 80
VOUCHER_MIN_QUOTE_SIZE = 4
VOUCHER_FLATTEN_WINDOW_START_TS = 980_000

# Taker tuning (apply only on smile-fit strikes — extrapolated VEV_5400 has bias)
TAKER_EDGE = 2
TAKER_SIZE = 10
TAKER_STRIKES = set(VOUCHER_STRIKES_FIT)


def tte_years(day: int, ts_in_day: int) -> float:
    base = TTE_DAYS_AT_DAY.get(day)
    if base is None:
        return 0.0
    return max(0.0, (base - ts_in_day / TIMESTAMPS_PER_DAY) / YEAR_DAYS)


# =========================================================================
# Trader
# =========================================================================

class Trader:
    HYDROGEL_CONFIG = {
        "improve_ticks": 1,
        "min_quote_spread": 6,
        "quote_size": 10,
        "min_quote_size": 2,
        "skew_inventory_scale": 50,
        "unwind_inventory_limit": 60,
        "hard_inventory_limit": 120,
        "skew_imbalance_threshold": 0.10,
        "take_distance": 4,
        "flatten_window_start_ts": 980_000,
        "flatten_quote_offset": 2,
    }

    VELVET_CONFIG = {
        "improve_ticks": 1,
        "min_quote_spread": 2,
        "quote_size": 8,
        "min_quote_size": 2,
        "skew_inventory_scale": 30,
        "unwind_inventory_limit": 40,
        "hard_inventory_limit": 80,
        "skew_imbalance_threshold": 0.10,
        "take_distance": 2,
        "flatten_window_start_ts": 980_000,
        "flatten_quote_offset": 1,
    }

    ITM_CONFIG = {
        "improve_ticks": 1,
        "min_quote_spread": 6,
        "quote_size": 10,
        "min_quote_size": 2,
        "skew_inventory_scale": 75,
        "unwind_inventory_limit": 90,
        "hard_inventory_limit": 180,
        "skew_imbalance_threshold": 0.10,
        "take_distance": 4,
        "flatten_window_start_ts": 980_000,
        "flatten_quote_offset": 2,
    }

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        ts = int(state.timestamp)

        # 1. HYDROGEL_PACK
        h_depth = state.order_depths.get(HYDROGEL)
        if h_depth is not None:
            pos = int(state.position.get(HYDROGEL, 0))
            orders = self._mm_delta1(HYDROGEL, h_depth, pos, ts, LIMIT_HYDROGEL,
                                     self.HYDROGEL_CONFIG)
            if orders:
                result[HYDROGEL] = orders

        # 2. VELVETFRUIT_EXTRACT
        v_depth = state.order_depths.get(VELVET)
        if v_depth is not None:
            pos = int(state.position.get(VELVET, 0))
            orders = self._mm_delta1(VELVET, v_depth, pos, ts, LIMIT_VELVET,
                                     self.VELVET_CONFIG)
            if orders:
                result[VELVET] = orders

        # 3. VEV_4000 deep-ITM MM
        for sym in ALL_VOUCHERS_ITM:
            depth = state.order_depths.get(sym)
            if depth is None:
                continue
            pos = int(state.position.get(sym, 0))
            orders = self._mm_delta1(sym, depth, pos, ts, LIMIT_VOUCHER,
                                     self.ITM_CONFIG)
            if orders:
                result[sym] = orders

        # 4. VEV_5000-5400 smile-anchored MM
        u_depth = state.order_depths.get(VELVET)
        if u_depth is not None and u_depth.buy_orders and u_depth.sell_orders:
            u_mid = (max(u_depth.buy_orders) + min(u_depth.sell_orders)) / 2.0
            day = ts // TIMESTAMPS_PER_DAY
            ts_in_day = ts % TIMESTAMPS_PER_DAY
            T = tte_years(day, ts_in_day)
            if T > 0:
                smile = self._fit_smile(state, u_mid, T)
                for K in VOUCHER_STRIKES_SMILE:
                    sym = f"VEV_{K}"
                    depth = state.order_depths.get(sym)
                    if depth is None:
                        continue
                    bb, ba = self._best_prices(depth)
                    if bb is None or ba is None:
                        continue
                    pos = int(state.position.get(sym, 0))
                    orders = self._quote_smile_voucher(sym, K, depth, pos, u_mid, T,
                                                      smile, ts_in_day)
                    if orders:
                        result[sym] = orders

        return result, 0, ""

    # ----- HYDROGEL / VELVETFRUIT / ITM MM (delta-1, smile-free) -----

    def _mm_delta1(self, sym, depth, position, timestamp, limit, cfg):
        bb, ba = self._best_prices(depth)
        if bb is None or ba is None:
            return []
        spread = ba - bb
        touch_mid = (bb + ba) / 2.0
        bid_room = max(0, limit - position)
        ask_room = max(0, limit + position)

        if position > cfg["hard_inventory_limit"]:
            return self._take_to_neutralize(sym, depth, position, touch_mid, cfg, "sell")
        if position < -cfg["hard_inventory_limit"]:
            return self._take_to_neutralize(sym, depth, position, touch_mid, cfg, "buy")
        if spread < cfg["min_quote_spread"]:
            return []

        bid_price, ask_price = self._delta1_quote_prices(
            depth, position, bb, ba, touch_mid, timestamp, cfg,
        )
        if bid_price >= ba:
            bid_price = ba - 1
        if ask_price <= bb:
            ask_price = bb + 1

        bid_qty, ask_qty = self._delta1_sized_quotes(position, bid_room, ask_room, cfg)

        orders = []
        if bid_qty > 0:
            orders.append(Order(sym, int(bid_price), int(bid_qty)))
        if ask_qty > 0:
            orders.append(Order(sym, int(ask_price), -int(ask_qty)))
        return orders

    def _delta1_quote_prices(self, depth, position, bb, ba, touch_mid, ts, cfg):
        if ts > cfg["flatten_window_start_ts"]:
            return (
                int(round(touch_mid - cfg["flatten_quote_offset"])),
                int(round(touch_mid + cfg["flatten_quote_offset"])),
            )
        if abs(position) > cfg["unwind_inventory_limit"]:
            if position > 0:
                return bb + cfg["improve_ticks"], int(round(touch_mid - 0.5))
            else:
                return int(round(touch_mid + 0.5)), ba - cfg["improve_ticks"]
        di = self._depth_imbalance(depth)
        bid_improve = cfg["improve_ticks"]
        ask_improve = cfg["improve_ticks"]
        if di >= cfg["skew_imbalance_threshold"]:
            bid_improve = 0
        elif di <= -cfg["skew_imbalance_threshold"]:
            ask_improve = 0
        return bb + bid_improve, ba - ask_improve

    @staticmethod
    def _delta1_sized_quotes(position, bid_room, ask_room, cfg):
        scale = cfg["skew_inventory_scale"]
        pressure = max(-2.0, min(2.0, position / scale))
        bid_factor = max(0.0, 1.0 - max(0.0, pressure))
        ask_factor = max(0.0, 1.0 + min(0.0, pressure))
        base = cfg["quote_size"]
        bid_qty = max(cfg["min_quote_size"], int(round(base * bid_factor)))
        ask_qty = max(cfg["min_quote_size"], int(round(base * ask_factor)))
        return min(bid_qty, bid_room), min(ask_qty, ask_room)

    def _take_to_neutralize(self, sym, depth, position, touch_mid, cfg, side):
        target_qty = abs(position) - cfg["unwind_inventory_limit"]
        orders = []
        if side == "sell":
            for price in sorted(depth.buy_orders.keys(), reverse=True):
                if price < touch_mid - cfg["take_distance"]:
                    break
                avail = int(depth.buy_orders[price])
                qty = min(target_qty, max(0, avail))
                if qty <= 0:
                    continue
                orders.append(Order(sym, int(price), -int(qty)))
                target_qty -= qty
                if target_qty <= 0:
                    break
        else:
            for price in sorted(depth.sell_orders.keys()):
                if price > touch_mid + cfg["take_distance"]:
                    break
                avail = abs(int(depth.sell_orders[price]))
                qty = min(target_qty, max(0, avail))
                if qty <= 0:
                    continue
                orders.append(Order(sym, int(price), int(qty)))
                target_qty -= qty
                if target_qty <= 0:
                    break
        return orders

    # ----- Voucher smile MM (5000-5400) -----

    def _quote_smile_voucher(self, sym, K, depth, position, u_mid, T, smile, ts_in_day):
        bb, ba = self._best_prices(depth)
        bid_room = max(0, VOUCHER_MAX_POSITION - position)
        ask_room = max(0, VOUCHER_MAX_POSITION + position)

        fair_px = None
        if smile is not None:
            a, b, c = smile
            m = u_mid / K
            fair_iv = a * m * m + b * m + c
            if fair_iv > 0:
                fair_px = bs_call_price(u_mid, K, T, fair_iv)

        # Taker leg (only on fit strikes).
        taker_orders = []
        if fair_px is not None and K in TAKER_STRIKES:
            if ba <= fair_px - TAKER_EDGE and ask_room > 0:
                ask_size = depth.sell_orders.get(ba, 0)
                avail = abs(int(ask_size))
                qty = min(TAKER_SIZE, avail, bid_room)
                if qty > 0:
                    taker_orders.append(Order(sym, int(ba), int(qty)))
            if bb >= fair_px + TAKER_EDGE and bid_room > 0:
                bid_size = depth.buy_orders.get(bb, 0)
                avail = int(bid_size)
                qty = min(TAKER_SIZE, avail, ask_room)
                if qty > 0:
                    taker_orders.append(Order(sym, int(bb), -int(qty)))

        # Pricing
        if ts_in_day > VOUCHER_FLATTEN_WINDOW_START_TS:
            if fair_px is not None:
                buy_price = int(fair_px)
                sell_price = int(round(fair_px + 0.5))
            else:
                touch_mid = (bb + ba) / 2.0
                buy_price = int(touch_mid)
                sell_price = int(round(touch_mid + 0.5))
        else:
            buy_price = bb + 1
            sell_price = ba - 1
            if fair_px is not None:
                buy_price = min(buy_price, int(fair_px))
                sell_price = max(sell_price, int(round(fair_px + 0.5)))

        if buy_price >= ba:
            buy_price = ba - 1
        if sell_price <= bb:
            sell_price = bb + 1

        bid_qty, ask_qty = self._voucher_sized_quotes(position, bid_room, ask_room)

        orders = list(taker_orders)
        if bid_qty > 0:
            orders.append(Order(sym, int(buy_price), int(bid_qty)))
        if ask_qty > 0:
            orders.append(Order(sym, int(sell_price), -int(ask_qty)))
        return orders

    @staticmethod
    def _voucher_sized_quotes(position, bid_room, ask_room):
        if position > VOUCHER_SOFT_INVENTORY_LIMIT:
            shrink = (position - VOUCHER_SOFT_INVENTORY_LIMIT) / (VOUCHER_MAX_POSITION - VOUCHER_SOFT_INVENTORY_LIMIT)
            shrink = min(1.0, max(0.0, shrink))
            bid_qty = max(VOUCHER_MIN_QUOTE_SIZE, int(round(VOUCHER_QUOTE_SIZE * (1.0 - shrink))))
            ask_qty = VOUCHER_QUOTE_SIZE
        elif position < -VOUCHER_SOFT_INVENTORY_LIMIT:
            shrink = (-position - VOUCHER_SOFT_INVENTORY_LIMIT) / (VOUCHER_MAX_POSITION - VOUCHER_SOFT_INVENTORY_LIMIT)
            shrink = min(1.0, max(0.0, shrink))
            bid_qty = VOUCHER_QUOTE_SIZE
            ask_qty = max(VOUCHER_MIN_QUOTE_SIZE, int(round(VOUCHER_QUOTE_SIZE * (1.0 - shrink))))
        else:
            bid_qty = VOUCHER_QUOTE_SIZE
            ask_qty = VOUCHER_QUOTE_SIZE
        return min(bid_qty, bid_room), min(ask_qty, ask_room)

    def _fit_smile(self, state, u_mid, T):
        ms, ivs = [], []
        for K in VOUCHER_STRIKES_FIT:
            depth = state.order_depths.get(f"VEV_{K}")
            if depth is None or not depth.buy_orders or not depth.sell_orders:
                continue
            mid = (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0
            iv = implied_vol_call(mid, u_mid, K, T)
            if iv != iv:
                continue
            ms.append(u_mid / K)
            ivs.append(iv)
        if len(ms) < 3:
            return None
        return fit_parabola(ms, ivs)

    # ----- Helpers -----

    @staticmethod
    def _best_prices(depth):
        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        return best_bid, best_ask

    @staticmethod
    def _depth_imbalance(depth):
        bids = sum(depth.buy_orders.values()) if depth.buy_orders else 0
        asks = sum(abs(v) for v in depth.sell_orders.values()) if depth.sell_orders else 0
        total = bids + asks
        if total == 0:
            return 0.0
        return (bids - asks) / total
