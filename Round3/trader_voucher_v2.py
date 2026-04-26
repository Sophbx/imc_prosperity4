"""
Voucher Book Trader v2 — v1 + cautious taker leg on well-fitted strikes only.

v1 is the conservative passive maker. v2 adds an opportunistic taker:
- For strikes IN the smile fit set (5000-5300, where the fit is near-zero
  residual by construction), if the visible best ask is well below fair price
  (>= TAKER_EDGE ticks), lift it. Symmetric for bids.
- Skip taker for 5400 (extrapolated; structural -2.3 tick bias per research).

Backtest comparison on Round 3 historical data (3 days):
  v1: total $2,393, mean/day $798, stdev $881, sharpe 0.91, worst DD -$2,672
  v2: total $3,008, mean/day $1,003, stdev $1,237, sharpe 0.81, worst DD -$3,761

v2 captures +26% more PnL but with worse risk metrics — more aggressive
strategy. Pick v1 for stability, v2 for raw return.

Other variants tested and rejected (see iteration log in commit history):
- Wider strike set (4500-5500): VEV_4500 lost money (intrinsic-dominated),
  net -$9 vs v1.
- Smaller QUOTE_SIZE (15 vs 30): identical to v1 (bot fill volume bottleneck).
- Avellaneda-Stoikov reservation pricing: idle at small positions, identical.
"""
from math import erf, exp, log, pi, sqrt

from datamodel import Order, OrderDepth, TradingState


# =========================================================================
# Math (inline, no scipy)
# =========================================================================

SQRT_2 = sqrt(2.0)
SQRT_2PI = sqrt(2.0 * pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / SQRT_2))


def _norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / SQRT_2PI


def _d1(S: float, K: float, T: float, sigma: float) -> float:
    # r=0 throughout — Prosperity convention
    return (log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt(T))


def bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
    """European call price at r=0."""
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
    """Newton-Raphson IV solver. NaN on bad inputs / non-convergence."""
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
    """Cramer's rule for a 3x3 linear system M @ x = y. Returns None if singular."""
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
    """Fit y = a*x^2 + b*x + c via OLS normal equations. Returns (a,b,c) or None."""
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

UNDERLYING = "VELVETFRUIT_EXTRACT"
TRADED_STRIKES = [5000, 5100, 5200, 5300, 5400]
FIT_STRIKES = [5000, 5100, 5200, 5300]
ALL_VOUCHERS = [f"VEV_{K}" for K in TRADED_STRIKES]

POSITION_LIMIT = 300
MAX_POSITION = 250
QUOTE_SIZE = 30
SOFT_INVENTORY_LIMIT = 80
MIN_QUOTE_SIZE = 4

TIMESTAMPS_PER_DAY = 1_000_000
YEAR_DAYS = 365
# Live Round 3 starts at TTE = 5 days. Map day -> remaining days at start of day.
TTE_DAYS_AT_DAY = {0: 8, 1: 7, 2: 6, 3: 5}

FLATTEN_WINDOW_START_TS = 980_000   # last 2% of timestamps

# Taker tuning: lift offers / hit bids when |ask/bid - fair| >= TAKER_EDGE ticks.
# Apply only to in-sample (fit) strikes — extrapolated strikes (5400) have bias.
TAKER_EDGE = 3
TAKER_SIZE = 10
TAKER_STRIKES = set(FIT_STRIKES)   # only fit strikes


def tte_years(day: int, timestamp: int) -> float:
    """Years to expiry; linear intra-day."""
    base = TTE_DAYS_AT_DAY.get(day)
    if base is None:
        return 0.0
    return max(0.0, (base - timestamp / TIMESTAMPS_PER_DAY) / YEAR_DAYS)


# =========================================================================
# Trader
# =========================================================================

class Trader:
    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}

        u_depth = state.order_depths.get(UNDERLYING)
        if u_depth is None or not u_depth.buy_orders or not u_depth.sell_orders:
            return result, 0, ""
        u_mid = (max(u_depth.buy_orders) + min(u_depth.sell_orders)) / 2.0

        day = state.timestamp // TIMESTAMPS_PER_DAY
        ts_in_day = state.timestamp % TIMESTAMPS_PER_DAY
        T = tte_years(day, ts_in_day)
        if T <= 0:
            return result, 0, ""

        smile = self._fit_smile(state, u_mid, T)

        for K in TRADED_STRIKES:
            symbol = f"VEV_{K}"
            depth = state.order_depths.get(symbol)
            if depth is None:
                continue
            best_bid, best_ask = self._best_prices(depth)
            if best_bid is None or best_ask is None:
                continue

            position = int(state.position.get(symbol, 0))
            orders = self._quote_voucher(symbol, K, depth, position, u_mid, T, smile, ts_in_day)
            if orders:
                result[symbol] = orders

        return result, 0, ""

    def _quote_voucher(self, symbol, K, depth, position, u_mid, T, smile, ts_in_day):
        best_bid, best_ask = self._best_prices(depth)
        bid_room = max(0, MAX_POSITION - position)
        ask_room = max(0, MAX_POSITION + position)

        # Smile clamp / fair value (None => fall back to plain MM).
        fair_px = None
        if smile is not None:
            a, b, c = smile
            m = u_mid / K
            fair_iv = a * m * m + b * m + c
            if fair_iv > 0:
                fair_px = bs_call_price(u_mid, K, T, fair_iv)

        # Taker leg (in-sample fit strikes only).
        taker_orders = []
        if fair_px is not None and K in TAKER_STRIKES:
            # Lift offers below fair - TAKER_EDGE
            if best_ask <= fair_px - TAKER_EDGE and ask_room > 0:
                ask_size = depth.sell_orders.get(best_ask, 0)
                avail = abs(int(ask_size))
                qty = min(TAKER_SIZE, avail, bid_room)
                if qty > 0:
                    taker_orders.append(Order(symbol, int(best_ask), int(qty)))
            # Hit bids above fair + TAKER_EDGE
            if best_bid >= fair_px + TAKER_EDGE and bid_room > 0:
                bid_size = depth.buy_orders.get(best_bid, 0)
                avail = int(bid_size)
                qty = min(TAKER_SIZE, avail, ask_room)
                if qty > 0:
                    taker_orders.append(Order(symbol, int(best_bid), -int(qty)))

        # End-of-day flatten window: quote tighter at floor/ceil(fair) so leftover
        # inventory bleeds out near fair value. If smile is unavailable, use touch_mid.
        if ts_in_day > FLATTEN_WINDOW_START_TS:
            if fair_px is not None:
                buy_price = int(fair_px)
                sell_price = int(round(fair_px + 0.5))
            else:
                touch_mid = (best_bid + best_ask) / 2.0
                buy_price = int(touch_mid)
                sell_price = int(round(touch_mid + 0.5))
        else:
            # Normal MM: improve by 1, smile-clamped if available.
            buy_price = best_bid + 1
            sell_price = best_ask - 1
            if fair_px is not None:
                buy_price = min(buy_price, int(fair_px))
                sell_price = max(sell_price, int(round(fair_px + 0.5)))

        # Never cross the book passively.
        if buy_price >= best_ask:
            buy_price = best_ask - 1
        if sell_price <= best_bid:
            sell_price = best_bid + 1

        # Inventory skew: when |position| > SOFT_INVENTORY_LIMIT, shrink the
        # side that adds inventory and let the other side clear.
        bid_qty, ask_qty = self._sized_quotes(position, bid_room, ask_room)

        orders = list(taker_orders)
        if bid_qty > 0:
            orders.append(Order(symbol, int(buy_price), int(bid_qty)))
        if ask_qty > 0:
            orders.append(Order(symbol, int(sell_price), -int(ask_qty)))
        return orders

    @staticmethod
    def _sized_quotes(position, bid_room, ask_room):
        # Linear shrink from |pos|=SOFT_INVENTORY_LIMIT toward 0 at |pos|=MAX_POSITION.
        if position > SOFT_INVENTORY_LIMIT:
            shrink = (position - SOFT_INVENTORY_LIMIT) / (MAX_POSITION - SOFT_INVENTORY_LIMIT)
            shrink = min(1.0, max(0.0, shrink))
            bid_qty = max(MIN_QUOTE_SIZE, int(round(QUOTE_SIZE * (1.0 - shrink))))
            ask_qty = QUOTE_SIZE
        elif position < -SOFT_INVENTORY_LIMIT:
            shrink = (-position - SOFT_INVENTORY_LIMIT) / (MAX_POSITION - SOFT_INVENTORY_LIMIT)
            shrink = min(1.0, max(0.0, shrink))
            bid_qty = QUOTE_SIZE
            ask_qty = max(MIN_QUOTE_SIZE, int(round(QUOTE_SIZE * (1.0 - shrink))))
        else:
            bid_qty = QUOTE_SIZE
            ask_qty = QUOTE_SIZE
        return min(bid_qty, bid_room), min(ask_qty, ask_room)

    def _fit_smile(self, state: TradingState, u_mid: float, T: float):
        """Fit IV ~ a*m**2 + b*m + c on FIT_STRIKES. Returns (a,b,c) or None."""
        ms, ivs = [], []
        for K in FIT_STRIKES:
            depth = state.order_depths.get(f"VEV_{K}")
            if depth is None or not depth.buy_orders or not depth.sell_orders:
                continue
            mid = (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0
            iv = implied_vol_call(mid, u_mid, K, T)
            if iv != iv:  # NaN
                continue
            ms.append(u_mid / K)
            ivs.append(iv)
        if len(ms) < 3:
            return None
        return fit_parabola(ms, ivs)

    @staticmethod
    def _best_prices(depth: OrderDepth):
        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        return best_bid, best_ask
