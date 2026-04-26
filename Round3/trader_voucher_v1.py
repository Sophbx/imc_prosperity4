"""
Voucher Book Trader v1 — smile-anchored market making for VEV options.

Strategy: fit a parabola IV(moneyness) every tick from the 4 inner strikes,
compute Black-Scholes fair price per traded strike, post bids/asks clamped
so we never bid above fair or ask below fair. Maker only. No taking. No
hedging with VELVETFRUIT. See Round3/Voucher/DESIGN.md.
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

        # Underlying mid
        u_depth = state.order_depths.get(UNDERLYING)
        if u_depth is None or not u_depth.buy_orders or not u_depth.sell_orders:
            return result, 0, ""
        u_mid = (max(u_depth.buy_orders) + min(u_depth.sell_orders)) / 2.0

        T = tte_years(state.timestamp // TIMESTAMPS_PER_DAY, state.timestamp % TIMESTAMPS_PER_DAY)
        if T <= 0:
            return result, 0, ""

        # Smile fit
        smile = self._fit_smile(state, u_mid, T)
        if smile is None:
            return result, 0, ""

        # Diagnostic only this task: print fit coefs once per ~10k ticks.
        if state.timestamp % 100_000 == 0:
            a, b, c = smile
            print(f"ts={state.timestamp} u_mid={u_mid:.2f} T={T:.5f} smile=(a={a:.4f}, b={b:.4f}, c={c:.4f})")

        return result, 0, ""

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
