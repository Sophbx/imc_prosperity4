"""Shared helpers for Round 3 analysis notebooks.

See DESIGN.md for the full specification.
"""
from __future__ import annotations

import math
import os
import re
from math import log, sqrt, exp, pi
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import brentq


# ---------- data loading ----------

def extract_day_from_filename(path: str) -> int:
    """Pull the integer day N out of a filename like 'prices_round_3_day_-1.csv'.

    Returns 0 when no 'day_<int>' segment is present.
    """
    m = re.search(r"day_(-?\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def load_prices(data_dir: str, price_files: list[str]) -> pd.DataFrame:
    """Load and concatenate Prosperity price CSVs.

    CSVs are ';'-separated. Each row is augmented with an integer 'day' column
    parsed from the filename if the column is missing.
    """
    frames = []
    for fn in price_files:
        df = pd.read_csv(os.path.join(data_dir, fn), sep=";")
        if "day" not in df.columns:
            df["day"] = extract_day_from_filename(fn)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_trades(data_dir: str, trade_files: list[str]) -> pd.DataFrame:
    """Load and concatenate Prosperity trade CSVs.

    Renames the 'symbol' column to 'product' for consistency with prices, and
    tags each row with an integer 'day' from the filename.
    """
    frames = []
    for fn in trade_files:
        df = pd.read_csv(os.path.join(data_dir, fn), sep=";")
        df = df.rename(columns={"symbol": "product"})
        df["day"] = extract_day_from_filename(fn)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------- tiny math utilities ----------

def safe_corr(x, y) -> float:
    """Pearson correlation that returns NaN on zero-variance or empty input.

    Aligns x and y, drops any row where either is NaN, then returns NaN if
    fewer than 2 rows remain or either side is constant.
    """
    z = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(z) < 2:
        return float("nan")
    if z["x"].std(ddof=0) == 0 or z["y"].std(ddof=0) == 0:
        return float("nan")
    return float(z["x"].corr(z["y"]))


def weighted_avg(values: Iterable[float], weights: Iterable[float]) -> float:
    """Volume-weighted average that returns NaN when all weights are zero/NaN."""
    values = np.asarray(list(values), dtype=float)
    weights = np.asarray(list(weights), dtype=float)
    mask = ~(np.isnan(values) | np.isnan(weights))
    values = values[mask]
    weights = weights[mask]
    total = weights.sum()
    if total == 0:
        return float("nan")
    return float((values * weights).sum() / total)


# ---------- Black-Scholes (r=0 by default for Prosperity) ----------


def _d1(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    return (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """European call price. T in years, r continuously compounded, sigma annualized."""
    if T <= 0 or sigma <= 0:
        return max(S - K * exp(-r * T), 0.0)
    d1 = _d1(S, K, T, sigma, r)
    d2 = d1 - sigma * sqrt(T)
    return S * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)


def bs_call_delta(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else (0.5 if S == K else 0.0)
    return _norm_cdf(_d1(S, K, T, sigma, r))


def bs_call_vega(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Vega per 1.00 change in sigma (i.e. in *absolute* units, not per 1%)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return S * _norm_pdf(_d1(S, K, T, sigma, r)) * sqrt(T)


def bs_call_gamma(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    return _norm_pdf(_d1(S, K, T, sigma, r)) / (S * sigma * sqrt(T))


def implied_vol_call(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.0,
    lo: float = 1e-4,
    hi: float = 5.0,
    tol: float = 1e-6,
) -> float:
    """Solve BS implied vol for a European call via Brent's method.

    Returns NaN when:
    - T <= 0 (expired)
    - price is below intrinsic value max(S - K*exp(-rT), 0)
    - price sits above the value implied by sigma = hi (we don't extrapolate)
    """
    if T <= 0 or price <= 0 or S <= 0 or K <= 0:
        return float("nan")
    intrinsic = max(S - K * exp(-r * T), 0.0)
    if price < intrinsic - tol:
        return float("nan")
    f_lo = bs_call_price(S, K, T, lo, r) - price
    f_hi = bs_call_price(S, K, T, hi, r) - price
    if f_lo > 0 or f_hi < 0:
        return float("nan")
    try:
        return float(brentq(
            lambda sigma: bs_call_price(S, K, T, sigma, r) - price,
            lo, hi, xtol=tol,
        ))
    except ValueError:
        return float("nan")


# ---------- Round 3 option conventions ----------

TIMESTAMPS_PER_DAY = 10_000
YEAR_DAYS = 365
TTE_DAYS_AT_DAY = {0: 8, 1: 7, 2: 6}  # per wiki


def tte_years(day: int, timestamp: int) -> float:
    """Time to expiry in years for a Round 3 voucher at (day, timestamp).

    Uses a linear intra-day schedule: TTE at start of day d is TTE_DAYS_AT_DAY[d] days,
    decreasing linearly to TTE_DAYS_AT_DAY[d] - 1 over TIMESTAMPS_PER_DAY ticks.
    """
    days_remaining = TTE_DAYS_AT_DAY[day] - timestamp / TIMESTAMPS_PER_DAY
    return days_remaining / YEAR_DAYS
