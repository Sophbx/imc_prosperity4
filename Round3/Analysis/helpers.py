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

TIMESTAMPS_PER_DAY = 1_000_000  # actual data: timestamps run 0..999900 in steps of 100
YEAR_DAYS = 365
TTE_DAYS_AT_DAY = {0: 8, 1: 7, 2: 6}  # per wiki


def tte_years(day: int, timestamp: int) -> float:
    """Time to expiry in years for a Round 3 voucher at (day, timestamp).

    Uses a linear intra-day schedule: TTE at start of day d is TTE_DAYS_AT_DAY[d] days,
    decreasing linearly to TTE_DAYS_AT_DAY[d] - 1 over TIMESTAMPS_PER_DAY ticks.
    """
    days_remaining = TTE_DAYS_AT_DAY[day] - timestamp / TIMESTAMPS_PER_DAY
    return days_remaining / YEAR_DAYS


# ---------- book feature engineering (ported from Round1/analysis_v2.py) ----------

VISIBLE_LEVELS = 3
HORIZON = 10


def row_weighted_avg(row, price_cols, vol_cols):
    vals = [row[c] for c in price_cols]
    wts = [row[c] for c in vol_cols]
    return weighted_avg(vals, wts)


def row_boundary_bid(row, bid_price_cols):
    vals = [row[c] for c in bid_price_cols if pd.notna(row[c])]
    return np.min(vals) if len(vals) > 0 else np.nan


def row_boundary_ask(row, ask_price_cols):
    vals = [row[c] for c in ask_price_cols if pd.notna(row[c])]
    return np.max(vals) if len(vals) > 0 else np.nan


def row_size_wall_price_and_volume(row, price_cols, vol_cols, tie_break="frontier"):
    """
    Size-wall = price level with largest displayed volume.
    tie_break="frontier" means if equal sizes, choose the more aggressive / nearer level:
      - for bids: higher price
      - for asks: lower price
    Since columns are already ordered from frontier outward, keeping the first max does this.
    """
    best_idx = None
    best_vol = -np.inf

    for i, vcol in enumerate(vol_cols):
        vol = row[vcol]
        if pd.notna(vol) and vol > best_vol:
            best_vol = vol
            best_idx = i

    if best_idx is None:
        return pd.Series([np.nan, np.nan])

    return pd.Series([row[price_cols[best_idx]], row[vol_cols[best_idx]]])


def next_trade_direction_from_price_vs_mid(price, mid):
    if pd.isna(price) or pd.isna(mid):
        return np.nan
    if price > mid:
        return 1
    if price < mid:
        return -1
    return 0


def build_book_features(prices):
    df = prices.copy()

    bid_price_cols = [f"bid_price_{i}" for i in range(1, VISIBLE_LEVELS + 1)]
    bid_vol_cols   = [f"bid_volume_{i}" for i in range(1, VISIBLE_LEVELS + 1)]
    ask_price_cols = [f"ask_price_{i}" for i in range(1, VISIBLE_LEVELS + 1)]
    ask_vol_cols   = [f"ask_volume_{i}" for i in range(1, VISIBLE_LEVELS + 1)]

    # ---------------------------
    # Frontier = best prices
    # ---------------------------
    df["frontier_bid"] = df["bid_price_1"]   # highest bid
    df["frontier_ask"] = df["ask_price_1"]   # lowest ask
    df["touch_mid"] = (df["frontier_bid"] + df["frontier_ask"]) / 2.0
    df["touch_spread"] = df["frontier_ask"] - df["frontier_bid"]

    # Check original dataset mid_price
    df["dataset_mid_minus_touch_mid"] = df["mid_price"] - df["touch_mid"]

    # ---------------------------
    # Boundary = visible outer edge of current scanned book
    # ---------------------------
    df["boundary_bid"] = df.apply(lambda r: row_boundary_bid(r, bid_price_cols), axis=1)
    df["boundary_ask"] = df.apply(lambda r: row_boundary_ask(r, ask_price_cols), axis=1)
    df["boundary_mid"] = (df["boundary_bid"] + df["boundary_ask"]) / 2.0
    df["boundary_width"] = df["boundary_ask"] - df["boundary_bid"]

    # ---------------------------
    # Size-wall / max-size mode
    # ---------------------------
    df[["size_wall_bid", "size_wall_bid_vol"]] = df.apply(
        lambda r: row_size_wall_price_and_volume(r, bid_price_cols, bid_vol_cols), axis=1
    )
    df[["size_wall_ask", "size_wall_ask_vol"]] = df.apply(
        lambda r: row_size_wall_price_and_volume(r, ask_price_cols, ask_vol_cols), axis=1
    )
    df["size_wall_mid"] = (df["size_wall_bid"] + df["size_wall_ask"]) / 2.0
    df["size_wall_width"] = df["size_wall_ask"] - df["size_wall_bid"]

    # ---------------------------
    # VWAP centers
    # ---------------------------
    df["bid_vwap_center"] = df.apply(lambda r: row_weighted_avg(r, bid_price_cols, bid_vol_cols), axis=1)
    df["ask_vwap_center"] = df.apply(lambda r: row_weighted_avg(r, ask_price_cols, ask_vol_cols), axis=1)
    df["side_vwap_mid"] = (df["bid_vwap_center"] + df["ask_vwap_center"]) / 2.0

    all_price_cols = bid_price_cols + ask_price_cols
    all_vol_cols = bid_vol_cols + ask_vol_cols
    df["full_book_vwap_center"] = df.apply(lambda r: row_weighted_avg(r, all_price_cols, all_vol_cols), axis=1)

    # ---------------------------
    # Original dataset mid
    # ---------------------------
    # This is already present as df["mid_price"]
    # We keep it as another fair-value-like reference.
    #
    # Five fair value estimators we track:
    #   1) touch_mid
    #   2) boundary_mid
    #   3) size_wall_mid
    #   4) side_vwap_mid
    #   5) full_book_vwap_center
    # And compare them to the dataset mid_price.

    # ---------------------------
    # Depth / imbalance
    # ---------------------------
    df["frontier_bid_vol"] = df["bid_volume_1"]
    df["frontier_ask_vol"] = df["ask_volume_1"]

    df["frontier_imbalance"] = (
        (df["frontier_bid_vol"] - df["frontier_ask_vol"]) /
        (df["frontier_bid_vol"] + df["frontier_ask_vol"])
    )

    df["bid_depth_total"] = df[bid_vol_cols].sum(axis=1, min_count=1)
    df["ask_depth_total"] = df[ask_vol_cols].sum(axis=1, min_count=1)
    df["depth_imbalance"] = (
        (df["bid_depth_total"] - df["ask_depth_total"]) /
        (df["bid_depth_total"] + df["ask_depth_total"])
    )

    df["size_wall_vol_imbalance"] = (
        (df["size_wall_bid_vol"] - df["size_wall_ask_vol"]) /
        (df["size_wall_bid_vol"] + df["size_wall_ask_vol"])
    )

    # ---------------------------
    # Layer 2: relative dislocations / correspondence
    # ---------------------------
    fair_value_cols = [
        "touch_mid",
        "boundary_mid",
        "size_wall_mid",
        "side_vwap_mid",
        "full_book_vwap_center",
        "mid_price",
    ]

    for c in fair_value_cols:
        df[f"{c}_minus_touch_mid"] = df[c] - df["touch_mid"]

    df["boundary_minus_touch_width"] = df["boundary_width"] - df["touch_spread"]
    df["sizewall_minus_touch_width"] = df["size_wall_width"] - df["touch_spread"]
    df["side_vwap_skew"] = df["ask_vwap_center"] - df["bid_vwap_center"]

    # ---------------------------
    # Future price targets
    # ---------------------------
    grp = df.groupby(["product", "day"])
    df[f"fwd_touch_mid_change_{HORIZON}"] = grp["touch_mid"].shift(-HORIZON) - df["touch_mid"]
    df[f"fwd_dataset_mid_change_{HORIZON}"] = grp["mid_price"].shift(-HORIZON) - df["mid_price"]

    df[f"fwd_direction_{HORIZON}"] = np.where(
        df[f"fwd_touch_mid_change_{HORIZON}"] > 0, 1,
        np.where(df[f"fwd_touch_mid_change_{HORIZON}"] < 0, -1, 0)
    )

    # Future max/min over next HORIZON rows for break/bounce style tests
    df[f"future_max_touch_mid_{HORIZON}"] = grp["touch_mid"].transform(
        lambda s: s.shift(-1).rolling(HORIZON, min_periods=1).max()
    )
    df[f"future_min_touch_mid_{HORIZON}"] = grp["touch_mid"].transform(
        lambda s: s.shift(-1).rolling(HORIZON, min_periods=1).min()
    )

    return df


# ---------- trade feature engineering (ported from Round1/analysis_v2.py) ----------

def build_trade_features(trades, book_df):
    # Merge book touch mid into trades to infer trade direction proxy
    trades = trades.copy()

    mini_book = book_df[["product", "day", "timestamp", "touch_mid", "frontier_bid", "frontier_ask",
                         "size_wall_bid", "size_wall_ask", "frontier_imbalance", "depth_imbalance",
                         "size_wall_vol_imbalance"]].copy()

    trades = trades.rename(columns={"symbol": "product"})
    trades = trades.merge(mini_book, on=["product", "day", "timestamp"], how="left")

    # Per-trade direction proxy by trade price vs touch mid
    trades["trade_sign_proxy"] = trades.apply(
        lambda r: next_trade_direction_from_price_vs_mid(r["price"], r["touch_mid"]), axis=1
    )

    # Trade location vs book: A
    def classify_trade_location(row):
        if pd.isna(row["price"]):
            return "no_trade"
        p = row["price"]
        bid1 = row["frontier_bid"]
        ask1 = row["frontier_ask"]
        bw = row["size_wall_bid"]
        aw = row["size_wall_ask"]

        if pd.notna(ask1) and p >= ask1:
            return "at_or_above_ask_touch"
        if pd.notna(bid1) and p <= bid1:
            return "at_or_below_bid_touch"
        if pd.notna(bid1) and pd.notna(ask1) and bid1 < p < ask1:
            return "inside_touch"

        # extra rough wall markers
        if pd.notna(aw) and p >= aw:
            return "near_or_above_ask_wall"
        if pd.notna(bw) and p <= bw:
            return "near_or_below_bid_wall"

        return "other"

    trades["trade_location"] = trades.apply(classify_trade_location, axis=1)

    grouped = trades.groupby(["product", "day", "timestamp"], dropna=False)

    agg = grouped.apply(
        lambda g: pd.Series({
            "trade_count": len(g),
            "trade_volume": g["quantity"].sum(),
            "trade_vwap": weighted_avg(g["price"].to_numpy(), g["quantity"].to_numpy()),
            "trade_price_mean": g["price"].mean(),
            "trade_sign_proxy": np.sign(np.nansum(g["trade_sign_proxy"] * g["quantity"])),
            "frac_at_or_above_ask_touch": (g["trade_location"] == "at_or_above_ask_touch").mean(),
            "frac_at_or_below_bid_touch": (g["trade_location"] == "at_or_below_bid_touch").mean(),
            "frac_inside_touch": (g["trade_location"] == "inside_touch").mean(),
        }),
        include_groups=False,
    ).reset_index()

    agg = agg.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)
    agg["next_trade_sign_proxy"] = agg.groupby(["product", "day"])["trade_sign_proxy"].shift(-1)

    return agg, trades


def merge_book_and_trade(book_df, trade_agg):
    df = book_df.merge(trade_agg, on=["product", "day", "timestamp"], how="left")

    for c in ["trade_count", "trade_volume", "frac_at_or_above_ask_touch",
              "frac_at_or_below_bid_touch", "frac_inside_touch"]:
        df[c] = df[c].fillna(0)

    df["trade_vwap_minus_touch_mid"] = df["trade_vwap"] - df["touch_mid"]
    df["trade_vwap_minus_sizewall_mid"] = df["trade_vwap"] - df["size_wall_mid"]

    return df
