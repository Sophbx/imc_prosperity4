#!/usr/bin/env python3
"""
Order Book / Price / Market-State Analysis Script
=================================================

What this script does
---------------------
1. Loads IMC-style price and trade CSV files (semicolon-separated by default).
2. Computes:
   - mid price
   - trade VWAP
   - rolling trade VWAP
   - spread = best_ask - best_bid
   - wall prices and wall imbalance
   - depth / pressure / concentration / slope features
3. Characterizes order-book states:
   - spread state
   - depth state
   - pressure state
   - combined market state
4. Performs unsupervised market-state analysis:
   - PCA
   - KMeans clustering
   - t-SNE (sampled)
5. Performs "bot behavior" analysis as far as data allows:
   - if buyer/seller IDs are present: per-participant stats
   - otherwise: aggressor-side flow proxy and burst-pattern analysis
6. Saves:
   - feature tables
   - summaries
   - plots

Why this script is structured this way
--------------------------------------
In many IMC/Prosperity datasets, trade files do not always contain reliable per-bot IDs.
So the script is robust:
- if participant IDs exist -> do participant analysis
- if not -> infer behavior from trade flow + order-book states

Example usage
-------------
python orderbook_market_analysis.py \
  --prices prices_round_1_day_-2.csv prices_round_1_day_-1.csv prices_round_1_day_0.csv \
  --trades trades_round_1_day_-2.csv trades_round_1_day_-1.csv trades_round_1_day_0.csv \
  --output-dir analysis_output \
  --sample-size 500

Dependencies
------------
pip install pandas numpy matplotlib scikit-learn
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PRICE_LEVELS_TO_SCAN = 3  # best 3 bid/ask levels
ROLLING_VWAP_WINDOW = 20
ROLLING_RET_WINDOW = 20
RANDOM_SEED = 42


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def extract_day_from_filename(path: str) -> int:
    """
    Try to extract the day integer from a filename like:
      prices_round_1_day_-2.csv
      trades_round_1_day_0.csv
    """
    base = os.path.basename(path)
    m = re.search(r"day_(-?\d+)", base)
    if m:
        return int(m.group(1))
    return 0


def robust_read_csv(path: str) -> pd.DataFrame:
    """
    Read CSV with a few fallback separators.
    Most Prosperity-style files use semicolon.
    """
    seps = [";", ",", "\t"]
    last_err = None
    for sep in seps:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read {path}. Last error: {last_err}")


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total_w = np.nansum(weights)
    if total_w <= 0:
        return np.nan
    return np.nansum(values * weights) / total_w


def first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(a, b)
    if isinstance(out, pd.Series):
        out = out.where(np.isfinite(out), np.nan)
    elif isinstance(out, pd.DataFrame):
        out = out.where(np.isfinite(out), np.nan)
    elif isinstance(out, np.ndarray):
        out[~np.isfinite(out)] = np.nan
    elif not np.isfinite(out):
        out = np.nan
    return out


# ---------------------------------------------------------------------
# Column inference
# ---------------------------------------------------------------------

@dataclass
class PriceSchema:
    timestamp: str
    product: str
    bid_prices: List[str]
    bid_volumes: List[str]
    ask_prices: List[str]
    ask_volumes: List[str]


@dataclass
class TradeSchema:
    timestamp: str
    product: str
    price: str
    quantity: str
    buyer: Optional[str]
    seller: Optional[str]


def infer_price_schema(df: pd.DataFrame) -> PriceSchema:
    cols = list(df.columns)

    timestamp = first_existing(df, ["timestamp", "time", "ts"])
    product = first_existing(df, ["product", "symbol", "instrument"])

    if timestamp is None or product is None:
        raise ValueError(
            f"Could not find timestamp/product columns in price file. Columns={cols}"
        )

    bid_prices = []
    bid_volumes = []
    ask_prices = []
    ask_volumes = []

    # Common IMC style:
    # bid_price_1, bid_volume_1, ..., ask_price_1, ask_volume_1
    for i in range(1, 11):
        bp = f"bid_price_{i}"
        bv = f"bid_volume_{i}"
        ap = f"ask_price_{i}"
        av = f"ask_volume_{i}"

        if bp in df.columns and bv in df.columns:
            bid_prices.append(bp)
            bid_volumes.append(bv)
        if ap in df.columns and av in df.columns:
            ask_prices.append(ap)
            ask_volumes.append(av)

    if not bid_prices or not ask_prices:
        raise ValueError(
            f"Could not infer bid/ask ladder columns from price file. Columns={cols}"
        )

    return PriceSchema(
        timestamp=timestamp,
        product=product,
        bid_prices=bid_prices[:PRICE_LEVELS_TO_SCAN],
        bid_volumes=bid_volumes[:PRICE_LEVELS_TO_SCAN],
        ask_prices=ask_prices[:PRICE_LEVELS_TO_SCAN],
        ask_volumes=ask_volumes[:PRICE_LEVELS_TO_SCAN],
    )


def infer_trade_schema(df: pd.DataFrame) -> TradeSchema:
    cols = list(df.columns)

    timestamp = first_existing(df, ["timestamp", "time", "ts"])
    product = first_existing(df, ["product", "symbol", "instrument"])
    price = first_existing(df, ["price", "trade_price"])
    quantity = first_existing(df, ["quantity", "qty", "volume", "size"])
    buyer = first_existing(df, ["buyer", "buyer_id", "bidder"])
    seller = first_existing(df, ["seller", "seller_id", "asker"])

    if timestamp is None or product is None or price is None or quantity is None:
        raise ValueError(
            f"Could not infer key trade columns. Columns={cols}"
        )

    return TradeSchema(
        timestamp=timestamp,
        product=product,
        price=price,
        quantity=quantity,
        buyer=buyer,
        seller=seller,
    )


# ---------------------------------------------------------------------
# Loading and standardization
# ---------------------------------------------------------------------

def load_price_files(paths: List[str]) -> pd.DataFrame:
    dfs = []

    for path in paths:
        df = robust_read_csv(path)
        schema = infer_price_schema(df)

        num_cols = [schema.timestamp] + schema.bid_prices + schema.bid_volumes + schema.ask_prices + schema.ask_volumes
        df = coerce_numeric(df, num_cols)

        df["day"] = extract_day_from_filename(path)
        df["source_file"] = os.path.basename(path)

        rename_map = {
            schema.timestamp: "timestamp",
            schema.product: "product",
        }
        for i, c in enumerate(schema.bid_prices, start=1):
            rename_map[c] = f"bid_price_{i}"
        for i, c in enumerate(schema.bid_volumes, start=1):
            rename_map[c] = f"bid_volume_{i}"
        for i, c in enumerate(schema.ask_prices, start=1):
            rename_map[c] = f"ask_price_{i}"
        for i, c in enumerate(schema.ask_volumes, start=1):
            rename_map[c] = f"ask_volume_{i}"

        df = df.rename(columns=rename_map)

        keep_cols = (
            ["day", "source_file", "timestamp", "product"] +
            [f"bid_price_{i}" for i in range(1, len(schema.bid_prices) + 1)] +
            [f"bid_volume_{i}" for i in range(1, len(schema.bid_volumes) + 1)] +
            [f"ask_price_{i}" for i in range(1, len(schema.ask_prices) + 1)] +
            [f"ask_volume_{i}" for i in range(1, len(schema.ask_volumes) + 1)]
        )

        dfs.append(df[keep_cols].copy())

    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)
    return out


def load_trade_files(paths: List[str]) -> pd.DataFrame:
    dfs = []

    for path in paths:
        df = robust_read_csv(path)
        schema = infer_trade_schema(df)

        num_cols = [schema.timestamp, schema.price, schema.quantity]
        df = coerce_numeric(df, num_cols)

        df["day"] = extract_day_from_filename(path)
        df["source_file"] = os.path.basename(path)

        rename_map = {
            schema.timestamp: "timestamp",
            schema.product: "product",
            schema.price: "price",
            schema.quantity: "quantity",
        }
        if schema.buyer is not None:
            rename_map[schema.buyer] = "buyer"
        if schema.seller is not None:
            rename_map[schema.seller] = "seller"

        df = df.rename(columns=rename_map)

        if "buyer" not in df.columns:
            df["buyer"] = np.nan
        if "seller" not in df.columns:
            df["seller"] = np.nan

        keep_cols = [
            "day", "source_file", "timestamp", "product",
            "price", "quantity", "buyer", "seller"
        ]
        dfs.append(df[keep_cols].copy())

    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------
# Feature engineering: order-book
# ---------------------------------------------------------------------

def compute_orderbook_features(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy()

    # Basic best prices/volumes
    df["best_bid"] = df["bid_price_1"]
    df["best_ask"] = df["ask_price_1"]
    df["best_bid_vol"] = df["bid_volume_1"]
    df["best_ask_vol"] = df["ask_volume_1"]

    # Mid and spread
    df["mid"] = (df["best_bid"] + df["best_ask"]) / 2.0
    df["spread"] = df["best_ask"] - df["best_bid"]

    # Level-1 queue imbalance
    df["queue_imbalance_l1"] = safe_div(
        df["best_bid_vol"] - df["best_ask_vol"],
        df["best_bid_vol"] + df["best_ask_vol"]
    )

    # Microprice
    # Standard form: weighted toward side with smaller resistance
    df["microprice"] = safe_div(
        df["best_ask"] * df["best_bid_vol"] + df["best_bid"] * df["best_ask_vol"],
        df["best_bid_vol"] + df["best_ask_vol"]
    )
    df["micro_minus_mid"] = df["microprice"] - df["mid"]

    # Total depth (top N levels)
    bid_depth_cols = [c for c in df.columns if c.startswith("bid_volume_")]
    ask_depth_cols = [c for c in df.columns if c.startswith("ask_volume_")]

    df["bid_depth_total"] = df[bid_depth_cols].sum(axis=1, min_count=1)
    df["ask_depth_total"] = df[ask_depth_cols].sum(axis=1, min_count=1)
    df["depth_total"] = df["bid_depth_total"] + df["ask_depth_total"]

    # Multi-level imbalance
    df["depth_imbalance"] = safe_div(
        df["bid_depth_total"] - df["ask_depth_total"],
        df["bid_depth_total"] + df["ask_depth_total"]
    )

    # Relative pressure:
    # weight closer levels more heavily
    bid_pressure = np.zeros(len(df))
    ask_pressure = np.zeros(len(df))
    total_weights = 0.0

    for i in range(1, PRICE_LEVELS_TO_SCAN + 1):
        bv = f"bid_volume_{i}"
        av = f"ask_volume_{i}"
        weight = 1.0 / i
        if bv in df.columns:
            bid_pressure += df[bv].fillna(0).to_numpy() * weight
        if av in df.columns:
            ask_pressure += df[av].fillna(0).to_numpy() * weight
        total_weights += weight

    df["bid_pressure_raw"] = bid_pressure
    df["ask_pressure_raw"] = ask_pressure
    df["relative_pressure"] = safe_div(
        df["bid_pressure_raw"] - df["ask_pressure_raw"],
        df["bid_pressure_raw"] + df["ask_pressure_raw"]
    )

    # Walls: largest displayed volume among scanned levels
    bid_vol_matrix = np.column_stack([df.get(f"bid_volume_{i}", pd.Series(np.nan, index=df.index)).to_numpy()
                                      for i in range(1, PRICE_LEVELS_TO_SCAN + 1)])
    ask_vol_matrix = np.column_stack([df.get(f"ask_volume_{i}", pd.Series(np.nan, index=df.index)).to_numpy()
                                      for i in range(1, PRICE_LEVELS_TO_SCAN + 1)])

    bid_price_matrix = np.column_stack([df.get(f"bid_price_{i}", pd.Series(np.nan, index=df.index)).to_numpy()
                                        for i in range(1, PRICE_LEVELS_TO_SCAN + 1)])
    ask_price_matrix = np.column_stack([df.get(f"ask_price_{i}", pd.Series(np.nan, index=df.index)).to_numpy()
                                        for i in range(1, PRICE_LEVELS_TO_SCAN + 1)])

    with np.errstate(all="ignore"):
        bid_wall_idx = np.nanargmax(np.where(np.isnan(bid_vol_matrix), -np.inf, bid_vol_matrix), axis=1)
        ask_wall_idx = np.nanargmax(np.where(np.isnan(ask_vol_matrix), -np.inf, ask_vol_matrix), axis=1)

    row_idx = np.arange(len(df))
    df["bid_wall_volume"] = bid_vol_matrix[row_idx, bid_wall_idx]
    df["ask_wall_volume"] = ask_vol_matrix[row_idx, ask_wall_idx]
    df["bid_wall_price"] = bid_price_matrix[row_idx, bid_wall_idx]
    df["ask_wall_price"] = ask_price_matrix[row_idx, ask_wall_idx]

    df["wall_volume_imbalance"] = safe_div(
        df["bid_wall_volume"] - df["ask_wall_volume"],
        df["bid_wall_volume"] + df["ask_wall_volume"]
    )
    df["wall_price_gap"] = df["ask_wall_price"] - df["bid_wall_price"]
    df["bid_wall_minus_ask_wall"] = df["bid_wall_price"] - df["ask_wall_price"]

    # Concentration: max-volume share of same-side visible depth
    df["bid_concentration"] = safe_div(df["bid_wall_volume"], df["bid_depth_total"])
    df["ask_concentration"] = safe_div(df["ask_wall_volume"], df["ask_depth_total"])

    # Gaps between levels
    if PRICE_LEVELS_TO_SCAN >= 2:
        bid_gaps = []
        ask_gaps = []
        for i in range(1, PRICE_LEVELS_TO_SCAN):
            if f"bid_price_{i}" in df.columns and f"bid_price_{i+1}" in df.columns:
                bid_gaps.append(df[f"bid_price_{i}"] - df[f"bid_price_{i+1}"])
            if f"ask_price_{i}" in df.columns and f"ask_price_{i+1}" in df.columns:
                ask_gaps.append(df[f"ask_price_{i+1}"] - df[f"ask_price_{i}"])

        if bid_gaps:
            df["bid_gap_mean"] = pd.concat(bid_gaps, axis=1).mean(axis=1)
        else:
            df["bid_gap_mean"] = np.nan

        if ask_gaps:
            df["ask_gap_mean"] = pd.concat(ask_gaps, axis=1).mean(axis=1)
        else:
            df["ask_gap_mean"] = np.nan
    else:
        df["bid_gap_mean"] = np.nan
        df["ask_gap_mean"] = np.nan

    # Depth slope approximation:
    # fit size against distance from best level index 1..N
    def slope_from_levels(vol_cols: List[str], local_df: pd.DataFrame) -> pd.Series:
        xs = np.arange(1, len(vol_cols) + 1, dtype=float)
        out = np.full(len(local_df), np.nan)
        for idx in range(len(local_df)):
            ys = local_df.iloc[idx][vol_cols].to_numpy(dtype=float)
            mask = np.isfinite(ys)
            if mask.sum() >= 2:
                x = xs[mask]
                y = ys[mask]
                # simple least-squares slope
                denom = np.sum((x - x.mean()) ** 2)
                if denom > 0:
                    out[idx] = np.sum((x - x.mean()) * (y - y.mean())) / denom
        return pd.Series(out, index=local_df.index)

    df["bid_slope"] = slope_from_levels([c for c in bid_depth_cols if c in df.columns], df)
    df["ask_slope"] = slope_from_levels([c for c in ask_depth_cols if c in df.columns], df)

    # Returns / trend variables
    df["mid_return_1"] = df.groupby(["product", "day"])["mid"].pct_change()
    df["mid_change_1"] = df.groupby(["product", "day"])["mid"].diff()
    df["mid_ma_short"] = df.groupby(["product", "day"])["mid"].transform(
        lambda s: s.rolling(10, min_periods=1).mean()
    )
    df["mid_ma_long"] = df.groupby(["product", "day"])["mid"].transform(
        lambda s: s.rolling(50, min_periods=1).mean()
    )
    df["mid_trend_signal"] = df["mid_ma_short"] - df["mid_ma_long"]
    df["mid_volatility_20"] = df.groupby(["product", "day"])["mid_return_1"].transform(
        lambda s: s.rolling(ROLLING_RET_WINDOW, min_periods=5).std()
    )

    return df


# ---------------------------------------------------------------------
# Trade aggregation / VWAP / flow
# ---------------------------------------------------------------------

def compute_trade_features(price_features: pd.DataFrame, trade_df: pd.DataFrame) -> pd.DataFrame:
    """
    Align trades to the same (product, day, timestamp) granularity.
    """
    trades = trade_df.copy()

    # Per timestamp trade stats
    grouped = trades.groupby(["product", "day", "timestamp"], dropna=False)

    trade_agg = grouped.apply(
        lambda g: pd.Series({
            "trade_count": len(g),
            "trade_volume": g["quantity"].sum(skipna=True),
            "trade_turnover": (g["price"] * g["quantity"]).sum(skipna=True),
            "trade_vwap": weighted_average(g["price"].to_numpy(), g["quantity"].to_numpy()),
            "trade_price_mean": g["price"].mean(),
            "trade_price_std": g["price"].std(ddof=0),
            "trade_max_price": g["price"].max(),
            "trade_min_price": g["price"].min(),
            "buyer_nonnull_count": g["buyer"].notna().sum(),
            "seller_nonnull_count": g["seller"].notna().sum(),
        })
    ).reset_index()

    # Merge into price table
    df = price_features.merge(
        trade_agg,
        on=["product", "day", "timestamp"],
        how="left"
    )

    fill_zero_cols = ["trade_count", "trade_volume", "trade_turnover", "buyer_nonnull_count", "seller_nonnull_count"]
    for c in fill_zero_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # Rolling trade VWAP
    def rolling_vwap(group: pd.DataFrame) -> pd.Series:
        num = (group["trade_vwap"].fillna(0) * group["trade_volume"].fillna(0)).rolling(
            ROLLING_VWAP_WINDOW, min_periods=1
        ).sum()
        den = group["trade_volume"].fillna(0).rolling(
            ROLLING_VWAP_WINDOW, min_periods=1
        ).sum()
        return num / den.replace(0, np.nan)

    df["rolling_trade_vwap_20"] = df.groupby(["product", "day"], group_keys=False).apply(rolling_vwap)

    # Trade premium/discount relative to book
    df["trade_vwap_minus_mid"] = df["trade_vwap"] - df["mid"]
    df["trade_vwap_minus_micro"] = df["trade_vwap"] - df["microprice"]

    # Infer aggressor side if possible from price vs quotes
    # buy if trade closer to ask, sell if closer to bid, otherwise unknown
    buy_score = np.abs(df["trade_vwap"] - df["best_ask"])
    sell_score = np.abs(df["trade_vwap"] - df["best_bid"])

    df["inferred_aggressor"] = np.where(
        df["trade_count"] <= 0,
        "none",
        np.where(
            buy_score < sell_score,
            "buy",
            np.where(sell_score < buy_score, "sell", "unknown")
        )
    )

    df["buy_trade_volume_proxy"] = np.where(df["inferred_aggressor"] == "buy", df["trade_volume"], 0.0)
    df["sell_trade_volume_proxy"] = np.where(df["inferred_aggressor"] == "sell", df["trade_volume"], 0.0)
    df["signed_flow_proxy"] = df["buy_trade_volume_proxy"] - df["sell_trade_volume_proxy"]

    return df


# ---------------------------------------------------------------------
# State characterization
# ---------------------------------------------------------------------

def make_quantile_state(series: pd.Series, low_q=0.33, high_q=0.67,
                        labels=("low", "mid", "high")) -> pd.Series:
    s = series.copy()
    q1 = s.quantile(low_q)
    q2 = s.quantile(high_q)

    return pd.Series(
        np.where(
            s <= q1, labels[0],
            np.where(s >= q2, labels[2], labels[1])
        ),
        index=s.index
    )


def characterize_market_states(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Spread state
    out["spread_state"] = make_quantile_state(
        out["spread"], labels=("tight", "normal", "wide")
    )

    # Depth state
    out["depth_state"] = make_quantile_state(
        out["depth_total"], labels=("thin", "medium", "thick")
    )

    # Pressure state from relative pressure
    rp = out["relative_pressure"]
    thr = max(0.05, float(rp.abs().quantile(0.4))) if rp.notna().any() else 0.05

    out["pressure_state"] = np.where(
        rp >= thr, "bid_pressure",
        np.where(rp <= -thr, "ask_pressure", "balanced")
    )

    # Trend state
    mts = out["mid_trend_signal"]
    thr_t = max(1e-12, float(mts.abs().quantile(0.4))) if mts.notna().any() else 1e-12
    out["trend_state"] = np.where(
        mts >= thr_t, "uptrend",
        np.where(mts <= -thr_t, "downtrend", "flat")
    )

    out["market_state"] = (
        out["spread_state"].astype(str) + "|" +
        out["depth_state"].astype(str) + "|" +
        out["pressure_state"].astype(str)
    )

    out["market_state_full"] = (
        out["spread_state"].astype(str) + "|" +
        out["depth_state"].astype(str) + "|" +
        out["pressure_state"].astype(str) + "|" +
        out["trend_state"].astype(str)
    )

    return out


# ---------------------------------------------------------------------
# Participant / "bot" analysis
# ---------------------------------------------------------------------

def analyze_participants(trade_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      participant_summary
      burst_summary
    """
    trades = trade_df.copy()

    has_buyer = trades["buyer"].notna().any() if "buyer" in trades.columns else False
    has_seller = trades["seller"].notna().any() if "seller" in trades.columns else False

    if has_buyer or has_seller:
        buy_side = trades[trades["buyer"].notna()].copy()
        buy_side["participant"] = buy_side["buyer"]
        buy_side["role"] = "buyer"

        sell_side = trades[trades["seller"].notna()].copy()
        sell_side["participant"] = sell_side["seller"]
        sell_side["role"] = "seller"

        ps = pd.concat([buy_side, sell_side], ignore_index=True)
        summary = ps.groupby(["participant", "product", "role"]).agg(
            trades=("price", "count"),
            total_qty=("quantity", "sum"),
            avg_qty=("quantity", "mean"),
            avg_price=("price", "mean"),
            price_std=("price", "std"),
            active_days=("day", "nunique"),
            active_timestamps=("timestamp", "nunique"),
        ).reset_index()

        # Burstiness: count max trades per timestamp for each participant
        burst = ps.groupby(["participant", "product", "timestamp"]).size().reset_index(name="count_at_timestamp")
        burst_summary = burst.groupby(["participant", "product"]).agg(
            mean_burst=("count_at_timestamp", "mean"),
            max_burst=("count_at_timestamp", "max"),
            std_burst=("count_at_timestamp", "std"),
        ).reset_index()

        return summary, burst_summary

    # If no participant IDs, fallback to "market burst" analysis
    burst = trades.groupby(["product", "day", "timestamp"]).size().reset_index(name="trade_count_at_timestamp")
    burst_summary = burst.groupby(["product"]).agg(
        mean_burst=("trade_count_at_timestamp", "mean"),
        max_burst=("trade_count_at_timestamp", "max"),
        std_burst=("trade_count_at_timestamp", "std"),
        timestamps_with_burst=("trade_count_at_timestamp", lambda s: int((s >= s.quantile(0.95)).sum())),
    ).reset_index()

    participant_summary = pd.DataFrame({
        "note": [
            "No non-null buyer/seller IDs found. True participant-level bot analysis is unavailable in this dataset."
        ]
    })

    return participant_summary, burst_summary


# ---------------------------------------------------------------------
# PCA / clustering / t-SNE
# ---------------------------------------------------------------------

def run_unsupervised_analysis(df: pd.DataFrame, output_dir: str, sample_size: int = 500) -> Dict[str, pd.DataFrame]:
    """
    Perform PCA, KMeans, and t-SNE on numeric microstructure features.
    """
    safe_mkdir(output_dir)

    feature_cols = [
        "mid",
        "spread",
        "queue_imbalance_l1",
        "micro_minus_mid",
        "bid_depth_total",
        "ask_depth_total",
        "depth_imbalance",
        "relative_pressure",
        "bid_wall_volume",
        "ask_wall_volume",
        "wall_volume_imbalance",
        "bid_concentration",
        "ask_concentration",
        "bid_gap_mean",
        "ask_gap_mean",
        "bid_slope",
        "ask_slope",
        "mid_trend_signal",
        "mid_volatility_20",
        "trade_count",
        "trade_volume",
        "trade_vwap_minus_mid",
        "signed_flow_proxy",
    ]

    use = df[["product", "day", "timestamp"] + feature_cols].copy()
    use = use.replace([np.inf, -np.inf], np.nan)

    # Drop columns that are entirely null
    valid_feature_cols = [c for c in feature_cols if use[c].notna().any()]
    use = use.dropna(subset=valid_feature_cols, how="any").reset_index(drop=True)

    results: Dict[str, pd.DataFrame] = {}

    if len(use) < 20 or len(valid_feature_cols) < 2:
        results["pca_scores"] = pd.DataFrame({"note": ["Not enough clean rows for PCA/KMeans/t-SNE."]})
        results["pca_loadings"] = pd.DataFrame({"note": ["Not enough clean rows for PCA/KMeans/t-SNE."]})
        results["cluster_summary"] = pd.DataFrame({"note": ["Not enough clean rows for PCA/KMeans/t-SNE."]})
        return results

    X = use[valid_feature_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # PCA
    n_components = min(5, Xs.shape[1], Xs.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pcs = pca.fit_transform(Xs)

    pca_scores = use[["product", "day", "timestamp"]].copy()
    for i in range(pcs.shape[1]):
        pca_scores[f"PC{i+1}"] = pcs[:, i]

    loadings = pd.DataFrame(
        pca.components_.T,
        index=valid_feature_cols,
        columns=[f"PC{i+1}" for i in range(pcs.shape[1])]
    )
    loadings["abs_PC1"] = loadings["PC1"].abs()
    loadings = loadings.sort_values("abs_PC1", ascending=False)

    explained = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "explained_variance_ratio": pca.explained_variance_ratio_
    })

    # KMeans on first few PCs
    km_dim = min(3, pcs.shape[1])
    km_input = pcs[:, :km_dim]
    n_clusters = min(5, max(2, len(km_input) // 100))
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=20)
    labels = kmeans.fit_predict(km_input)

    pca_scores["cluster"] = labels
    cluster_summary = pd.concat(
        [
            pca_scores.groupby("cluster")[["PC1"]].size().rename("count"),
            pca_scores.groupby("cluster")[[c for c in pca_scores.columns if c.startswith("PC")]].mean()
        ],
        axis=1
    ).reset_index()

    # Add original feature means by cluster
    tmp = use.copy()
    tmp["cluster"] = labels
    cluster_feature_means = tmp.groupby("cluster")[valid_feature_cols].mean().reset_index()
    cluster_summary = cluster_summary.merge(cluster_feature_means, on="cluster", how="left")

    # t-SNE on sample
    n_sample = min(sample_size, len(use))
    sample_idx = np.random.RandomState(RANDOM_SEED).choice(len(use), size=n_sample, replace=False)
    Xs_sample = Xs[sample_idx]
    use_sample = use.iloc[sample_idx].reset_index(drop=True)
    labels_sample = labels[sample_idx]

    if n_sample >= 30:
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, max(5, n_sample // 10)),
            learning_rate="auto",
            init="pca",
            random_state=RANDOM_SEED
        )
        emb = tsne.fit_transform(Xs_sample)

        tsne_df = use_sample[["product", "day", "timestamp"]].copy()
        tsne_df["tsne_1"] = emb[:, 0]
        tsne_df["tsne_2"] = emb[:, 1]
        tsne_df["cluster"] = labels_sample
    else:
        tsne_df = pd.DataFrame({"note": ["Not enough rows for stable t-SNE."]})

    # Save plots
    if "PC1" in pca_scores.columns and "PC2" in pca_scores.columns:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(pca_scores["PC1"], pca_scores["PC2"], c=pca_scores["cluster"], s=8)
        plt.title("PCA of Market States")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pca_market_states.png"), dpi=150)
        plt.close()

    if "tsne_1" in tsne_df.columns and "tsne_2" in tsne_df.columns:
        plt.figure(figsize=(8, 6))
        plt.scatter(tsne_df["tsne_1"], tsne_df["tsne_2"], c=tsne_df["cluster"], s=10)
        plt.title("t-SNE of Sampled Market States")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "tsne_market_states.png"), dpi=150)
        plt.close()

    # Scree plot
    plt.figure(figsize=(7, 4))
    plt.plot(
        np.arange(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_,
        marker="o"
    )
    plt.title("PCA Explained Variance")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_explained_variance.png"), dpi=150)
    plt.close()

    results["pca_scores"] = pca_scores
    results["pca_loadings"] = loadings
    results["pca_explained_variance"] = explained
    results["cluster_summary"] = cluster_summary
    results["tsne_sample"] = tsne_df

    return results


# ---------------------------------------------------------------------
# Summaries / plotting
# ---------------------------------------------------------------------

def save_product_plots(df: pd.DataFrame, output_dir: str) -> None:
    safe_mkdir(output_dir)

    for product, g in df.groupby("product"):
        g = g.sort_values(["day", "timestamp"]).reset_index(drop=True)

        # Mid / VWAP / microprice
        plt.figure(figsize=(12, 5))
        plt.plot(g.index, g["mid"], label="mid")
        if "trade_vwap" in g.columns:
            plt.plot(g.index, g["trade_vwap"], label="trade_vwap", alpha=0.7)
        if "rolling_trade_vwap_20" in g.columns:
            plt.plot(g.index, g["rolling_trade_vwap_20"], label="rolling_trade_vwap_20", alpha=0.8)
        if "microprice" in g.columns:
            plt.plot(g.index, g["microprice"], label="microprice", alpha=0.8)
        plt.title(f"{product}: Mid / VWAP / Microprice")
        plt.xlabel("Row Index (time ordered)")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{product}_price_trend.png"), dpi=150)
        plt.close()

        # Spread / wall price relationship
        plt.figure(figsize=(12, 5))
        plt.plot(g.index, g["spread"], label="spread")
        if "bid_wall_minus_ask_wall" in g.columns:
            plt.plot(g.index, g["bid_wall_minus_ask_wall"], label="bid_wall - ask_wall")
        plt.title(f"{product}: Spread and Bid/Ask Wall Price Difference")
        plt.xlabel("Row Index (time ordered)")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{product}_spread_and_walls.png"), dpi=150)
        plt.close()

        # Depth / pressure
        plt.figure(figsize=(12, 5))
        plt.plot(g.index, g["bid_depth_total"], label="bid_depth_total")
        plt.plot(g.index, g["ask_depth_total"], label="ask_depth_total")
        plt.plot(g.index, g["relative_pressure"], label="relative_pressure")
        plt.title(f"{product}: Depth and Relative Pressure")
        plt.xlabel("Row Index (time ordered)")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{product}_depth_pressure.png"), dpi=150)
        plt.close()

        # Signed flow proxy
        if "signed_flow_proxy" in g.columns:
            plt.figure(figsize=(12, 5))
            plt.plot(g.index, g["signed_flow_proxy"], label="signed_flow_proxy")
            plt.title(f"{product}: Signed Flow Proxy")
            plt.xlabel("Row Index (time ordered)")
            plt.ylabel("Signed Volume")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{product}_signed_flow_proxy.png"), dpi=150)
            plt.close()


def make_summaries(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    summaries = {}

    # Product summary
    product_summary = df.groupby("product").agg(
        rows=("mid", "size"),
        mid_mean=("mid", "mean"),
        mid_std=("mid", "std"),
        spread_mean=("spread", "mean"),
        spread_std=("spread", "std"),
        depth_mean=("depth_total", "mean"),
        depth_std=("depth_total", "std"),
        rel_pressure_mean=("relative_pressure", "mean"),
        rel_pressure_std=("relative_pressure", "std"),
        trade_count_sum=("trade_count", "sum"),
        trade_volume_sum=("trade_volume", "sum"),
        signed_flow_proxy_mean=("signed_flow_proxy", "mean"),
    ).reset_index()

    summaries["product_summary"] = product_summary

    # Market state summary
    state_summary = df.groupby(["product", "market_state"]).agg(
        count=("mid", "size"),
        mid_change_mean=("mid_change_1", "mean"),
        mid_return_mean=("mid_return_1", "mean"),
        spread_mean=("spread", "mean"),
        depth_mean=("depth_total", "mean"),
        rel_pressure_mean=("relative_pressure", "mean"),
        trade_volume_mean=("trade_volume", "mean"),
        signed_flow_mean=("signed_flow_proxy", "mean"),
    ).reset_index().sort_values(["product", "count"], ascending=[True, False])

    summaries["market_state_summary"] = state_summary

    # Full state summary
    full_state_summary = df.groupby(["product", "market_state_full"]).agg(
        count=("mid", "size"),
        mid_change_mean=("mid_change_1", "mean"),
        mid_return_mean=("mid_return_1", "mean"),
        spread_mean=("spread", "mean"),
        depth_mean=("depth_total", "mean"),
        rel_pressure_mean=("relative_pressure", "mean"),
        trade_volume_mean=("trade_volume", "mean"),
        signed_flow_mean=("signed_flow_proxy", "mean"),
    ).reset_index().sort_values(["product", "count"], ascending=[True, False])

    summaries["market_state_full_summary"] = full_state_summary

    return summaries


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def run_analysis(price_paths: List[str], trade_paths: List[str], output_dir: str, sample_size: int) -> None:
    safe_mkdir(output_dir)

    print("Loading price files...")
    price_df = load_price_files(price_paths)
    print(f"Loaded price rows: {len(price_df):,}")

    print("Loading trade files...")
    trade_df = load_trade_files(trade_paths)
    print(f"Loaded trade rows: {len(trade_df):,}")

    print("Computing order-book features...")
    book_df = compute_orderbook_features(price_df)

    print("Computing trade features...")
    merged_df = compute_trade_features(book_df, trade_df)

    print("Characterizing market states...")
    merged_df = characterize_market_states(merged_df)

    # Save main feature table
    merged_path = os.path.join(output_dir, "market_features.csv")
    merged_df.to_csv(merged_path, index=False)
    print(f"Saved feature table: {merged_path}")

    # Summaries
    print("Creating summaries...")
    summaries = make_summaries(merged_df)
    for name, sdf in summaries.items():
        path = os.path.join(output_dir, f"{name}.csv")
        sdf.to_csv(path, index=False)
        print(f"Saved summary: {path}")

    # Participant / burst analysis
    print("Analyzing participants / bot-like behavior...")
    participant_summary, burst_summary = analyze_participants(trade_df)
    participant_summary.to_csv(os.path.join(output_dir, "participant_summary.csv"), index=False)
    burst_summary.to_csv(os.path.join(output_dir, "burst_summary.csv"), index=False)

    # Unsupervised analysis
    print("Running PCA / clustering / t-SNE...")
    unsup_dir = os.path.join(output_dir, "unsupervised")
    safe_mkdir(unsup_dir)
    unsup = run_unsupervised_analysis(merged_df, unsup_dir, sample_size=sample_size)
    for name, udf in unsup.items():
        path = os.path.join(unsup_dir, f"{name}.csv")
        udf.to_csv(path, index=False)
        print(f"Saved unsupervised output: {path}")

    # Plots
    print("Saving plots...")
    plots_dir = os.path.join(output_dir, "plots")
    save_product_plots(merged_df, plots_dir)

    print("\nDone.")
    print(f"All outputs saved under: {output_dir}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze price changes, order book depth, and market states.")
    parser.add_argument(
        "--prices",
        nargs="+",
        required=True,
        help="List of price CSV files."
    )
    parser.add_argument(
        "--trades",
        nargs="+",
        required=True,
        help="List of trade CSV files."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_output",
        help="Directory to save outputs."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Sample size for t-SNE."
    )
    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args()
    run_analysis(
        price_paths=args.prices,
        trade_paths=args.trades,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
    )
