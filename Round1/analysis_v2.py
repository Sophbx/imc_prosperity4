#!/usr/bin/env python3

import os
import re
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================
PRICE_FILES = [
    "prices_round_1_day_-2.csv",
    "prices_round_1_day_-1.csv",
    "prices_round_1_day_0.csv",
]

TRADE_FILES = [
    "trades_round_1_day_-2.csv",
    "trades_round_1_day_-1.csv",
    "trades_round_1_day_0.csv",
]

VISIBLE_LEVELS = 3
HORIZON = 10
PRODUCTS = None   # Example: ["ASH_COATED_OSMIUM"]


# ============================================================
# SMALL HELPERS
# ============================================================
def print_section(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def extract_day_from_filename(path: str) -> int:
    m = re.search(r"day_(-?\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def safe_corr(x, y):
    z = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(z) < 3:
        return np.nan
    if z["x"].std() == 0 or z["y"].std() == 0:
        return np.nan
    return z["x"].corr(z["y"])


def weighted_avg(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights)
    if mask.sum() == 0:
        return np.nan
    values = values[mask]
    weights = weights[mask]
    if weights.sum() <= 0:
        return np.nan
    return np.sum(values * weights) / np.sum(weights)


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


# ============================================================
# LOADING
# ============================================================
def load_prices(script_dir):
    dfs = []
    for fn in PRICE_FILES:
        path = os.path.join(script_dir, fn)
        df = pd.read_csv(path, sep=";")
        if "day" not in df.columns:
            df["day"] = extract_day_from_filename(fn)
        dfs.append(df)

    prices = pd.concat(dfs, ignore_index=True)
    prices = prices.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)
    return prices


def load_trades(script_dir):
    dfs = []
    for fn in TRADE_FILES:
        path = os.path.join(script_dir, fn)
        df = pd.read_csv(path, sep=";")
        if "day" not in df.columns:
            df["day"] = extract_day_from_filename(fn)
        dfs.append(df)

    trades = pd.concat(dfs, ignore_index=True)
    trades = trades.sort_values(["symbol", "day", "timestamp"]).reset_index(drop=True)
    return trades


# ============================================================
# FEATURE ENGINEERING: PRICE / BOOK
# ============================================================
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


# ============================================================
# FEATURE ENGINEERING: TRADES
# ============================================================
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
        })
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


# ============================================================
# PRINTS: MID CHECK
# ============================================================
def print_mid_price_check(df, product):
    x = df[df["product"] == product].copy()

    print_section(f"MID-PRICE CHECK: {product}")
    print("Check dataset mid_price - (bid_price_1 + ask_price_1)/2")
    print(f"rows                                 : {len(x)}")
    print(f"max abs difference                   : {x['dataset_mid_minus_touch_mid'].abs().max():.12f}")
    print(f"mean difference                      : {x['dataset_mid_minus_touch_mid'].mean():.12f}")
    print(f"std difference                       : {x['dataset_mid_minus_touch_mid'].std():.12f}")
    print(f"fraction exactly zero                : {(x['dataset_mid_minus_touch_mid'] == 0).mean():.6f}")
    print(f"fraction approximately zero (<1e-12) : {(x['dataset_mid_minus_touch_mid'].abs() < 1e-12).mean():.6f}")


# ============================================================
# LAYER 1: STATIC GEOMETRY
# ============================================================
def print_layer1_static_geometry(df, product):
    x = df[df["product"] == product].copy()

    summary = {
        "touch_spread_mean": x["touch_spread"].mean(),
        "touch_spread_median": x["touch_spread"].median(),
        "boundary_width_mean": x["boundary_width"].mean(),
        "size_wall_width_mean": x["size_wall_width"].mean(),
        "frontier_imbalance_mean": x["frontier_imbalance"].mean(),
        "depth_imbalance_mean": x["depth_imbalance"].mean(),
        "size_wall_vol_imbalance_mean": x["size_wall_vol_imbalance"].mean(),
        "touch_mid_mean": x["touch_mid"].mean(),
        "boundary_mid_mean": x["boundary_mid"].mean(),
        "size_wall_mid_mean": x["size_wall_mid"].mean(),
        "side_vwap_mid_mean": x["side_vwap_mid"].mean(),
        "full_book_vwap_center_mean": x["full_book_vwap_center"].mean(),
        "dataset_mid_mean": x["mid_price"].mean(),
    }

    print_section(f"LAYER 1: STATIC GEOMETRY / BOOK SHAPE: {product}")
    for k, v in summary.items():
        print(f"{k:35s}: {v:.6f}")


# ============================================================
# LAYER 2: DISLOCATIONS / CORRESPONDENCE
# ============================================================
def print_layer2_dislocations(df, product):
    x = df[df["product"] == product].copy()

    fair_value_cols = [
        "touch_mid",
        "boundary_mid",
        "size_wall_mid",
        "side_vwap_mid",
        "full_book_vwap_center",
        "mid_price",
    ]

    corr = x[fair_value_cols].corr()

    dislocation_cols = [
        "boundary_mid_minus_touch_mid",
        "size_wall_mid_minus_touch_mid",
        "side_vwap_mid_minus_touch_mid",
        "full_book_vwap_center_minus_touch_mid",
        "mid_price_minus_touch_mid",
        "boundary_minus_touch_width",
        "sizewall_minus_touch_width",
        "side_vwap_skew",
        "frontier_imbalance",
        "depth_imbalance",
        "size_wall_vol_imbalance",
    ]

    disp = x[dislocation_cols].agg(["mean", "std", "min", "max"]).T

    print_section(f"LAYER 2A: FAIR-VALUE CORRESPONDENCE MATRIX: {product}")
    print(corr.round(6).to_string())

    print_section(f"LAYER 2B: DISLOCATION SUMMARY: {product}")
    print(disp.round(6).to_string())


# ============================================================
# LAYER 3: PREDICTIVE RELATIONS
# ============================================================
def print_layer3_predictive_price(df, product):
    x = df[df["product"] == product].copy()

    target_change = x[f"fwd_touch_mid_change_{HORIZON}"]
    target_dir = x[f"fwd_direction_{HORIZON}"]

    features = [
        "touch_spread",
        "boundary_width",
        "size_wall_width",
        "frontier_imbalance",
        "depth_imbalance",
        "size_wall_vol_imbalance",
        "boundary_mid_minus_touch_mid",
        "size_wall_mid_minus_touch_mid",
        "side_vwap_mid_minus_touch_mid",
        "full_book_vwap_center_minus_touch_mid",
        "trade_vwap_minus_touch_mid",
        "trade_count",
        "trade_volume",
        "trade_sign_proxy",
    ]

    rows = []
    for feat in features:
        rows.append({
            "feature": feat,
            f"corr_with_fwd_change_{HORIZON}": safe_corr(x[feat], target_change),
            f"corr_with_fwd_direction_{HORIZON}": safe_corr(x[feat], target_dir),
        })

    out = pd.DataFrame(rows)
    out["abs_corr_change"] = out[f"corr_with_fwd_change_{HORIZON}"].abs()
    out = out.sort_values("abs_corr_change", ascending=False)

    print_section(f"LAYER 3A: PREDICTIVE MATRIX FOR NEXT PRICE MOVE: {product}")
    print(out.round(6).to_string(index=False))


def print_predictive_next_trade_direction(df, product):
    x = df[df["product"] == product].copy()

    target = x["next_trade_sign_proxy"]

    features = [
        "frontier_imbalance",
        "depth_imbalance",
        "size_wall_vol_imbalance",
        "touch_spread",
        "boundary_width",
        "size_wall_width",
        "trade_sign_proxy",
        "trade_vwap_minus_touch_mid",
    ]

    rows = []
    for feat in features:
        rows.append({
            "feature": feat,
            "corr_with_next_trade_direction": safe_corr(x[feat], target)
        })

    out = pd.DataFrame(rows).sort_values("corr_with_next_trade_direction", key=lambda s: s.abs(), ascending=False)

    print_section(f"LAYER 3B: PREDICTIVE MATRIX FOR NEXT TRADE DIRECTION: {product}")
    print(out.round(6).to_string(index=False))


# ============================================================
# INTERACTION A: TRADE AT TOUCH / INSIDE / ETC
# ============================================================
def print_interaction_A_trade_location(raw_trades, product):
    t = raw_trades[raw_trades["product"] == product].copy()

    if len(t) == 0:
        print_section(f"INTERACTION A: TRADE LOCATION VS BOOK: {product}")
        print("No trades.")
        return

    out = t["trade_location"].value_counts(dropna=False).to_frame("count")
    out["fraction"] = out["count"] / out["count"].sum()

    print_section(f"INTERACTION A: TRADE LOCATION VS ORDER BOOK: {product}")
    print(out.round(6).to_string())


# ============================================================
# INTERACTION B: TRADE FLOW VS PRE-TRADE IMBALANCE
# ============================================================
def print_interaction_B_tradeflow_vs_imbalance(df, product):
    x = df[df["product"] == product].copy()
    x = x.dropna(subset=["trade_sign_proxy", "frontier_imbalance", "depth_imbalance"])

    rows = [
        {
            "feature": "frontier_imbalance",
            "corr_with_current_trade_sign": safe_corr(x["frontier_imbalance"], x["trade_sign_proxy"]),
            "corr_with_next_trade_sign": safe_corr(x["frontier_imbalance"], x["next_trade_sign_proxy"]),
        },
        {
            "feature": "depth_imbalance",
            "corr_with_current_trade_sign": safe_corr(x["depth_imbalance"], x["trade_sign_proxy"]),
            "corr_with_next_trade_sign": safe_corr(x["depth_imbalance"], x["next_trade_sign_proxy"]),
        },
        {
            "feature": "size_wall_vol_imbalance",
            "corr_with_current_trade_sign": safe_corr(x["size_wall_vol_imbalance"], x["trade_sign_proxy"]),
            "corr_with_next_trade_sign": safe_corr(x["size_wall_vol_imbalance"], x["next_trade_sign_proxy"]),
        },
    ]
    out = pd.DataFrame(rows)

    print_section(f"INTERACTION B: TRADE FLOW VS PRE-TRADE IMBALANCE: {product}")
    print(out.round(6).to_string(index=False))


# ============================================================
# INTERACTION C: BOOK DEPLETION AFTER TRADES
# ============================================================
def print_interaction_C_book_depletion(df, product):
    x = df[df["product"] == product].copy().sort_values(["day", "timestamp"])

    x["next_frontier_ask"] = x.groupby(["product", "day"])["frontier_ask"].shift(-1)
    x["next_frontier_bid"] = x.groupby(["product", "day"])["frontier_bid"].shift(-1)
    x["next_ask_depth"] = x.groupby(["product", "day"])["ask_depth_total"].shift(-1)
    x["next_bid_depth"] = x.groupby(["product", "day"])["bid_depth_total"].shift(-1)

    x["delta_frontier_ask"] = x["next_frontier_ask"] - x["frontier_ask"]
    x["delta_frontier_bid"] = x["next_frontier_bid"] - x["frontier_bid"]
    x["delta_ask_depth"] = x["next_ask_depth"] - x["ask_depth_total"]
    x["delta_bid_depth"] = x["next_bid_depth"] - x["bid_depth_total"]

    buy_rows = x[x["trade_sign_proxy"] > 0]
    sell_rows = x[x["trade_sign_proxy"] < 0]

    out = pd.DataFrame([
        {
            "event": "after_buy_trade",
            "mean_delta_frontier_ask": buy_rows["delta_frontier_ask"].mean(),
            "mean_delta_frontier_bid": buy_rows["delta_frontier_bid"].mean(),
            "mean_delta_ask_depth": buy_rows["delta_ask_depth"].mean(),
            "mean_delta_bid_depth": buy_rows["delta_bid_depth"].mean(),
        },
        {
            "event": "after_sell_trade",
            "mean_delta_frontier_ask": sell_rows["delta_frontier_ask"].mean(),
            "mean_delta_frontier_bid": sell_rows["delta_frontier_bid"].mean(),
            "mean_delta_ask_depth": sell_rows["delta_ask_depth"].mean(),
            "mean_delta_bid_depth": sell_rows["delta_bid_depth"].mean(),
        },
    ])

    print_section(f"INTERACTION C: BOOK DEPLETION / NEXT-STEP BOOK CHANGE AFTER TRADES: {product}")
    print(out.round(6).to_string(index=False))


# ============================================================
# INTERACTION D: BREAK VS BOUNCE AT RESISTANCE / SUPPORT
# ============================================================
def print_interaction_D_break_vs_bounce(df, product):
    x = df[df["product"] == product].copy()

    # Buy-side resistance test:
    # if current trade is buy-ish, does future touch_mid break above current ask size-wall?
    buy_test = x[x["trade_sign_proxy"] > 0].copy()
    buy_test["break_up"] = buy_test[f"future_max_touch_mid_{HORIZON}"] > buy_test["size_wall_ask"]
    buy_test["bounce_or_fail"] = ~buy_test["break_up"]

    # Sell-side support test:
    # if current trade is sell-ish, does future touch_mid break below current bid size-wall?
    sell_test = x[x["trade_sign_proxy"] < 0].copy()
    sell_test["break_down"] = sell_test[f"future_min_touch_mid_{HORIZON}"] < sell_test["size_wall_bid"]
    sell_test["bounce_or_fail"] = ~sell_test["break_down"]

    out = pd.DataFrame([
        {
            "case": "buy_trade_vs_ask_size_wall",
            "count": len(buy_test),
            "break_rate": buy_test["break_up"].mean() if len(buy_test) > 0 else np.nan,
            "bounce_or_fail_rate": buy_test["bounce_or_fail"].mean() if len(buy_test) > 0 else np.nan,
        },
        {
            "case": "sell_trade_vs_bid_size_wall",
            "count": len(sell_test),
            "break_rate": sell_test["break_down"].mean() if len(sell_test) > 0 else np.nan,
            "bounce_or_fail_rate": sell_test["bounce_or_fail"].mean() if len(sell_test) > 0 else np.nan,
        },
    ])

    print_section(f"INTERACTION D: BREAK VS BOUNCE AT RESISTANCE / SUPPORT: {product}")
    print(out.round(6).to_string(index=False))


# ============================================================
# INTERACTION E: TRADE-INFORMED FUTURE TARGETS / ADVERSE SELECTION
# ============================================================
def print_interaction_E_trade_informed_targets(df, product):
    x = df[df["product"] == product].copy()

    # matrix by trade_sign_proxy and wall imbalance bucket
    valid = x.dropna(subset=["trade_sign_proxy", "size_wall_vol_imbalance", f"fwd_touch_mid_change_{HORIZON}"]).copy()
    if len(valid) >= 20:
        valid["wall_imb_bucket"] = pd.qcut(valid["size_wall_vol_imbalance"], q=3, duplicates="drop")
        matrix = valid.groupby(["trade_sign_proxy", "wall_imb_bucket"]).agg(
            count=(f"fwd_touch_mid_change_{HORIZON}", "size"),
            avg_future_change=(f"fwd_touch_mid_change_{HORIZON}", "mean"),
            prob_up=(f"fwd_touch_mid_change_{HORIZON}", lambda s: (s > 0).mean()),
            prob_down=(f"fwd_touch_mid_change_{HORIZON}", lambda s: (s < 0).mean()),
        )
    else:
        matrix = pd.DataFrame({"note": ["Not enough rows for trade-sign x wall-imbalance matrix"]})

    # Adverse selection:
    # If a buyer lifts ask / trade_sign_proxy > 0, adverse if future mid falls.
    # If a seller hits bid / trade_sign_proxy < 0, adverse if future mid rises.
    buy_rows = x[x["trade_sign_proxy"] > 0].copy()
    sell_rows = x[x["trade_sign_proxy"] < 0].copy()

    buy_adverse = (buy_rows[f"fwd_touch_mid_change_{HORIZON}"] < 0).mean() if len(buy_rows) > 0 else np.nan
    sell_adverse = (sell_rows[f"fwd_touch_mid_change_{HORIZON}"] > 0).mean() if len(sell_rows) > 0 else np.nan

    adverse = pd.DataFrame([
        {
            "case": "after_buy_trade",
            "count": len(buy_rows),
            f"adverse_selection_rate_{HORIZON}": buy_adverse,
        },
        {
            "case": "after_sell_trade",
            "count": len(sell_rows),
            f"adverse_selection_rate_{HORIZON}": sell_adverse,
        },
    ])

    print_section(f"INTERACTION E1: TRADE-INFORMED FUTURE TARGETS: {product}")
    print(matrix.round(6).to_string())

    print_section(f"INTERACTION E2: ADVERSE SELECTION AFTER FILL: {product}")
    print(adverse.round(6).to_string(index=False))


# ============================================================
# FAIR VALUE TRACKER SUMMARY
# ============================================================
def print_fair_value_estimators(df, product):
    x = df[df["product"] == product].copy()

    fair_values = [
        "touch_mid",
        "boundary_mid",
        "size_wall_mid",
        "side_vwap_mid",
        "full_book_vwap_center",
        "mid_price",
    ]

    rows = []
    target = x[f"fwd_touch_mid_change_{HORIZON}"]

    for c in fair_values:
        rows.append({
            "estimator": c,
            "mean": x[c].mean(),
            "std": x[c].std(),
            "mean_minus_touch_mid": (x[c] - x["touch_mid"]).mean(),
            "corr_with_future_change": safe_corr(x[c] - x["touch_mid"], target),
        })

    out = pd.DataFrame(rows)

    print_section(f"FAIR VALUE ESTIMATORS: {product}")
    print(out.round(6).to_string(index=False))


# ============================================================
# MAIN
# ============================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    prices = load_prices(script_dir)
    trades = load_trades(script_dir)

    if PRODUCTS is not None:
        prices = prices[prices["product"].isin(PRODUCTS)].copy()
        trades = trades[trades["symbol"].isin(PRODUCTS)].copy()

    book_df = build_book_features(prices)
    trade_agg, raw_trades = build_trade_features(trades, book_df)
    df = merge_book_and_trade(book_df, trade_agg)

    products = sorted(df["product"].dropna().unique())

    print_section("DATA SHAPE")
    print(f"price rows                          : {len(prices)}")
    print(f"trade rows                          : {len(trades)}")
    print(f"products                            : {products}")
    print(f"visible levels used                 : {VISIBLE_LEVELS}")
    print(f"prediction horizon                  : {HORIZON}")

    for product in products:
        print_mid_price_check(df, product)

        print_layer1_static_geometry(df, product)
        print_layer2_dislocations(df, product)
        print_fair_value_estimators(df, product)

        print_layer3_predictive_price(df, product)
        print_predictive_next_trade_direction(df, product)

        print_interaction_A_trade_location(raw_trades, product)
        print_interaction_B_tradeflow_vs_imbalance(df, product)
        print_interaction_C_book_depletion(df, product)
        print_interaction_D_break_vs_bounce(df, product)
        print_interaction_E_trade_informed_targets(df, product)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.max_rows", 300)
    main()