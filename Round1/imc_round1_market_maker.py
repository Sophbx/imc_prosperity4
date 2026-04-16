from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_PRODUCTS = ("ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT")


@dataclass(frozen=True)
class ProductConfig:
    rolling_window: int
    edge: float
    inventory_limit: int
    inventory_penalty: float


PRODUCT_CONFIGS: dict[str, ProductConfig] = {
    "ASH_COATED_OSMIUM": ProductConfig(
        rolling_window=60,
        edge=2.0,
        inventory_limit=20,
        inventory_penalty=0.25,
    ),
    "INTARIAN_PEPPER_ROOT": ProductConfig(
        rolling_window=10,
        edge=0.0,
        inventory_limit=20,
        inventory_penalty=0.0,
    ),
}


PRICE_FILE_PATTERN = re.compile(r"prices_round_1_day_(-?\d+)\.csv$")
TRADE_FILE_PATTERN = re.compile(r"trades_round_1_day_(-?\d+)\.csv$")


def infer_day(path: Path, pattern: re.Pattern[str]) -> int:
    match = pattern.search(path.name)
    if match is None:
        raise ValueError(f"Could not infer trading day from {path}")
    return int(match.group(1))


def discover_files(data_dir: Path, prefix: str) -> list[Path]:
    files = sorted(data_dir.glob(f"{prefix}_round_1_day_*.csv"))
    if not files:
        raise FileNotFoundError(f"No files matching {prefix}_round_1_day_*.csv in {data_dir}")
    return files


def load_prices(data_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in discover_files(data_dir, "prices"):
        day = infer_day(path, PRICE_FILE_PATTERN)
        frame = pd.read_csv(path, sep=";")
        frame["day"] = day
        frames.append(frame)

    prices = pd.concat(frames, ignore_index=True)
    prices = prices[prices["product"].isin(TARGET_PRODUCTS)].copy()
    prices = prices.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)
    return prices


def load_trades(data_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in discover_files(data_dir, "trades"):
        day = infer_day(path, TRADE_FILE_PATTERN)
        frame = pd.read_csv(path, sep=";")
        frame["day"] = day
        frames.append(frame)

    trades = pd.concat(frames, ignore_index=True)
    trades = trades[trades["symbol"].isin(TARGET_PRODUCTS)].copy()
    trades = trades.sort_values(["symbol", "day", "timestamp"]).reset_index(drop=True)
    return trades


def build_quote_state(prices: pd.DataFrame, product: str, config: ProductConfig) -> pd.DataFrame:
    quotes = prices.loc[prices["product"] == product].copy()
    quotes = quotes[quotes["bid_price_1"].notna() & quotes["ask_price_1"].notna()].copy()

    quotes["mid_price"] = (quotes["bid_price_1"] + quotes["ask_price_1"]) / 2.0
    top_book_volume = quotes["bid_volume_1"].fillna(0) + quotes["ask_volume_1"].fillna(0)
    quotes["microprice"] = np.where(
        top_book_volume > 0,
        (
            quotes["ask_price_1"] * quotes["bid_volume_1"].fillna(0)
            + quotes["bid_price_1"] * quotes["ask_volume_1"].fillna(0)
        )
        / top_book_volume,
        quotes["mid_price"],
    )
    quotes["rolling_mid"] = quotes.groupby("day")["mid_price"].transform(
        lambda series: series.rolling(config.rolling_window, min_periods=1).mean()
    )
    quotes["fair_value"] = 0.5 * quotes["rolling_mid"] + 0.5 * quotes["microprice"]

    return quotes[
        [
            "day",
            "timestamp",
            "bid_price_1",
            "ask_price_1",
            "mid_price",
            "fair_value",
        ]
    ].copy()


def simulate_product(
    product: str,
    quotes: pd.DataFrame,
    trades: pd.DataFrame,
    config: ProductConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    product_trades = trades.loc[trades["symbol"] == product].copy()
    merged = product_trades.merge(quotes, on=["day", "timestamp"], how="left")
    merged = merged.sort_values(["day", "timestamp"]).reset_index(drop=True)

    day_results: list[dict[str, float | int | str]] = []
    fill_records: list[dict[str, float | int | str]] = []

    for day, day_trades in merged.groupby("day", sort=True):
        position = 0
        cash = 0.0
        buy_qty = 0
        sell_qty = 0
        fill_count = 0
        last_mid = np.nan

        for row in day_trades.itertuples(index=False):
            if pd.isna(row.bid_price_1) or pd.isna(row.ask_price_1) or pd.isna(row.fair_value):
                continue

            last_mid = float(row.mid_price)
            reservation_price = float(row.fair_value) - config.inventory_penalty * position

            quote_bid = position < config.inventory_limit and float(row.bid_price_1) <= reservation_price - config.edge
            quote_ask = position > -config.inventory_limit and float(row.ask_price_1) >= reservation_price + config.edge

            if quote_bid and float(row.price) == float(row.bid_price_1):
                quantity = min(int(row.quantity), config.inventory_limit - position)
                if quantity > 0:
                    position += quantity
                    cash -= quantity * float(row.bid_price_1)
                    buy_qty += quantity
                    fill_count += 1
                    fill_records.append(
                        {
                            "product": product,
                            "day": int(day),
                            "timestamp": int(row.timestamp),
                            "side": "BUY",
                            "price": float(row.bid_price_1),
                            "quantity": quantity,
                            "position_after": position,
                        }
                    )

            if quote_ask and float(row.price) == float(row.ask_price_1):
                quantity = min(int(row.quantity), config.inventory_limit + position)
                if quantity > 0:
                    position -= quantity
                    cash += quantity * float(row.ask_price_1)
                    sell_qty += quantity
                    fill_count += 1
                    fill_records.append(
                        {
                            "product": product,
                            "day": int(day),
                            "timestamp": int(row.timestamp),
                            "side": "SELL",
                            "price": float(row.ask_price_1),
                            "quantity": quantity,
                            "position_after": position,
                        }
                    )

        if np.isnan(last_mid):
            last_mid = 0.0

        mtm_pnl = cash + position * last_mid
        day_results.append(
            {
                "product": product,
                "day": int(day),
                "buy_qty": buy_qty,
                "sell_qty": sell_qty,
                "fills": fill_count,
                "end_position": position,
                "close_mid": last_mid,
                "mark_to_market_pnl": mtm_pnl,
            }
        )

    return pd.DataFrame(day_results), pd.DataFrame(fill_records)


def run_backtest(data_dir: Path, show_fills: int) -> None:
    prices = load_prices(data_dir)
    trades = load_trades(data_dir)

    summaries: list[pd.DataFrame] = []
    sample_fills: list[pd.DataFrame] = []

    for product in TARGET_PRODUCTS:
        config = PRODUCT_CONFIGS[product]
        quote_state = build_quote_state(prices, product, config)
        product_summary, product_fills = simulate_product(product, quote_state, trades, config)
        summaries.append(product_summary)
        if show_fills > 0 and not product_fills.empty:
            sample_fills.append(product_fills.head(show_fills))

    summary = pd.concat(summaries, ignore_index=True)
    total = (
        summary.groupby("product", as_index=False)[["buy_qty", "sell_qty", "fills", "mark_to_market_pnl"]]
        .sum()
        .sort_values("product")
    )

    print("Per-day backtest summary")
    print(summary.to_string(index=False))
    print()
    print("Product totals")
    print(total.to_string(index=False))
    print()
    print(f"Grand total PnL: {summary['mark_to_market_pnl'].sum():,.2f}")

    if sample_fills:
        print()
        print("Sample fills")
        print(pd.concat(sample_fills, ignore_index=True).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest a passive market-making strategy on IMC Prosperity Round 1 CSV files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"C:\Users\gusiy\OneDrive\Desktop\imc_prosperity4\Round1\Data"),
        help="Directory containing prices_round_1_day_*.csv and trades_round_1_day_*.csv",
    )
    parser.add_argument(
        "--show-fills",
        type=int,
        default=10,
        help="Number of sample fills to print per product",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_backtest(args.data_dir, show_fills=args.show_fills)
