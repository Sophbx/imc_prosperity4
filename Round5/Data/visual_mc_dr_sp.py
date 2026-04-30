from operator import lt

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("Round5/vs_day4")
OUT_DIR.mkdir(exist_ok=True)

GROUPS = {
    "organic_microchip": "MICROCHIP_",
    "domestic_robotics": "ROBOT_",
    "snack_pack": "SNACKPACK_",
}

BASE_DIR = Path(__file__).resolve().parent

PRICE_FILE = BASE_DIR / "prices_round_5_day_4.csv"
TRADE_FILE = BASE_DIR / "trades_round_5_day_4.csv"

print("Script directory:", BASE_DIR)
print("Price file exists?", PRICE_FILE.exists())
print("Trade file exists?", TRADE_FILE.exists())

prices = pd.read_csv(PRICE_FILE, sep=";")

trades = pd.read_csv(TRADE_FILE, sep=";")

def filter_group(df, col, prefix):
    return df[df[col].str.startswith(prefix)].copy()

for group_name, prefix in GROUPS.items():
    p = filter_group(prices, "product", prefix)
    t = filter_group(trades, "symbol", prefix)

    p["spread"] = p["ask_price_1"] - p["bid_price_1"]
    p["book_volume"] = (
        p["bid_volume_1"].fillna(0)
        + p["bid_volume_2"].fillna(0)
        + p["bid_volume_3"].fillna(0)
        + p["ask_volume_1"].fillna(0)
        + p["ask_volume_2"].fillna(0)
        + p["ask_volume_3"].fillna(0)
    )

    # Save cleaned extracted data
    p.to_csv(OUT_DIR / f"{group_name}_prices.csv", index=False)
    t.to_csv(OUT_DIR / f"{group_name}_trades.csv", index=False)

    # 1. Mid-price movement
    plt.figure(figsize=(12, 6))
    for product, sub in p.groupby("product"):
        plt.plot(sub["timestamp"], sub["mid_price"], label=product)

    plt.title(f"{group_name}: Mid Price Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Mid Price")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{group_name}_mid_price.png", dpi=200)
    plt.show()

    # 2. Normalized mid-price movement
    plt.figure(figsize=(12, 6))
    for product, sub in p.groupby("product"):
        sub = sub.sort_values("timestamp")
        normalized = sub["mid_price"] / sub["mid_price"].iloc[0]
        plt.plot(sub["timestamp"], normalized, label=product)

    plt.title(f"{group_name}: Normalized Mid Price")
    plt.xlabel("Timestamp")
    plt.ylabel("Price / Initial Price")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{group_name}_normalized_mid_price.png", dpi=200)
    plt.show()

    # 3. Bid-ask spread
    plt.figure(figsize=(12, 6))
    for product, sub in p.groupby("product"):
        plt.plot(sub["timestamp"], sub["spread"], label=product)

    plt.title(f"{group_name}: Bid-Ask Spread")
    plt.xlabel("Timestamp")
    plt.ylabel("Ask 1 - Bid 1")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{group_name}_spread.png", dpi=200)
    plt.show()

    # 4. Top-of-book volume bands: bid-side vs ask-side dots
    p["bid_top3_volume"] = (
        p["bid_volume_1"].fillna(0)
        + p["bid_volume_2"].fillna(0)
        + p["bid_volume_3"].fillna(0)
    )

    p["ask_top3_volume"] = (
        p["ask_volume_1"].fillna(0)
        + p["ask_volume_2"].fillna(0)
        + p["ask_volume_3"].fillna(0)
    )

    plt.figure(figsize=(12, 6))

    for product, sub in p.groupby("product"):
        sub = sub.sort_values("timestamp")

        plt.scatter(
            sub["timestamp"],
            sub["ask_top3_volume"],
            s=8,
            alpha=0.45,
            label=f"{product} ask volume"
        )

        plt.scatter(
            sub["timestamp"],
            -sub["bid_top3_volume"],
            s=8,
            alpha=0.45,
            label=f"{product} bid volume"
        )

    plt.axhline(0, linewidth=1)

    plt.title(f"{group_name}: Top-3 Book Volume Bands")
    plt.xlabel("Timestamp")
    plt.ylabel("Volume: ask above 0, bid below 0")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{group_name}_volume_bands.png", dpi=200)
    plt.show()

    # 5. Trade prices
    if len(t) > 0:
        plt.figure(figsize=(12, 6))
        for symbol, sub in t.groupby("symbol"):
            plt.scatter(sub["timestamp"], sub["price"], s=sub["quantity"] * 2, alpha=0.5, label=symbol)

        plt.title(f"{group_name}: Executed Trades")
        plt.xlabel("Timestamp")
        plt.ylabel("Trade Price")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{group_name}_trades.png", dpi=200)
        plt.show()

    # 6. Summary table
    summary = p.groupby("product").agg(
        avg_mid_price=("mid_price", "mean"),
        std_mid_price=("mid_price", "std"),
        min_mid_price=("mid_price", "min"),
        max_mid_price=("mid_price", "max"),
        avg_spread=("spread", "mean"),
        avg_book_volume=("book_volume", "mean"),
        pnl_end=("profit_and_loss", "last"),
    )

    print("\n" + "=" * 80)
    print(group_name.upper())
    print(summary.sort_values("std_mid_price", ascending=False))