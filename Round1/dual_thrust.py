import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def dual_thrust(df, n=20, k1=1.5, k2=1.5):
    df = df.copy()

    # Indicator comp
    df["HH"] = df["High"].shift(1).rolling(n).max()
    df["LL"] = df["Low"].shift(1).rolling(n).min()
    df["HC"] = df["Close"].shift(1).rolling(n).max()
    df["LC"] = df["Close"].shift(1).rolling(n).min()

    # floating range
    df["Range"] = np.maximum(df["HH"] - df["LC"], df["HC"] - df["LL"])

    # high and low bar
    df["Upper"] = df["Open"] + k1 * df["Range"]
    df["Lower"] = df["Open"] - k2 * df["Range"]

    return df


def add_dual_thrust_signals(df):
    df = df.copy()

    # signal
    df["long_signal"] = df["High"] > df["Upper"]
    df["short_signal"] = df["Low"] < df["Lower"]

    return df


def backtest_dual_thrust(df):
    df = df.copy()

    position = []
    pos = 0

    for i in range(len(df)):
        long_sig = df.iloc[i]["long_signal"]
        short_sig = df.iloc[i]["short_signal"]

        if long_sig and short_sig:
            position.append(pos)
            continue

        if long_sig:
            pos = 1
        elif short_sig:
            pos = -1

        position.append(pos)

    df["position"] = position
    return df


def add_returns(df):
    df = df.copy()

    df["ret"] = df["Close"].pct_change()
    df["strategy_ret"] = df["position"].shift(1) * df["ret"]
    df["equity"] = (1 + df["strategy_ret"].fillna(0)).cumprod()

    return df


def run_dual_thrust_strategy(df, n=20, k1=1.5, k2=1.5):
    df = dual_thrust(df, n=n, k1=k1, k2=k2)
    df = add_dual_thrust_signals(df)
    df = backtest_dual_thrust(df)
    df = add_returns(df)
    return df


if __name__ == "__main__":

    df = pd.read_csv("Data/prices_round_1_day_0.csv", sep=';')

    # If Date：
    # df["Date"] = pd.to_datetime(df["Date"])
    # df = df.sort_values("Date").reset_index(drop=True)

    result = run_dual_thrust_strategy(df, n=20, k1=1.5, k2=1.5)

    print(result[[
        "Open", "High", "Low", "Close",
        "Upper", "Lower",
        "long_signal", "short_signal",
        "position", "equity"
    ]].tail())

    plt.figure(figsize=(12, 6))
    plt.plot(result["equity"])
    plt.title("Dual Thrust Strategy Equity Curve")
    plt.xlabel("Index")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()