"""Generate analysis_options.ipynb from a deterministic cell list.

Run me: `python3 build_options_notebook.py`
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(("md", s))
py = lambda s: C.append(("py", s))

# ---------- Part A ----------
md("""# Round 3 — Options Analysis

Analysis of the 10 VEV vouchers and their underlying `VELVETFRUIT_EXTRACT`.

See `DESIGN.md` for the notebook outline.""")

md("## A. Setup & load")

py("""import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import helpers

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 100)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")

sns.set_theme(context="notebook", style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 4)""")

py("""DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "Data"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.getcwd(), "output"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRICE_FILES = [
    "prices_round_3_day_0.csv",
    "prices_round_3_day_1.csv",
    "prices_round_3_day_2.csv",
]
TRADE_FILES = [
    "trades_round_3_day_0.csv",
    "trades_round_3_day_1.csv",
    "trades_round_3_day_2.csv",
]

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{K}" for K in STRIKES]
UNDERLYING = "VELVETFRUIT_EXTRACT"

IV_DOWNSAMPLE = 10  # compute IV on every Nth row""")

py("""prices_all = helpers.load_prices(DATA_DIR, PRICE_FILES)

# Underlying path + voucher rows
u = prices_all[prices_all["product"] == UNDERLYING].copy()
v = prices_all[prices_all["product"].isin(VOUCHERS)].copy()

# Attach strike integer + TTE years
v["strike"] = v["product"].str.replace("VEV_", "", regex=False).astype(int)
v["tte_years"] = [helpers.tte_years(int(d), int(t))
                  for d, t in zip(v["day"], v["timestamp"])]

# Attach underlying mid on (day, timestamp) — prices are tick-aligned
u_key = u[["day", "timestamp", "mid_price"]].rename(columns={"mid_price": "underlying_mid"})
v = v.merge(u_key, on=["day", "timestamp"], how="left")
v["moneyness"] = v["underlying_mid"] / v["strike"]

print(f"underlying rows: {len(u):>7,}")
print(f"voucher rows   : {len(v):>7,}")""")

md("### A.5 Data quality summary")

py("""qc = (v.groupby(["strike", "day"])
        .agg(rows=("timestamp", "size"),
             nan_mid=("mid_price", lambda s: s.isna().sum()),
             zero_mid=("mid_price", lambda s: (s == 0).sum()),
             mean_quoted_size=("bid_volume_1", lambda s: s.fillna(0).mean()))
        .reset_index())
display(qc)""")

# ---------- Part B ----------
md("## B. Price behaviour")

py("""fig, ax = plt.subplots(figsize=(12, 4))
for d, color in zip(sorted(u["day"].unique()), sns.color_palette("tab10")):
    s = u[u["day"] == d].sort_values("timestamp")
    ax.plot(s["timestamp"].to_numpy() + d * helpers.TIMESTAMPS_PER_DAY,
            s["mid_price"].to_numpy(), label=f"day {d}", color=color, lw=0.8)
ax.set_title(f"{UNDERLYING}: mid_price across 3 days")
ax.legend()
plt.tight_layout()
plt.show()""")

py("""fig, axes = plt.subplots(3, 4, figsize=(18, 9), sharex=False)
for ax, K in zip(axes.ravel(), STRIKES):
    s_v = v[v["strike"] == K].sort_values(["day", "timestamp"])
    for d, color in zip(sorted(s_v["day"].unique()), sns.color_palette("tab10")):
        sd = s_v[s_v["day"] == d]
        ax.plot(sd["timestamp"].to_numpy() + d * helpers.TIMESTAMPS_PER_DAY,
                sd["mid_price"].to_numpy(), color=color, lw=0.7, label=f"d{d}")
    ax.set_title(f"VEV_{K}")
# turn off unused axes
for ax in axes.ravel()[len(STRIKES):]:
    ax.axis("off")
plt.tight_layout()
plt.show()""")

py("""per_strike = (v.groupby("strike")
                .agg(rows=("timestamp", "size"),
                     mean_mid=("mid_price", "mean"),
                     mean_spread=("mid_price", lambda s: (s.std() if len(s) else np.nan)),
                     mean_quoted_size_bid=("bid_volume_1", lambda s: s.fillna(0).mean()),
                     mean_quoted_size_ask=("ask_volume_1", lambda s: s.fillna(0).mean()))
                .reset_index())
# Recompute mean_spread more precisely as ask_price_1 - bid_price_1
spread = (v.assign(spread=v["ask_price_1"] - v["bid_price_1"])
            .groupby("strike")["spread"].mean().rename("mean_spread"))
per_strike = per_strike.drop(columns="mean_spread").merge(spread, on="strike")
display(per_strike)""")

# ---------- Part C: Voucher <-> underlying ----------
md("## C. Voucher ↔ underlying relationship")

md("""If a voucher is a real option, its mid should move with the underlying.
The scatter slope per strike is an empirical delta.

Moneyness = underlying / strike. Moneyness < 1 = out-of-money, > 1 = in-the-money.""")

py("""fig, axes = plt.subplots(3, 4, figsize=(18, 11))
slopes = {}
for ax, K in zip(axes.ravel(), STRIKES):
    s = v[v["strike"] == K].dropna(subset=["mid_price", "underlying_mid"])
    if len(s) < 100:
        ax.axis("off"); continue
    ax.scatter(s["underlying_mid"], s["mid_price"], s=2, alpha=0.3)
    slope, intercept = np.polyfit(s["underlying_mid"].to_numpy(float),
                                  s["mid_price"].to_numpy(float), 1)
    slopes[K] = slope
    xs = np.linspace(s["underlying_mid"].min(), s["underlying_mid"].max(), 50)
    ax.plot(xs, slope * xs + intercept, color="red", lw=1.0)
    mean_money = s["moneyness"].mean()
    ax.set_title(f"VEV_{K}: slope={slope:.3f}  moneyness≈{mean_money:.3f}")
for ax in axes.ravel()[len(STRIKES):]:
    ax.axis("off")
plt.tight_layout()
plt.show()
print("Empirical deltas:", {k: round(s, 4) for k, s in slopes.items()})""")

py("""ROLL = 500
fig, ax = plt.subplots(figsize=(13, 4))
for K in STRIKES:
    s = v[v["strike"] == K].sort_values(["day", "timestamp"]).copy()
    s["voucher_ret"] = s.groupby("day")["mid_price"].diff()
    s["underlying_ret"] = s.groupby("day")["underlying_mid"].diff()
    rc = (s[["voucher_ret", "underlying_ret"]]
            .rolling(ROLL).corr().unstack()["voucher_ret"]["underlying_ret"])
    ax.plot(rc.to_numpy(), label=f"VEV_{K}", lw=0.8)
ax.set_title(f"Rolling-{ROLL} correlation of tick returns vs underlying")
ax.legend(ncol=5, fontsize=8, loc="lower center")
plt.tight_layout()
plt.show()""")

py("""viols = v.assign(intrinsic=np.maximum(v["underlying_mid"] - v["strike"], 0))
viols = viols[viols["mid_price"] < viols["intrinsic"] - 1e-6]
print(f"intrinsic-floor violations: {len(viols)} rows "
      f"({100*len(viols)/max(len(v), 1):.3f}%)")
if len(viols):
    display(viols.groupby("strike").size().rename("violations"))""")

# emit
for kind, src in C:
    if kind == "md":
        nb.cells.append(nbf.v4.new_markdown_cell(src))
    else:
        nb.cells.append(nbf.v4.new_code_cell(src))

nbf.write(nb, "analysis_options.ipynb")
print(f"wrote analysis_options.ipynb with {len(nb.cells)} cells")
