"""Plot v10 vs v12_merged cumulative PnL at three time horizons.

Reads activity.csv (tick × product) for each day, sums profit_and_loss across
all products per tick, and concatenates D+2 → D+3 → D+4 with continuous tick
indexing (0..30000).

Outputs three PNGs into ./Submission:
  pnl_v10_vs_v12_1000ticks.png   — first 1000 ticks
  pnl_v10_vs_v12_1day.png        — first day  (10000 ticks)
  pnl_v10_vs_v12_3days.png       — all three days (30000 ticks)
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).parent

V10 = HERE / "v10_pruned"  / "rust_runs"
V12 = HERE / "v12_merged"  / "rust_runs"

DAYS = [
    ("D+2", "pebbles_v10-round5-day+2", "pebbles_v12-round5-day+2"),
    ("D+3", "pebbles_v10-round5-day+3", "pebbles_v12-round5-day+3"),
    ("D+4", "pebbles_v10-round5-day+4", "pebbles_v12-round5-day+4"),
]


def load_pnl(run_dir: Path) -> pd.Series:
    """Total PnL per tick (timestamp), summed across products."""
    df = pd.read_csv(run_dir / "activity.csv", sep=";")
    s = df.groupby("timestamp")["profit_and_loss"].sum().sort_index()
    return s


def build_concatenated(root: Path, day_dirs: list[str]) -> pd.Series:
    """Concatenate the 3 days, carrying end-of-day PnL forward as a baseline."""
    parts: list[pd.Series] = []
    cumulative_offset = 0.0
    tick_offset = 0
    for d in day_dirs:
        s = load_pnl(root / d)
        # rust BT timestamps within a day are tick*100 (0,100,...,999900); normalize to 0..9999
        s.index = (s.index // 100).astype(int)
        # Daily PnL is mark-to-market starting from 0 each day. Add cumulative
        # offset so the curve continues smoothly across day boundaries.
        s_shifted = s + cumulative_offset
        s_shifted.index = s_shifted.index + tick_offset
        parts.append(s_shifted)
        cumulative_offset += float(s.iloc[-1])
        tick_offset += 10000
    return pd.concat(parts)


print("loading v10…")
v10 = build_concatenated(V10, [d[1] for d in DAYS])
print(f"  v10 ticks={len(v10)}  final={v10.iloc[-1]:,.0f}")

print("loading v12…")
v12 = build_concatenated(V12, [d[2] for d in DAYS])
print(f"  v12 ticks={len(v12)}  final={v12.iloc[-1]:,.0f}")


def plot(xmax: int, title: str, fname: str):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(v10.index, v10.values, label=f"v10_pruned  ({v10.iloc[-1]:,.0f})",
            color="#1f77b4", linewidth=1.4)
    ax.plot(v12.index, v12.values, label=f"v12_merged  ({v12.iloc[-1]:,.0f})",
            color="#d62728", linewidth=1.4)
    # day boundaries
    for x in (10000, 20000):
        if x <= xmax:
            ax.axvline(x, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlim(0, xmax)
    # Tighten y-limits to data within the visible window
    sub10 = v10.loc[v10.index <= xmax]
    sub12 = v12.loc[v12.index <= xmax]
    lo = min(sub10.min(), sub12.min())
    hi = max(sub10.max(), sub12.max())
    pad = (hi - lo) * 0.05 or 1
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("tick")
    ax.set_ylabel("cumulative PnL (seashells)")
    ax.set_title(title)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    out = HERE / fname
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


plot(1000,  "v10 vs v12 — first 1000 ticks (D+2 open)",     "pnl_v10_vs_v12_1000ticks.png")
plot(10000, "v10 vs v12 — first day (D+2, 10000 ticks)",   "pnl_v10_vs_v12_1day.png")
plot(30000, "v10 vs v12 — all three days (D+2 → D+4)",      "pnl_v10_vs_v12_3days.png")

print("done.")
