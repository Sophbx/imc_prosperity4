"""Plot v12 vs v14_robust vs v14_research cumulative PnL across all 3 days."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).parent

VARIANTS = [
    ("v12_merged",   "v12 (current)",      "#1f77b4", "pebbles_v12"),
    ("v14_robust",   "v14_robust",         "#2ca02c", "v14_robust"),
    ("v14_research", "v14_research",       "#d62728", "v14_verified"),
]


def load_concat(folder: str, prefix: str) -> pd.Series:
    parts: list[pd.Series] = []
    cum_offset = 0.0
    tick_offset = 0
    for d in [2, 3, 4]:
        run_dir = HERE / folder / "rust_runs" / f"{prefix}-round5-day+{d}"
        df = pd.read_csv(run_dir / "activity.csv", sep=";")
        s = df.groupby("timestamp")["profit_and_loss"].sum().sort_index()
        s.index = (s.index // 100).astype(int)
        s_shifted = s + cum_offset
        s_shifted.index = s_shifted.index + tick_offset
        parts.append(s_shifted)
        cum_offset += float(s.iloc[-1])
        tick_offset += 10000
    return pd.concat(parts)


curves = {}
for folder, label, color, prefix in VARIANTS:
    print(f"loading {folder}…")
    curves[label] = (load_concat(folder, prefix), color)
    print(f"  final: {curves[label][0].iloc[-1]:,.0f}")


def plot(xmax: int, title: str, fname: str):
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, (s, color) in curves.items():
        sub = s.loc[s.index <= xmax]
        ax.plot(sub.index, sub.values,
                label=f"{label}  ({s.iloc[-1]:,.0f})",
                color=color, linewidth=1.4)
    for x in (10000, 20000):
        if x <= xmax:
            ax.axvline(x, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("tick (concatenated D+2 → D+3 → D+4)")
    ax.set_ylabel("cumulative PnL (seashells)")
    ax.set_title(title)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / fname, dpi=140)
    plt.close(fig)
    print(f"  wrote {fname}")


plot(1000,  "v12 vs v14_robust vs v14_research — first 1000 ticks",
     "pnl_v12_v14robust_v14research_1000ticks.png")
plot(10000, "v12 vs v14_robust vs v14_research — first day (D+2)",
     "pnl_v12_v14robust_v14research_1day.png")
plot(30000, "v12 vs v14_robust vs v14_research — all three days",
     "pnl_v12_v14robust_v14research_3days.png")

print("done.")
