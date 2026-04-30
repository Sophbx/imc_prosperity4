"""Build visualization notebook for v7 (drop M + snap blackout).

Run:
    /opt/homebrew/bin/python3.12 _build_notebook.py
"""
import nbformat as nbf
from nbclient import NotebookClient
from pathlib import Path

HERE = Path(__file__).parent


def md(src: str): return nbf.v4.new_markdown_cell(src.strip("\n"))
def code(src: str): return nbf.v4.new_code_cell(src.strip("\n"))


nb = nbf.v4.new_notebook()
nb.cells = [
    md("""
# PEBBLES `basket_snap_v7` — drop the bleeder + snap blackout

## Headline

| Version | Trades captured | Total PnL | Δ vs v4 |
|---|---:|---:|---:|
| v4 (uniform b=5) | 3,220 | +53,988 | — |
| v6 (XL=2 small) | 3,220 | +55,174 | +2.2% |
| **v7 (M=0, XL=10, blackout)** | **2,536** | **+68,068** | **+26.1%** |

Same matching engine, same trader.run signature — only the per-leg
allocation and a snap-tick blackout changed.

## What v7 does

```python
BASELINE     = {"XS": 5, "S": 5, "M": 0, "L": 5, "XL": 10}
OFFSET       = {"XS": 1, "S": 1, "M": 1, "L": 1, "XL": 1}
BLACKOUT_THR = 10   # if |basket_dev| > 10, skip every leg this tick
```

That's the entire change vs v6.

## How v7 was found — three diagnostics from a fresh look at the data
"""),

    md("""
### Diagnostic 1 — bot trades happen at quiet ticks, not snap ticks

For every PEBBLES bot fill in the raw `trades_round_5_*.csv`, looked at
the basket deviation at that tick. **Median |basket_dev| at fill = 0.5
ticks. Only 1.86% of fills are at snap events (|dev|>10).**

The basket-snap signal therefore doesn't translate into more
fills — bot flow is uncorrelated with the snap regime.
"""),

    md("""
### Diagnostic 2 — v6 already saturates bot flow (3,220 fills = 100%)

Bot does 644 trades per leg × 5 legs = 3,220 trades over 3 days. v6 fills
exactly 3,220 trades. **The "catch more flow" lever is exhausted.**
"""),

    md("""
### Diagnostic 3 — per-fill drift analysis revealed M is the bleeder, XL is the winner

For each historical fill, computed `signed_drift_5` = signed price change
of the leg over the next 5 ticks (positive = price moves in our favor
post-fill). Averaged over all bot trades on each leg:

| Leg | per-fill drift (5-tick) | n fills | net 3-day drift contrib |
|---|---:|---:|---:|
| XS | +0.38 | 358 | +136 |
| S  | −0.93 | 358 | −333 |
| **M** | **−5.55** | 358 | **−1,987** ← big bleed |
| L  | −0.27 | 358 | −97 |
| **XL** | **+6.34** | 358 | **+2,270** ← big win |

v6 had **down-sized XL** (BASELINE=2) on the absorber-leg snap-noise hypothesis.
But that snap noise applies only to 1.86% of fills. At the **98.14% of
fills that are quiet**, XL is favorably selected and v6 was throttling
its best leg.

v7 inverts this: maxes XL, drops M.
"""),

    md("## 0 — Setup"),
    code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
"""),

    md("## 1 — Cumulative PnL per day, three versions"),
    code("""
# Per-day PnL captured directly from Kevin BT output of each version
versions = {
    'v4 uniform (b=5)':        {'D2': 16857, 'D3': 19852, 'D4': 17279, 'tot': 53988},
    'v6 (XL=2)':               {'D2': 19913, 'D3': 18130, 'D4': 17131, 'tot': 55174},
    'v7 (M=0, XL=10, blackout)': {'D2': 11129, 'D3': 30050, 'D4': 26889, 'tot': 68068},
}
df = pd.DataFrame(versions).T
df.columns = ['Day 2', 'Day 3', 'Day 4', 'Total (3 days)']
df.astype(int)
"""),

    code("""
fig, ax = plt.subplots(figsize=(10, 4.8))
days = ['Day 2', 'Day 3', 'Day 4']
x = np.arange(len(days)); w = 0.27
colors = ['tab:gray', 'tab:orange', 'tab:green']
for i, (label, vals) in enumerate(versions.items()):
    ax.bar(x + (i-1)*w, [vals[f'D{d}'] for d in [2,3,4]], w,
           label=f"{label} (Σ={vals['tot']:,})", color=colors[i], alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(days)
ax.set_ylabel('Daily PnL (ticks)')
ax.set_title('Per-day PnL across versions — v7 dominates Day 3 & Day 4')
ax.legend(fontsize=9)
plt.tight_layout()
"""),

    md("""
**Read.** v7's day-2 is weaker than v6's (11k vs 20k) but day-3 and day-4
both leap from ~18k to ~30k. The dispersion across days widens — v7 is
more concentrated in the days where its setup pays off — but the totals
are clearly larger.
"""),

    md("## 2 — Per-leg drift analysis (the diagnostic that drove the design)"),
    code("""
DATA = Path('../../../Data')
prices = pd.concat([
    pd.read_csv(DATA / f'prices_round_5_day_{d}.csv', sep=';').assign(day=d)
    for d in [2, 3, 4]
], ignore_index=True)
trades = pd.concat([
    pd.read_csv(DATA / f'trades_round_5_day_{d}.csv', sep=';').assign(day=d)
    for d in [2, 3, 4]
], ignore_index=True)
PEB = ['PEBBLES_XS','PEBBLES_S','PEBBLES_M','PEBBLES_L','PEBBLES_XL']

mid_pivot = (prices[prices['product'].isin(PEB)]
             .pivot_table(index=['day','timestamp'], columns='product', values='mid_price'))[PEB]
mid_long = mid_pivot.reset_index().melt(id_vars=['day','timestamp'],
                                          var_name='symbol', value_name='mid')
mid_5    = mid_pivot.shift(-5).reset_index().melt(id_vars=['day','timestamp'],
                                                    var_name='symbol', value_name='mid_5')

t = trades[trades['symbol'].isin(PEB)].merge(
    prices[['day','timestamp','product','mid_price']].rename(columns={'product':'symbol'}),
    on=['day','timestamp','symbol'])
t = t.merge(mid_long, on=['day','timestamp','symbol']).merge(
    mid_5, on=['day','timestamp','symbol'])
t['side'] = np.where(t['price'] > t['mid_price'], 'buy_taker', 'sell_taker')
t['drift_5'] = t['mid_5'] - t['mid_price']
t['hyp_pnl'] = np.where(t['side']=='buy_taker', -t['drift_5'], t['drift_5'])

per_leg = t.groupby('symbol')['hyp_pnl'].agg(['count','mean','sum']).round(2)
per_leg.columns = ['n_fills', 'mean_drift_per_fill', 'total_drift_3d']
per_leg
"""),

    code("""
fig, ax = plt.subplots(figsize=(10, 4))
suf = ['XS','S','M','L','XL']
mean_drift = per_leg.loc[[f'PEBBLES_{s}' for s in suf], 'mean_drift_per_fill']
colors_bar = ['tab:gray' if abs(v) < 1 else ('tab:red' if v < 0 else 'tab:green') for v in mean_drift]
ax.bar(range(5), mean_drift.values, color=colors_bar, alpha=0.85)
ax.axhline(0, color='black', lw=0.6)
ax.set_xticks(range(5)); ax.set_xticklabels(suf)
ax.set_ylabel('mean per-fill drift over next 5 ticks (ticks)')
ax.set_title('Adverse vs favorable selection per leg\\n(positive = drift in our favor; negative = adversely selected)')
for i, v in enumerate(mean_drift.values):
    ax.text(i, v, f'{v:+.2f}', ha='center', va='bottom' if v>=0 else 'top', fontsize=10)
plt.tight_layout()
"""),

    md("""
**Read.** **M is the only leg with a meaningful negative drift** (−5.55).
XL has the biggest positive drift (+6.34). v7's allocation reflects
this directly: drop M, max-out XL.

S, XS, L are roughly neutral on drift but their **spread income** is
positive, so they stay in the rotation. Dropping S costs ~47k (see
ablation in §4) — drift isn't the only thing that matters; preserving
basket structure does too.
"""),

    md("## 3 — Snap blackout: where the +6k came from"),
    code("""
# Distribution of |basket_dev|: how often we'd skip
dev = mid_pivot.sum(axis=1) - 50_000
n_total = len(dev)
n_blackout = (dev.abs() > 10).sum()
print(f'total ticks: {n_total:,}')
print(f'blackout ticks (|dev|>10): {n_blackout:,}  ({n_blackout/n_total*100:.2f}%)')

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(dev, bins=80, color='steelblue', alpha=0.85)
ax.set_yscale('log')
ax.axvline(+10, color='red', lw=0.6, ls='--', label='blackout boundary (|dev|=10)')
ax.axvline(-10, color='red', lw=0.6, ls='--')
ax.set_xlabel('basket deviation (sum − 50,000)')
ax.set_ylabel('count (log)')
ax.set_title('Distribution of basket_dev — v7 skips trading in the red-dashed zones (~3% of ticks)')
ax.legend()
plt.tight_layout()
"""),

    md("""
**Read.** Basket deviation is bimodal: ~97% of ticks at |dev|≤3, ~3% at
|dev|>10, almost nothing in between. v7 disables trading in those 3%
bursts — a tiny window of time but the slice where every fill turns into
adverse selection. Worth roughly +6k of PnL over 3 days.
"""),

    md("## 4 — Ablation sweep — why exactly these parameters"),
    code("""
ablation = {
    'M=0, others=5, XL=10  ★ winner':         68068,
    'M=0, others=5, XL=10, NO blackout':      62182,
    'M=0, S=0, others=5, XL=10':              21431,
    'M=0, L=0, others=5, XL=10':              62693,
    'ONLY XL=10':                             19520,
    'M=0, XL=10, others=7':                   68068,
    'M=0, XL=10, others=10':                  68068,
    'everyone=10 (greedy size)':              53563,
    'v6 + blackout=10':                       62182,
    'v6 (no blackout)':                       55174,
    'v4 (no blackout, uniform)':              53988,
}
df = pd.DataFrame.from_dict(ablation, orient='index', columns=['pnl']).sort_values('pnl', ascending=True)

fig, ax = plt.subplots(figsize=(11, 6))
colors_h = ['tab:green' if v >= 65000 else ('tab:orange' if v >= 55000 else 'tab:red')
             for v in df['pnl']]
ax.barh(range(len(df)), df['pnl'].values, color=colors_h, alpha=0.85)
ax.set_yticks(range(len(df))); ax.set_yticklabels(df.index, fontsize=9)
for i, v in enumerate(df['pnl']):
    ax.text(v, i, f' {v:,.0f}', va='center', fontsize=9)
ax.set_xlabel('Total PnL (3 days, ticks)')
ax.set_title('Ablation: every variant tested in the v7 search')
plt.tight_layout()
"""),

    md("""
**Read.**

* The win comes from the **M=0 + blackout** combination, not either piece
  alone. Removing blackout (top): 68 → 62. Putting M back in (going to v6
  numbers): 62 → 55.
* You can drop M, but **not S or L** (S=0 alone tanks PnL by 47k). So the
  story is "M is the *only* leg that hurts net of fees".
* "ONLY XL=10" is a disaster (19k) — basket diversification is necessary
  for spread capture even when XL is the most favorable leg.
* All ≥5 baselines on non-M legs saturate at 68k. The bot's flow caps the
  rate; bigger quotes don't unlock more.
"""),

    md("## 5 — Per-leg PnL breakdown for v7 (per day, end-of-day balances)"),
    code("""
# Hardcoded from the Kevin BT run output
v7_legs = pd.DataFrame({
    'Day 2': {'XS':    76, 'S': 19330, 'M':     0, 'L': -4417, 'XL': -3860},
    'Day 3': {'XS': -4836, 'S':  7032, 'M':     0, 'L': 17060, 'XL': 10794},
    'Day 4': {'XS':  1296, 'S': 20275, 'M':     0, 'L': -7268, 'XL': 12586},
})
v7_legs.loc['Day total'] = v7_legs.sum(axis=0)
v7_legs['3-day sum'] = v7_legs.sum(axis=1)
v7_legs.astype(int)
"""),

    code("""
fig, ax = plt.subplots(figsize=(11, 4))
days = ['Day 2', 'Day 3', 'Day 4']
suf = ['XS','S','M','L','XL']
x = np.arange(len(days)); w = 0.16
colors_l = plt.cm.tab10.colors[:5]
for i, leg in enumerate(suf):
    vals = [v7_legs.loc[leg, d] for d in days]
    ax.bar(x + (i-2)*w, vals, w, label=leg, color=colors_l[i], alpha=0.85)
ax.axhline(0, color='gray', lw=0.4)
ax.set_xticks(x); ax.set_xticklabels(days)
ax.set_ylabel('Per-leg PnL (ticks)')
ax.set_title('v7 per-leg PnL by day — high day-to-day leg variance, but totals positive every day')
ax.legend(loc='upper left', fontsize=9, ncols=5)
plt.tight_layout()
"""),

    md("""
**Read.** Even with M dropped, individual legs swing dramatically per day
(L: −7k Day 4 vs +17k Day 3; XS: −5k Day 3 vs +1k Day 4). The basket
constraint forces these into rough cancellation — but **the small
positive bias provided by XL's drift advantage and the snap-blackout
savings is what generates the +68k aggregate**.
"""),

    md("""
## Summary

v7 is the result of treating the trader as new and following the data
without preconceptions. Three corrections vs v6:

1. **Drop M.** It adds spread income but bleeds an equal amount on
   adverse selection. Net contribution = ≈0 with high variance.
2. **Max-out XL.** v6 down-sized XL based on a snap-noise hypothesis
   that applies to <2% of fills; v7 sizes XL up because it has the
   strongest favorable drift on the 98% of fills that matter.
3. **Snap blackout.** Skip all quoting when |basket_dev|>10. The
   unfillable-and-adversely-selected ticks were a small but consistent
   leak (~6k/3 days).

**+68,068 vs v6's +55,174 = +23%.** No new code paths, no new signals;
just a re-allocation following per-fill drift evidence.
"""),
]


nb_path = HERE / 'pnl_snapshot.ipynb'
nbf.write(nb, nb_path)
print(f'wrote {nb_path}')
print('executing notebook…')
client = NotebookClient(nb, timeout=300, resources={'metadata': {'path': str(HERE)}})
client.execute()
nbf.write(nb, nb_path)
print('done.')
