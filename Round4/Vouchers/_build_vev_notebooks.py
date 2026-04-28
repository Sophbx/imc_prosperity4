"""Builds the VEV deep-dive notebooks. Run after data is in ../Data/.

  python3 _build_vev_notebooks.py

Produces:
  04_vev_underlying_and_pricing.ipynb  — VE dynamics + Black-Scholes vs market
  05_vev_microstructure_and_strategy.ipynb — counterparty flow + strategy
"""
import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).parent

def md(s): return nbf.v4.new_markdown_cell(s.strip("\n"))
def code(s): return nbf.v4.new_code_cell(s.strip("\n"))

# ---------------------------------------------------------------------------
# Notebook 04: VE underlying + option pricing fundamentals
# ---------------------------------------------------------------------------
nb04 = nbf.v4.new_notebook()
nb04.cells = [
    md("""
# VEV Round 4 — Part 1: Underlying & Pricing Fundamentals

Treating R4 as a brand-new round. Goal of this notebook:

1. Characterize VE (the underlying) behavior — mean reversion, effective vol
2. Compute Black-Scholes theoretical prices for every strike at every TTE
3. Compute implied vol per strike, see if there's a smile/skew
4. Identify which strikes are "fair", "rich", or "cheap"

Setup:
- 10 vouchers: VEV_4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500
- All are CALL options on VE (right to buy at strike)
- TTE: 7 days at start of round 1; rounds use 1 day each → R4 starts at TTE=4
- Historical data: R4 day 1 (TTE=7), day 2 (TTE=6), day 3 (TTE=5)
- Live R4 day 4: TTE=4 (we don't have data; it's the future)
"""),

    code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log, sqrt, exp
from scipy.stats import norm
from scipy.optimize import brentq
from pathlib import Path

DATA = Path('../Data')
prices = pd.concat([
    pd.read_csv(DATA / f'prices_round_4_day_{d}.csv', sep=';').assign(day_n=d)
    for d in [1,2,3]
], ignore_index=True)

ve = prices[prices['product']=='VELVETFRUIT_EXTRACT'].sort_values(['day_n','timestamp']).reset_index(drop=True)
print(f'VE rows: {len(ve)}, days: {sorted(ve.day_n.unique())}')
print(f'VE mid stats per day:')
for d in [1,2,3]:
    sub = ve[ve.day_n==d]
    print(f'  Day {d} (TTE={8-d}d): range [{sub.mid_price.min():.0f}, {sub.mid_price.max():.0f}], '
          f'mean={sub.mid_price.mean():.1f}, std={sub.mid_price.std():.2f}')
"""),

    md("## 1 — VE underlying time series"),
    code("""
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharey=True)
for ax, d in zip(axes, [1, 2, 3]):
    sub = ve[ve.day_n == d]
    ax.plot(sub.timestamp, sub.mid_price, lw=0.7)
    ax.set_title(f'Day {d} (TTE={8-d}d) — VE mid')
    ax.set_ylabel('mid')
    open_p, close_p = sub.mid_price.iloc[0], sub.mid_price.iloc[-1]
    ax.axhline(open_p, color='green', lw=0.5, ls='--', alpha=0.5, label=f'open {open_p:.0f}')
    ax.axhline(close_p, color='red', lw=0.5, ls='--', alpha=0.5, label=f'close {close_p:.0f}')
    ax.legend(loc='upper right', fontsize=8)
axes[-1].set_xlabel('timestamp')
plt.tight_layout()
"""),

    md("""
**Read.** VE drifts within each day:
- Day 1: 5245 → 5266 (+0.4%)
- Day 2: 5268 → 5296 (+0.5%)
- Day 3: 5296 → 5232 (−1.2%)

Cross-day gaps are tiny (Day 1 close 5266 → Day 2 open 5268 = +2; Day 2 close 5296 → Day 3 open 5296 = +0). Essentially overnight = 0.
"""),

    md("## 2 — Mean reversion test (variance ratio)"),
    code("""
# Compute variance ratios at multiple horizons. Within-day only.
ve['lret'] = np.log(ve.mid_price / ve.mid_price.shift(1))
ve.loc[ve.day_n != ve.day_n.shift(1), 'lret'] = np.nan

horizons = [1, 5, 10, 50, 100, 500, 1000, 2000, 5000, 9999]
data = []
base_var = None
for h in horizons:
    rets = []
    for d in [1,2,3]:
        sub = ve[ve.day_n==d].reset_index(drop=True)
        if h >= len(sub): continue
        r = np.log(sub.mid_price / sub.mid_price.shift(h)).dropna()
        rets.append(r)
    s = pd.concat(rets).std()
    if h == 1: base_var = s**2
    var_h = s**2
    var_rw = base_var * h  # random-walk prediction
    data.append({'h': h, 'std': s, 'var': var_h, 'var_rw': var_rw, 'ratio': var_h/var_rw})

df_vr = pd.DataFrame(data)
print('Variance ratio test:')
print(df_vr.round(6).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(df_vr.h, df_vr['ratio'], marker='o')
ax.axhline(1, color='red', lw=0.7, ls='--', label='random walk')
ax.set_xscale('log')
ax.set_xlabel('horizon h (ticks)')
ax.set_ylabel('Var(h-tick return) / [h × Var(1-tick return)]')
ax.set_title('VE variance ratio — < 1 means mean-reverting')
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
"""),

    md("""
**Read.** Variance ratio drops to ~0.40 at horizons of 5K–10K ticks. **VE
mean-reverts strongly within day** — terminal variance is only ~40% of the
random-walk projection. This means:
- The 1-tick log-return std (0.000217) **overstates** the relevant vol for
  option pricing.
- The right "effective σ" for terminal-payoff modeling needs to use the
  observed terminal variance, not the per-tick variance.
"""),

    md("## 3 — Vol estimate: 三种口径"),
    code("""
# Three ways to estimate σ for option pricing:
ticks_per_day = 10000
trading_days_per_year = 252

# Method 1: Naive 1-tick × √(annual ticks)
sigma_1tick = ve.lret.std()
sigma_naive_annual = sigma_1tick * np.sqrt(ticks_per_day * trading_days_per_year)

# Method 2: Within-day terminal log-return std × √252
day_returns = []
for d in [1,2,3]:
    sub = ve[ve.day_n==d]
    day_returns.append(np.log(sub.mid_price.iloc[-1] / sub.mid_price.iloc[0]))
sigma_daily_realized = np.std(day_returns)
sigma_daily_annual = sigma_daily_realized * np.sqrt(trading_days_per_year)

# Method 3: 5K-tick (half-day) std, scale up
ret5k = []
for d in [1,2,3]:
    sub = ve[ve.day_n==d]
    if len(sub) > 5000:
        ret5k.extend(np.log(sub.mid_price.iloc[5000:].values / sub.mid_price.iloc[:-5000].values))
sigma_5k_std = np.std(ret5k)
sigma_5k_annual = sigma_5k_std * np.sqrt(2 * trading_days_per_year)

print(f"σ estimate methods (annualized):")
print(f"  1) Naive 1-tick × √(2.52M ticks):    {sigma_naive_annual:.1%}")
print(f"  2) Day open→close × √252 (n=3):       {sigma_daily_annual:.1%}")
print(f"  3) 5k-tick std × √(2 × 252):          {sigma_5k_annual:.1%}")
print()
print('Method 1 ASSUMES random walk (wrong for VE). Method 2/3 use')
print('terminal-style observations and are more relevant for options.')
print('We will use ~10-15% as our "effective" σ baseline.')
"""),

    md("""
**Read.** Three different lenses give very different annualized σ:
- Method 1 (naive random-walk): **34%** — wrong (mean reversion violates assumption)
- Method 2 (3 day-end observations): **12.5%** — right form, but tiny sample
- Method 3 (half-day): **~10%** — right form, more samples

For Black-Scholes valuation, the "effective σ" should be **somewhere between
10% and 20%**. The market's implied vol (next section) sits in this band.
"""),

    md("## 4 — Black-Scholes theoretical vs market mid (per day)"),
    code("""
def bs_call(S, K, T, sigma, r=0):
    if T <= 0 or sigma <= 0: return max(0, S - K)
    d1 = (log(S/K) + 0.5*sigma**2 * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * norm.cdf(d1) - K * norm.cdf(d2)

def implied_vol(price, S, K, T, lo=0.001, hi=5.0):
    intr = max(0, S - K)
    if price <= intr + 0.01 or price > S: return np.nan
    try:
        return brentq(lambda s: bs_call(S,K,T,s) - price, lo, hi)
    except Exception:
        return np.nan

VE_DAY_MEAN = {1: 5248.4, 2: 5255.4, 3: 5239.2}
TTE = {1: 7, 2: 6, 3: 5}
strikes = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]

# Build a comparison table at "fair" σ = 0.15
SIGMA_BASELINE = 0.15
rows = []
for d in [1,2,3]:
    S = VE_DAY_MEAN[d]; T = TTE[d] / 252
    for K in strikes:
        sym = f'VEV_{K}'
        sub = prices[(prices['product']==sym) & (prices.day_n==d)]
        if len(sub)==0: continue
        mkt = sub.mid_price.mean()
        bs = bs_call(S, K, T, SIGMA_BASELINE)
        iv = implied_vol(mkt, S, K, T)
        rows.append({
            'day': d, 'TTE': TTE[d], 'strike': K, 'S': S,
            'mkt_mid': round(mkt, 2),
            f'BS_{int(SIGMA_BASELINE*100)}%': round(bs, 2),
            'IV': round(iv*100, 1) if not np.isnan(iv) else np.nan,
            'intrinsic': round(max(0, S-K), 1),
            'time_value': round(mkt - max(0, S-K), 2)
        })
df_cmp = pd.DataFrame(rows)
print('Per-day market mid vs BS@15% vs IV (averaged over the day):')
print(df_cmp.to_string(index=False))
"""),

    md("""
**Read.** Several patterns:
- **Deep ITM (4000, 4500)**: market trades at intrinsic ± a tick. Time value
  is essentially 0. Useless for vol trading. They're delta-1 proxies for the
  underlying.
- **OTM 5300–5500**: market mid is well below BS@15%. IV around 18–20%.
  These are the "real" option strikes.
- **6000 / 6500**: market pinned at 0.5 (between bid 0 and ask 1). BS@15%
  also gives ~0. These are essentially worthless under any reasonable vol.
- **Time value is preserved** even on short TTEs for OTM strikes — there's
  real optionality being priced in.
"""),

    md("## 5 — Implied vol smile/skew"),
    code("""
# IV per strike at each day — only for liquid OTM strikes
fig, ax = plt.subplots(figsize=(10, 5))

iv_strikes = [5000, 5100, 5200, 5300, 5400, 5500]
for d in [1,2,3]:
    ivs = []
    for K in iv_strikes:
        sub = df_cmp[(df_cmp.day==d) & (df_cmp.strike==K)]
        ivs.append(sub.IV.iloc[0] if len(sub) else np.nan)
    ax.plot(iv_strikes, ivs, marker='o', ms=8, label=f'Day {d} (TTE={TTE[d]})')

ax.set_xlabel('Strike')
ax.set_ylabel('Implied Vol (%)')
ax.set_title('VEV implied vol smile across strikes')
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
"""),

    md("""
**Read.** The implied vol surface is **flat at ~19–20% across strikes 5000-5500**
on all 3 days. There's very little skew or smile — every strike implies
basically the same vol. So:
- The market doesn't price in skew (i.e., it doesn't think tails are heavier
  than lognormal)
- Vol is roughly stable day-over-day (TTE=7, 6, 5 all show 18-20%)
- For R4 day 4 (TTE=4), expect IV around 19% if the pattern continues
"""),

    md("## 6 — Time evolution of IV (intra-day)"),
    code("""
# Sample IV per strike at multiple snapshots within day 3 (TTE=5)
T = 5/252
iv_samples = []
for K in [5200, 5300, 5400, 5500]:
    sym = f'VEV_{K}'
    sub_p = prices[(prices['product']==sym) & (prices.day_n==3)].copy()
    sub_ve = ve[ve.day_n==3][['timestamp','mid_price']].rename(columns={'mid_price':'S'})
    merged = sub_p.merge(sub_ve, on='timestamp', how='left')
    merged['IV'] = merged.apply(
        lambda r: implied_vol(r.mid_price, r.S, K, T) * 100 if not np.isnan(r.S) else np.nan,
        axis=1)
    iv_samples.append((K, merged))

fig, ax = plt.subplots(figsize=(12, 5))
for K, m in iv_samples:
    sample = m.iloc[::100]  # subsample for plot
    ax.plot(sample.timestamp, sample.IV, label=f'VEV_{K}', lw=1)
ax.set_title('Day 3 (TTE=5) — IV per strike over time')
ax.set_xlabel('timestamp')
ax.set_ylabel('IV %')
ax.set_ylim(15, 25)
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
"""),

    md("""
**Read.** IV is remarkably stable intra-day — narrow band ~18-21% throughout.
No vol regime shifts. The market's σ assumption is consistent.

For our strategy: **selling at IV 19% with realized 12.5% has positive
expectation** of about half a vol point per day in option premium. But it
comes with tail risk — needs sizing.
"""),

    md("""
## 7 — TTE extrapolation: what to expect at TTE=4 (R4 live)
"""),

    code("""
# Forecast theoretical prices at TTE=4 using IV=19% and current S~5240
S_now = 5240  # current spot estimate
T_live = 4/252
sigma_live = 0.19  # implied from market

print(f"Forecasted theoretical prices at TTE=4 (S={S_now}, σ={sigma_live*100:.0f}%):\\n")
print(f"{'strike':>7}{'BS@19%':>10}{'BS@15%':>10}{'BS@12.5%':>10}{'BS@10%':>10}")
for K in strikes:
    prices_at_vol = []
    for s in [0.19, 0.15, 0.125, 0.10]:
        prices_at_vol.append(bs_call(S_now, K, T_live, s))
    print(f"{K:>7}{prices_at_vol[0]:>10.2f}{prices_at_vol[1]:>10.2f}{prices_at_vol[2]:>10.2f}{prices_at_vol[3]:>10.2f}")
"""),

    md("""
**Read.** Across plausible σ scenarios (10–19%), the forecasted theoretical
values for OTM strikes are:
- **VEV_5300**: 4.7 to 16.5 (likely ~10–14)
- **VEV_5400**: 0.4 to 6.1 (likely ~2–4)
- **VEV_5500**: 0.02 to 1.7 (likely ~0.5–1)
- **VEV_6000 / 6500**: ~0 under any reasonable σ

If realized vol matches market IV (≈19%), market mid prices are accurate.
If realized is ≤12.5%, market is overpricing vol → sell calls. Risk: realized
spikes higher (vol expansion → tail risk on shorts).

## Summary

1. VE is mean-reverting (variance ratio ≈ 0.40 at long horizons).
2. Effective σ for options = 10–15%; market IV = 19%; realized "ought to be"
   somewhere in between.
3. Implied vol surface is flat (~19% across all strikes) — no skew priced.
4. VEV_4000/4500 trade as delta-1 (no time value).
5. VEV_5300–5500 are the "real" option strikes; ~3–7% premium over realized
   vol in expectation.
6. VEV_6000/6500 are pinned at minimum tick (0.5) — essentially worthless.

Continue to **05_vev_microstructure_and_strategy.ipynb** for trading flow
analysis and concrete strategies.
"""),
]


# ---------------------------------------------------------------------------
# Notebook 05: microstructure + strategy
# ---------------------------------------------------------------------------
nb05 = nbf.v4.new_notebook()
nb05.cells = [
    md("""
# VEV Round 4 — Part 2: Microstructure & Strategy

This notebook digs into:

1. Per-strike trading volume, spread, book depth
2. Counterparty (Mark) flow analysis per strike
3. Mark 01 / Mark 22 dominance and what we can extract from it
4. Special handling for VEV_6000/6500 ("free lottery tickets")
5. Concrete strategy proposals with sizing estimates
"""),

    code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path('../Data')
prices = pd.concat([
    pd.read_csv(DATA / f'prices_round_4_day_{d}.csv', sep=';').assign(day_n=d)
    for d in [1,2,3]
], ignore_index=True)
trades = pd.concat([
    pd.read_csv(DATA / f'trades_round_4_day_{d}.csv', sep=';').assign(day_n=d)
    for d in [1,2,3]
], ignore_index=True)

vev_strikes = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
vev_syms = [f'VEV_{k}' for k in vev_strikes]

# Helper: merge trades with their tick-time mid
def with_mid(t_df, sym):
    p = prices[prices['product']==sym][['day_n','timestamp','bid_price_1','ask_price_1','mid_price']]
    return t_df.merge(p, on=['day_n','timestamp'], how='left')

print('Trade rows per strike:')
print(trades[trades.symbol.isin(vev_syms)].symbol.value_counts().to_string())
"""),

    md("## 1 — Per-strike liquidity & spread overview"),
    code("""
rows = []
for K, sym in zip(vev_strikes, vev_syms):
    p = prices[prices['product']==sym]
    t = trades[trades['symbol']==sym]
    sp = p['ask_price_1'] - p['bid_price_1']
    rows.append({
        'strike': K,
        'mid_min': p.mid_price.min(),
        'mid_avg': round(p.mid_price.mean(), 2),
        'mid_max': p.mid_price.max(),
        'spread_avg': round(sp.mean(), 2),
        'L1_size': round((p.bid_volume_1.fillna(0) + p.ask_volume_1.fillna(0)).mean()/2, 1),
        'trades': len(t),
        'volume': int(t.quantity.sum()),
        'avg_trade_qty': round(t.quantity.mean(), 1) if len(t) else 0,
    })
df_liq = pd.DataFrame(rows)
print('Liquidity profile by strike:')
print(df_liq.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].bar([str(K) for K in vev_strikes], df_liq.trades, color='steelblue', edgecolor='black')
axes[0].set_title('Trades per strike (3 days)')
axes[0].set_xlabel('Strike')
axes[0].set_ylabel('Trades')
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(df_liq.trades):
    axes[0].text(i, v+5, str(v), ha='center', fontsize=8)

axes[1].bar([str(K) for K in vev_strikes], df_liq.spread_avg, color='salmon', edgecolor='black')
axes[1].set_title('Avg bid-ask spread per strike')
axes[1].set_xlabel('Strike')
axes[1].set_ylabel('Spread (ticks)')
axes[1].grid(axis='y', alpha=0.3)
for i, v in enumerate(df_liq.spread_avg):
    axes[1].text(i, v+0.3, f'{v:.1f}', ha='center', fontsize=8)
plt.tight_layout()
"""),

    md("""
**Read.** Three groups by liquidity:
- **Liquid**: 4000 (442 trades), 5300 (164), 5400 (276), 5500 (306), 6000 (317), 6500 (317)
- **Dead zone**: 4500, 5000, 5100 (3 trades each over 3 days!)
- **Thin**: 5200 (47 trades)

The dead zone (4500/5000/5100) is striking. These are ITM strikes where
the option behaves like the underlying minus a constant — yet barely trades.
The market makes inside quotes but flow doesn't engage them.

Spread widens with distance from spot:
- VEV_4000 spread = 21 (≈1.6% of mid)
- VEV_5500 spread = 1.1 (just 1 tick — at minimum quoting granularity)
- VEV_6000/6500 spread = 1.0 (pinned bid 0 / ask 1)
"""),

    md("## 2 — Counterparty mix per strike"),
    code("""
rows = []
all_marks = sorted(set(trades.buyer.dropna()) | set(trades.seller.dropna()))
for K, sym in zip(vev_strikes, vev_syms):
    t = trades[trades.symbol==sym]
    if len(t)==0:
        rows.append({'strike': K, **{f'{m}_buy': 0 for m in all_marks}, **{f'{m}_sell': 0 for m in all_marks}})
        continue
    buy = t.groupby('buyer').size()
    sell = t.groupby('seller').size()
    row = {'strike': K}
    for m in all_marks:
        row[f'{m}_buy'] = int(buy.get(m, 0))
        row[f'{m}_sell'] = int(sell.get(m, 0))
    rows.append(row)
mix = pd.DataFrame(rows).set_index('strike')

# Show only the columns where there's at least one nonzero
nonzero_cols = [c for c in mix.columns if mix[c].sum() > 0]
print('Trade counts per strike per Mark (buyer / seller):')
print(mix[nonzero_cols].to_string())
"""),

    md("""
**Read.** A startlingly clean picture across vouchers:
- **Mark 01 is the BUYER** in nearly every voucher trade
- **Mark 22 is the SELLER** in nearly every voucher trade
- Mark 14 / Mark 38 sprinkle a few trades but minor

The voucher market is essentially Mark 01 ↔ Mark 22, paired in volume, on
every strike. They alone generate 90%+ of voucher activity.
"""),

    md("## 3 — Mark 22's selling edge to Mark 01"),
    code("""
# At what price does Mark 22 sell vs the mid? That's free spread to Mark 01.
liquid_otm = [5300, 5400, 5500, 6000, 6500]
results = []
for K in liquid_otm:
    sym = f'VEV_{K}'
    t = trades[(trades.symbol==sym) & (trades.seller=='Mark 22')]
    t = with_mid(t, sym)
    n = len(t)
    if n == 0: continue
    edge = t.mid_price - t.price       # mid > sell price → edge to buyer
    results.append({
        'strike': K,
        'n_M22_sells': n,
        'avg_sell_price': round(t.price.mean(), 2),
        'avg_mid_at_sell': round(t.mid_price.mean(), 2),
        'avg_edge_per_share': round(edge.mean(), 2),
        'total_edge': round(edge.sum() * 1, 1),  # not weighted by qty yet
        'tot_qty_sold': int(t.quantity.sum()),
        'edge_x_qty': round((edge * t.quantity).sum(), 1),
    })
df_m22 = pd.DataFrame(results)
print('Mark 22 selling pattern on OTM vouchers (3 days):')
print(df_m22.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(df_m22.strike.astype(str), df_m22.avg_edge_per_share, color='lightseagreen', edgecolor='black')
ax.axhline(0, color='black', lw=0.5)
ax.set_xlabel('Strike')
ax.set_ylabel('Mid − Mark 22 sell price')
ax.set_title('Mark 22 systematically sells BELOW mid on every OTM strike')
for i, v in enumerate(df_m22.avg_edge_per_share):
    ax.text(i, v+0.02, f'{v:.2f}', ha='center', fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
"""),

    md("""
**Read.** Mark 22 sells 0.5–0.9 *below mid* on every OTM strike, every day.
Mark 01 collects this. This is structural alpha — not a trade idea, a
systematic mispricing.

For VEV_6000 / VEV_6500 the "edge per share" is exactly 0.5 because mid is
0.5 (bid 0 / ask 1) and Mark 22 sells at 0. Each contract has a real (if
small) terminal payoff probability under any non-degenerate vol assumption.
Half a seashell per share × 1100 shares per day = ~$550/day **per strike**
of pure carry.
"""),

    md("## 4 — Time evolution: when does Mark 22 sell?"),
    code("""
# Cumulative Mark 22 sell volume per strike over time across all 3 days
fig, ax = plt.subplots(figsize=(11, 5))
for K, color in zip(liquid_otm, ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']):
    t = trades[(trades.symbol==f'VEV_{K}') & (trades.seller=='Mark 22')].copy()
    t['ts_cont'] = t.timestamp + (t.day_n - 1) * 1_000_000
    t = t.sort_values('ts_cont')
    cum = t.quantity.cumsum()
    ax.plot(t.ts_cont, cum, label=f'VEV_{K}', color=color, lw=1.4)
for boundary in [1_000_000, 2_000_000]:
    ax.axvline(boundary, color='gray', lw=0.5, ls='--')
ax.set_xlabel('continuous timestamp')
ax.set_ylabel('cumulative Mark 22 sell qty')
ax.set_title('Mark 22 sell flow across 3 days (continuous timeline)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
"""),

    md("""
**Read.** Mark 22 sells smoothly throughout the day on all strikes — no
clustering at open/close, no event-driven bursts. **Predictable steady
flow** at roughly 100–400 lots/day per strike. We can plan to capture this
with a passive bid that sits all day.
"""),

    md("## 5 — VEV_6000 and VEV_6500: the free-lottery anomaly"),
    code("""
# Distribution of trade prices on the deep OTM strikes
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, K in zip(axes, [6000, 6500]):
    t = trades[trades.symbol==f'VEV_{K}']
    counts = t.price.value_counts().sort_index()
    ax.bar(counts.index, counts.values, color='gold', edgecolor='black', width=0.8)
    ax.set_title(f'VEV_{K}: trade-price distribution (3 days)')
    ax.set_xlabel('Trade price')
    ax.set_ylabel('Number of trades')
    ax.grid(axis='y', alpha=0.3)
    for i, (p, c) in enumerate(zip(counts.index, counts.values)):
        ax.text(p, c+5, f'{int(c)} trades', ha='center', fontsize=9)
plt.tight_layout()

# Position of buyers vs sellers
print('VEV_6000 / VEV_6500 — buyer & seller breakdown:')
for K in [6000, 6500]:
    t = trades[trades.symbol==f'VEV_{K}']
    print(f'\\n  VEV_{K}: {len(t)} trades, all at price 0.0')
    print(f'    Buyers:  {t.buyer.value_counts().to_dict()}')
    print(f'    Sellers: {t.seller.value_counts().to_dict()}')
"""),

    md("""
**Read.** **100% of VEV_6000 / VEV_6500 trades are at price 0**, exclusively
between Mark 22 (seller) and Mark 01 (buyer). Each contract:
- Pays max(VE_T − 6000, 0) at expiry
- Under realized vol, terminal probability of S > 6000 is essentially 0
- BS @ 19% IV gives ~$0 value too

But the contracts ARE traded (to Mark 01), so the market has no problem
clearing them at zero. **If we sit on the bid at 0, we receive free options.**
Even if the EV is microscopic, with 300 limit and 1100 lots/day available,
we can grab a full position with no capital outlay.

Risk: if VE has an unexpected gap up to 6000+ at expiry, we'd profit
massively. Standard risk: theta decay → 0 over 4 days → no harm.
"""),

    md("## 6 — Capturing Mark 22's flow: edge math per strategy"),
    code("""
# If we undercut Mark 01's bid by 1 tick on each OTM strike, what's the
# annualized capture? Use 3-day Mark 22 volume × edge per share.
print('If we capture 100% of Mark 22 sell flow at bid+1 (i.e. 1 above bid 0 = price 1):\\n')
print(f"{'strike':>7}{'M22 vol/3d':>12}{'edge/share':>12}{'PnL/3d':>12}{'PnL/day':>10}")
for K in liquid_otm:
    sym = f'VEV_{K}'
    t = trades[(trades.symbol==sym) & (trades.seller=='Mark 22')]
    t = with_mid(t, sym)
    if len(t)==0: continue
    qty = t.quantity.sum()
    edge = (t.mid_price - t.price).mean()
    # If we undercut to bid+1, our buy price is bid+1 (one tick above what M22 lifts).
    # Edge becomes (mid - (bid+1)) = mid - bid - 1 = (spread/2) - 1 (since mid = bid+spread/2)
    # On strikes with spread=1, this nukes the edge to 0 — undercut isn't viable
    spread = (prices[(prices['product']==sym)]['ask_price_1']
              - prices[(prices['product']==sym)]['bid_price_1']).mean()
    captured_edge = max(0.0, spread/2 - 1)  # what we get IF we undercut bid+1
    print(f"{K:>7}{int(qty):>12}{edge:>12.2f}{captured_edge*qty:>12.1f}{captured_edge*qty/3:>10.1f}")
"""),

    md("""
**Read.** Outright bid+1 undercut doesn't work for OTM vouchers because
their **spread is already 1 tick** (bid 0, ask 1; or for 5400 spread is 1.3
on average). So undercutting Mark 01 by 1 tick collapses our edge to ≤0.

Two alternatives:
- **Sit alongside Mark 01 at the same bid price** — split flow with him
  (FIFO is unknown, but the wiki says price-time priority; whoever places
  first wins; safe assumption: we get some fraction)
- **Trade at the bid (price 0)** for VEV_6000/6500 — same as Mark 01, free
  optionality

For 5300/5400/5500: we can NOT improve Mark 01's price. Best is matching
his bid and hoping for FIFO crumbs.
"""),

    md("## 7 — A different angle: short-vol on VEV_5300/5400/5500"),
    code("""
# Selling the OTM strikes at market mid: how does PnL look if vol = 12.5% realized?
from math import log, sqrt, exp
from scipy.stats import norm

def bs_call(S, K, T, sigma, r=0):
    if T <= 0 or sigma <= 0: return max(0, S - K)
    d1 = (log(S/K) + 0.5*sigma**2 * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * norm.cdf(d1) - K * norm.cdf(d2)

S = 5240
T_live = 4/252
print(f'\\nShort-vol thought experiment (live R4 day 4, S={S}, TTE=4):\\n')
print(f"{'strike':>7}{'mkt_mid_now':>12}{'BS@19%':>10}{'BS@12.5%':>10}{'EV/share':>10}{'limit':>8}{'EV_total':>10}")

# Use day-3 (TTE=5) market mid as proxy for TTE=4 mid
day3_mids = {}
for K in [5300, 5400, 5500]:
    sym = f'VEV_{K}'
    sub = prices[(prices['product']==sym) & (prices.day_n==3)]
    day3_mids[K] = sub.mid_price.iloc[-1]  # end-of-day-3 mid

for K in [5300, 5400, 5500]:
    sym = f'VEV_{K}'
    mkt_eod_d3 = day3_mids[K]
    bs_19 = bs_call(S, K, T_live, 0.19)
    bs_125 = bs_call(S, K, T_live, 0.125)
    ev_per_share = mkt_eod_d3 - bs_125  # if we sell at market, PnL = sell_price - true_value
    pos_limit = 300
    print(f"{K:>7}{mkt_eod_d3:>12.2f}{bs_19:>10.2f}{bs_125:>10.2f}{ev_per_share:>10.2f}{pos_limit:>8}{ev_per_share*pos_limit:>10.1f}")

print('\\nCAVEAT: this assumes σ realized = 12.5%. If realized > 19%, short loses.')
print('Sample size for σ=12.5% is only 3 days, so conviction is moderate.')
"""),

    md("""
**Read.** Selling OTM calls at market mid, IF realized σ ≈ 12.5%:
- VEV_5300: ~$1–4 EV per share (depending on vol assumption) × 300 limit ≈
  $300–1200 per day
- VEV_5400: similar
- VEV_5500: similar but smaller

Aggregate: roughly $1–4K EV per day from short-vol if realized σ stays
~12-15%. Meaningful but not huge, with **real tail risk** (a vol spike
loses big).

For a single 1-day round, this is highly path-dependent. Worth doing
for a fraction of position limit, not full size.
"""),

    md("""
## 8 — Strategy synthesis (what to actually trade)

Three concrete plays, ordered by conviction:

### Play A — VEV_6000 / VEV_6500 free lottery (high conviction)

- Sit on the bid at 0 for both strikes
- Capture Mark 22's outflow alongside Mark 01 (FIFO split, whatever we get)
- Build position toward 300 limit (free options; no capital outlay)
- Expected PnL: probably ~0, but >= 0 in expectation. No downside.
- Tail upside: if VE jumps, payoff is uncapped

### Play B — VEV_5300 / VEV_5400 / VEV_5500 short-vol (medium conviction)

- Quote ASK alongside Mark 22 (or at same level)
- When Mark 01 lifts our ask, we collect option premium
- Hedge: small long VE (deltahedge if we have the engine)
- Expected EV: $1–4K/day combined, with vol spike risk
- Sizing: 100–150 contracts max per strike (half of limit)

### Play C — VEV_4000 deltahedge (medium conviction)

- VEV_4000 trades at intrinsic; effectively delta-1 vs VE
- Use it as a hedge when we have unwanted VE delta from option positions
- Or: pure spread-capture in VEV_4000 itself (442 trades/3 days, decent flow)

### What NOT to do

- **VEV_4500 / VEV_5000 / VEV_5100**: dead zone. 3 trades over 3 days. No
  flow to trade against. Skip entirely.
- **VEV_5200**: thin. 47 trades. Probably skip unless data improves.
- **Aggressive long-vol**: market IV (19%) > realized (12.5%); long is -EV
  on average.

### Position-limit budget

Total budget = 10 strikes × 300 = 3000 contracts. Realistic plan:
- VEV_6000/6500: 600 (max long both for free lottery)
- VEV_5300/5400/5500: 450 short total (3× 150)
- VEV_4000: optional hedge inventory

Total active: ~1050 contracts. Well within structural limits, with room
to scale up if alpha confirms in live.
"""),
]

def write(nb, path: Path):
    nbf.write(nb, str(path))
    print(f'wrote {path.name} ({len(nb.cells)} cells)')

write(nb04, HERE / '04_vev_underlying_and_pricing.ipynb')
write(nb05, HERE / '05_vev_microstructure_and_strategy.ipynb')
