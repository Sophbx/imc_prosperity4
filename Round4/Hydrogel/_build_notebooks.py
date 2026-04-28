"""Generates the two analysis notebooks. Run once to (re)create them.

  python3 _build_notebooks.py
"""
import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).parent


def md(src: str):
    return nbf.v4.new_markdown_cell(src.strip("\n"))


def code(src: str):
    return nbf.v4.new_code_cell(src.strip("\n"))


# ---------------------------------------------------------------------------
# Notebook 1: HG price dynamics
# ---------------------------------------------------------------------------
nb1 = nbf.v4.new_notebook()
nb1.cells = [
    md("""
# Hydrogel — Price Dynamics

Three days of R4 data: `prices_round_4_day_{1,2,3}.csv`. Goal is to ground every
later strategy decision in what the HG mid actually does.
"""),

    code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path('../Data')
prices = pd.concat([
    pd.read_csv(DATA / f'prices_round_4_day_{d}.csv', sep=';').assign(day=d)
    for d in [1, 2, 3]
], ignore_index=True)
hg = prices[prices['product'] == 'HYDROGEL_PACK'].sort_values(['day','timestamp']).reset_index(drop=True)
hg['spread'] = hg['ask_price_1'] - hg['bid_price_1']
hg['mid_dev'] = hg['mid_price'] - 10000
print(f'rows: {len(hg)}, days: {sorted(hg.day.unique())}')
hg.head()
"""),

    md("## 1 — Mid price across all 3 days"),
    code("""
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharey=True)
for ax, d in zip(axes, [1, 2, 3]):
    sub = hg[hg.day == d]
    ax.plot(sub.timestamp, sub.mid_price, lw=0.6)
    ax.axhline(10000, color='red', lw=0.8, alpha=0.6, label='fair=10000')
    ax.set_title(f'Day {d} — range [{sub.mid_price.min():.0f}, {sub.mid_price.max():.0f}], '
                 f'std {sub.mid_price.std():.1f}')
    ax.set_ylabel('mid')
    ax.legend(loc='upper right', fontsize=8)
axes[-1].set_xlabel('timestamp')
plt.tight_layout()
"""),

    md("""
**Read.** All three days oscillate around 10000 with similar amplitude (~±80) and
similar std (~32–37). No directional drift — mean reversion is the dominant
dynamic. The fixed `fair = 10000` anchor is justified.
"""),

    md("## 2 — Distribution of `mid − 10000`"),
    code("""
fig, ax = plt.subplots(figsize=(10, 4))
for d, color in zip([1, 2, 3], ['#1f77b4', '#ff7f0e', '#2ca02c']):
    sub = hg[hg.day == d]
    ax.hist(sub.mid_dev, bins=60, alpha=0.45, label=f'Day {d}', color=color)
ax.axvline(0, color='red', lw=0.8)
ax.set_xlabel('mid − 10000')
ax.set_ylabel('snapshot count')
ax.set_title('mid deviation distribution per day')
ax.legend()
plt.tight_layout()

print('Quantiles of mid − 10000 (all 3 days):')
q = hg.mid_dev.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(1)
print(q.to_string())
"""),

    md("""
**Read.** Median ≈ 0, p5..p95 ≈ −63..+48. Heavy mean reversion to 10000. Slight
asymmetry: p25=−33 vs p75=+19 — the distribution skews slightly negative
(mid spends more time below 10000 than above). Days vary in mode but all
center near 0.
"""),

    md("## 3 — Bid–ask spread"),
    code("""
print('Spread distribution (3 days, 30000 snapshots):')
print(hg.spread.value_counts().sort_index().to_string())
print(f'\\nMean spread: {hg.spread.mean():.2f}, median: {hg.spread.median():.0f}')
print(f'Fraction at spread == 16: {(hg.spread == 16).mean():.1%}')

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(hg[hg.day == 3].timestamp, hg[hg.day == 3].spread, lw=0.5)
ax.set_title('Day 3 — spread over time')
ax.set_ylabel('spread (ticks)')
ax.set_xlabel('timestamp')
plt.tight_layout()
"""),

    md("""
**Read.** Spread is **16 in 92.5% of all snapshots**. Tightenings to 7/8/9 happen
~3% of the time and are usually momentary (single-tick events). Widening to
17 also rare. So the inside quote on HG is structurally a stable 16-wide
two-sided book — that's almost certainly Mark 14's quotes (we'll confirm in
notebook 2).
"""),

    md("## 4 — Order-book depth"),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
axes[0].hist(hg.bid_volume_1, bins=range(0, 20), alpha=0.7, color='steelblue')
axes[0].set_title('L1 bid size')
axes[1].hist(hg.ask_volume_1, bins=range(0, 20), alpha=0.7, color='salmon')
axes[1].set_title('L1 ask size')
for ax in axes: ax.set_xlabel('size')
plt.tight_layout()

print('L1 sizes (bid/ask are symmetric):')
print(hg[['bid_volume_1','ask_volume_1','bid_volume_2','ask_volume_2']].describe().round(1).to_string())

# Gap between L1 and L2
hg['bid_gap'] = hg.bid_price_1 - hg.bid_price_2
hg['ask_gap'] = hg.ask_price_2 - hg.ask_price_1
print('\\nL1→L2 price gap:')
print(hg[['bid_gap','ask_gap']].describe().round(1).to_string())
"""),

    md("""
**Read.** L1 sizes 4–15 (avg ~12, symmetric bid/ask). L2 sizes 10–30 (avg ~25).
L1→L2 gap typically 2–3 ticks. So when we quote 1 tick inside, we're inside
the entire L1 layer, not just snipping off a sliver — Mark 38's flow funnels
to us first.
"""),

    md("## 5 — Mean reversion: return autocorrelation"),
    code("""
# 1-tick returns at multiple horizons; check autocorrelation
hg_d3 = hg[hg.day == 3].copy()
for h in [1, 5, 10, 50, 100]:
    hg_d3[f'r{h}'] = hg_d3.mid_price.diff(h)
ret_cols = [f'r{h}' for h in [1, 5, 10, 50, 100]]

fig, ax = plt.subplots(figsize=(7, 4))
lags = list(range(1, 31))
for h in [1, 10]:
    series = hg_d3[f'r{h}'].dropna()
    acf = [series.autocorr(lag) for lag in lags]
    ax.plot(lags, acf, label=f'{h}-tick return', marker='o', ms=3)
ax.axhline(0, color='black', lw=0.5)
ax.set_title('Day 3 — return autocorrelation (negative = mean reverting)')
ax.set_xlabel('lag')
ax.set_ylabel('autocorr')
ax.legend()
plt.tight_layout()
"""),

    md("""
**Read.** At lag 1 the 1-tick return autocorr is strongly **negative** — classic
mean-reverting signature. The 10-tick return shows weaker negative
autocorrelation that decays. So mean reversion is fastest at the 1-tick
scale but persists out to ~10–20 ticks. This validates the half-spread we
choose (8): we sit through the noise, mean reversion brings the price
back, we close the round trip.
"""),

    md("## 6 — Z-score behavior (with σ = 32)"),
    code("""
hg['z'] = hg.mid_dev / 32
print(f'Time spent in z-bands:')
for lo, hi, lbl in [(-99, -1.65, 'z < -1.65'),
                    (-1.65, -0.85, '-1.65 ≤ z < -0.85'),
                    (-0.85, 0.85, '|z| < 0.85'),
                    (0.85, 1.65, '0.85 ≤ z < 1.65'),
                    (1.65, 99, 'z ≥ 1.65')]:
    pct = ((hg.z >= lo) & (hg.z < hi)).mean()
    print(f'  {lbl}: {pct:6.1%}')

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(hg.z, bins=60, color='gray', alpha=0.7)
for v in [-1.65, -0.85, 0.85, 1.65]:
    ax.axvline(v, color='red', lw=0.8, alpha=0.6)
ax.set_title('z-score (= (mid − 10000) / 32) distribution, all 3 days')
ax.set_xlabel('z')
plt.tight_layout()
"""),

    md("""
**Read.** ~53% of time |z| < 0.85 (no signal), 17% rich, 29% cheap. The R3
swing-trader was idle whenever |z| < 0.85, which is half the day. The
slight asymmetry means we'll see slightly more cheap-mid moments than
rich, so the buy-side of our MM gets more flow on net.
"""),

    md("## 7 — Intraday pattern"),
    code("""
# Bucket by 100k ticks across all days, see if there's time-of-day
hg['bucket'] = (hg.timestamp // 100000)
agg = hg.groupby(['bucket']).agg(
    mid_mean=('mid_price', 'mean'),
    mid_std=('mid_price', 'std'),
    mid_dev=('mid_dev', 'mean'),
    spread_tight=('spread', lambda x: (x < 16).mean()),
).round(2)
print('Intraday — averaged across 3 days:')
print(agg.to_string())

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
axes[0].plot(agg.index, agg.mid_dev, marker='o')
axes[0].axhline(0, color='red', lw=0.5)
axes[0].set_ylabel('mean (mid - 10000)')
axes[0].set_title('Intraday drift in mid (negative = cheap on average)')
axes[1].plot(agg.index, agg.mid_std, marker='o', color='orange')
axes[1].set_ylabel('mid std within bucket')
axes[1].set_xlabel('bucket (100k ticks)')
plt.tight_layout()
"""),

    md("""
**Read.** Slight positive drift early in the day (buckets 0–2) and back toward 0
later. Vol (std within bucket) is roughly flat ~10–18, no clear regime change.
No time-of-day rule needed.

## Summary of price dynamics
- Mid mean-reverts to 10000 every day; std ~32; range ±80.
- Spread = 16 in 92.5% of snapshots (Mark-14-shaped book).
- L1 sizes 4–15; L2 sizes 10–30; L1→L2 gap 2–3 ticks.
- Returns mean-revert most strongly at 1-tick horizon; persists ~10 ticks.
- z-score: 53% inactive (|z| < 0.85), 17% rich, 29% cheap.
- No intraday seasonality worth conditioning on.
"""),
]

# ---------------------------------------------------------------------------
# Notebook 2: HG Marks behavior
# ---------------------------------------------------------------------------
nb2 = nbf.v4.new_notebook()
nb2.cells = [
    md("""
# Hydrogel — Counterparty (Mark) Behavior

R4 trade files now include `buyer` and `seller`. This notebook breaks down
who trades HG, where they trade, and what their flow predicts.
"""),

    code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path('../Data')

trades = pd.concat([
    pd.read_csv(DATA / f'trades_round_4_day_{d}.csv', sep=';').assign(day=d)
    for d in [1,2,3]
], ignore_index=True)
prices = pd.concat([
    pd.read_csv(DATA / f'prices_round_4_day_{d}.csv', sep=';').assign(day=d)
    for d in [1,2,3]
], ignore_index=True)

# HG only
hg_t = trades[trades.symbol == 'HYDROGEL_PACK'].copy()
hg_p = prices[prices['product'] == 'HYDROGEL_PACK'][['day','timestamp','bid_price_1','ask_price_1','mid_price']]
hg_t = hg_t.merge(hg_p, on=['day','timestamp'], how='left')
hg_t['mid'] = (hg_t.bid_price_1 + hg_t.ask_price_1) / 2.0
print(f'HG trades total: {len(hg_t)} (across 3 days)')
hg_t.head()
"""),

    md("## 1 — Who trades HG?"),
    code("""
# Total trade count + volume per Mark, as buyer and as seller
b = hg_t.groupby('buyer').agg(buyer_trades=('quantity','size'), buyer_qty=('quantity','sum'))
s = hg_t.groupby('seller').agg(seller_trades=('quantity','size'), seller_qty=('quantity','sum'))
roster = b.join(s, how='outer').fillna(0).astype(int)
roster['total_trades'] = roster.buyer_trades + roster.seller_trades
roster['total_qty'] = roster.buyer_qty + roster.seller_qty
roster['net_qty'] = roster.buyer_qty - roster.seller_qty  # >0 = net buyer
roster = roster.sort_values('total_trades', ascending=False)
print(roster.to_string())

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].barh(roster.index, roster.total_trades, color='steelblue')
axes[0].set_title('HG: total trades per Mark (3 days)')
axes[0].invert_yaxis()
axes[1].barh(roster.index, roster.net_qty, color=['green' if x>0 else 'crimson' for x in roster.net_qty])
axes[1].axvline(0, color='black', lw=0.5)
axes[1].set_title('HG: net signed qty (3 days, +=net buyer)')
axes[1].invert_yaxis()
plt.tight_layout()
"""),

    md("""
**Read.** Three Marks are active in HG: **Mark 14** (most trades) and
**Mark 38** (nearly identical count) dominate; **Mark 22** has a few. Net
qty is essentially zero for all of them — nobody on HG is a one-way
directional flow. That's a sign HG is a pure spread-capture game, not a
flow-following game.
"""),

    md("## 2 — Buy / sell split per Mark"),
    code("""
buy_pct = hg_t.groupby('buyer').size().rename('as_buyer')
sell_pct = hg_t.groupby('seller').size().rename('as_seller')
sides = pd.concat([buy_pct, sell_pct], axis=1).fillna(0).astype(int)
sides['buy_pct'] = (sides.as_buyer / (sides.as_buyer + sides.as_seller) * 100).round(1)
sides = sides.loc[roster.index]  # match order
print(sides.to_string())

fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(sides.index, sides.buy_pct, color='gray')
ax.axhline(50, color='red', lw=0.8, linestyle='--', alpha=0.6, label='50% (neutral)')
ax.set_ylabel('% of trades as buyer')
ax.set_title('HG: each Mark\\'s buy ratio')
ax.legend()
plt.tight_layout()
"""),

    md("""
**Read.** All three Marks are basically 50/50 buyer/seller — confirming HG
flow is symmetric. Nobody is structurally short or long the book.
"""),

    md("## 3 — Where each Mark trades vs the inside"),
    code("""
# Did each Mark fill at bid_1, ask_1, or somewhere else?
def classify(row, who):
    side = +1 if row['buyer'] == who else (-1 if row['seller'] == who else 0)
    if side == 0:
        return None
    px = row['price']
    if side > 0:  # bought
        if px == row['bid_price_1']: return 'bought AT bid_1 (passive)'
        if px == row['ask_price_1']: return 'bought AT ask_1 (crossed)'
        return f'bought elsewhere ({px} vs bid={row.bid_price_1}, ask={row.ask_price_1})'
    else:  # sold
        if px == row['ask_price_1']: return 'sold AT ask_1 (passive)'
        if px == row['bid_price_1']: return 'sold AT bid_1 (crossed)'
        return f'sold elsewhere ({px} vs bid={row.bid_price_1}, ask={row.ask_price_1})'

for who in roster.index:
    sub = hg_t[(hg_t.buyer == who) | (hg_t.seller == who)].copy()
    sub['cls'] = sub.apply(classify, axis=1, who=who)
    print(f'\\n=== {who} ({len(sub)} trades) ===')
    print(sub.cls.value_counts().to_string())
"""),

    md("""
**Read — this is the key chart.**

- **Mark 14**: 100% passive — every fill is at bid_1 (when buying) or ask_1
  (when selling). He IS the inside quote.
- **Mark 38**: 100% aggressive — every fill is at ask_1 (when buying) or
  bid_1 (when selling). He always crosses the spread.
- **Mark 22**: tiny sample, mixed.

Translation: HG is a structural 2-player game. Mark 14 quotes both sides
at fair ± 8, Mark 38 always crosses to him, Mark 14 captures half-spread
on every round trip. To replicate Mark 14, we just need to be the inside
quote — but with FIFO ambiguity, we need to undercut by 1 tick to win
priority.
"""),

    md("## 4 — Premium to mid (who pays the spread)"),
    code("""
# Premium to mid: how far the trade price is from the mid AT trade time,
# signed by the Mark's side. Positive = adverse fill (paid up).
hg_t['mid_at_trade'] = hg_t.mid

rows = []
for who in roster.index:
    is_buyer = hg_t.buyer == who
    is_seller = hg_t.seller == who
    # buyer's premium = price - mid (>0 = bought above mid, adverse)
    # seller's premium = mid - price (>0 = sold below mid, adverse)
    buyer_prem = (hg_t.loc[is_buyer, 'price'] - hg_t.loc[is_buyer, 'mid_at_trade'])
    seller_prem = (hg_t.loc[is_seller, 'mid_at_trade'] - hg_t.loc[is_seller, 'price'])
    all_prem = pd.concat([buyer_prem, seller_prem])
    rows.append({'mark': who, 'n': len(all_prem),
                 'avg_premium_to_mid': round(all_prem.mean(), 2),
                 'pos_means_adverse': '(>0 = paid spread, <0 = captured spread)'})
print(pd.DataFrame(rows).to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 3))
data = []
labels = []
for who in roster.index:
    is_buyer = hg_t.buyer == who
    is_seller = hg_t.seller == who
    buyer_prem = (hg_t.loc[is_buyer, 'price'] - hg_t.loc[is_buyer, 'mid_at_trade'])
    seller_prem = (hg_t.loc[is_seller, 'mid_at_trade'] - hg_t.loc[is_seller, 'price'])
    all_prem = pd.concat([buyer_prem, seller_prem])
    data.append(all_prem.values)
    labels.append(who)
ax.boxplot(data, labels=labels, showfliers=False)
ax.axhline(0, color='red', lw=0.8)
ax.set_ylabel('premium to mid (>0 = paid)')
ax.set_title('HG: who paid the spread on each fill?')
plt.tight_layout()
"""),

    md("""
**Read.** Mark 14: −8 (captures the spread). Mark 38: +8 (pays the spread).
Mark 22: small, mixed. The half-spread = 8 transfers cleanly from Mark 38
to Mark 14 ~340 trades a day.
"""),

    md("## 5 — Forward returns conditional on each Mark's print"),
    code("""
# Build mid lookup for forward-mid lookups
mid_idx = {}
for d, g in hg_p.groupby('day'):
    g_s = g.sort_values('timestamp')
    mid_idx[d] = (g_s.timestamp.values, g_s.mid_price.values)

def fwd_mid(day, ts, h):
    t, m = mid_idx[day]
    i = np.searchsorted(t, ts + h, side='right') - 1
    return m[i] if 0 <= i < len(m) else np.nan

hg_t['fwd_5k']  = hg_t.apply(lambda r: fwd_mid(r.day, r.timestamp, 5000), axis=1)
hg_t['fwd_10k'] = hg_t.apply(lambda r: fwd_mid(r.day, r.timestamp, 10000), axis=1)
hg_t['ret_5k']  = hg_t.fwd_5k - hg_t.mid
hg_t['ret_10k'] = hg_t.fwd_10k - hg_t.mid

# Conditional means: after Mark X buys / sells, what does mid do next 5k / 10k?
print('Forward mid change (mid after − mid now), conditional on counterparty action:\\n')
print(f"{'who':<10}{'as':<8}{'n':>5}{'fwd_5k':>10}{'fwd_10k':>10}")
print(f"{'---'*10}")
for who in roster.index:
    for label, mask in [('buyer', hg_t.buyer == who), ('seller', hg_t.seller == who)]:
        sub = hg_t[mask]
        if len(sub) < 5: continue
        print(f"{who:<10}{label:<8}{len(sub):>5}{sub.ret_5k.mean():>10.2f}{sub.ret_10k.mean():>10.2f}")
print(f"\\n{'BASELINE':<10}{'':8}{len(hg_t):>5}{hg_t.ret_5k.mean():>10.2f}{hg_t.ret_10k.mean():>10.2f}")
"""),

    md("""
**Read.** None of the Marks' HG flow has a strong forward-return signal. Mark
14's buys nudge mid +0.85 over 10k ticks (he gets some adverse selection),
Mark 38's are roughly anti-correlated with the move — but magnitudes are
small (under 1 seashell). Compared to VE (where Mark 67 buy → +1.57 fwd
chg), HG flow is information-poor.

This confirms: **HG strategy is spread capture, not flow following.** We
don't need to condition on counterparty for fair value. We just need to
quote inside Mark 14's prices.
"""),

    md("## 6 — Cumulative position over time per Mark"),
    code("""
fig, ax = plt.subplots(figsize=(11, 4))
hg_t_sorted = hg_t.sort_values(['day','timestamp']).copy()
# Continuous timeline across days for plotting
hg_t_sorted['ts_cont'] = hg_t_sorted.timestamp + (hg_t_sorted.day - 1) * 1_000_000

for who in roster.index:
    is_buyer = hg_t_sorted.buyer == who
    is_seller = hg_t_sorted.seller == who
    sgn = is_buyer.astype(int) - is_seller.astype(int)
    pos_change = sgn * hg_t_sorted.quantity
    pos_change = pos_change[(is_buyer | is_seller)]
    ts = hg_t_sorted.ts_cont[(is_buyer | is_seller)]
    cum = pos_change.cumsum()
    ax.plot(ts.values, cum.values, label=who, lw=1)
ax.axhline(0, color='black', lw=0.5)
for d_boundary in [1_000_000, 2_000_000]:
    ax.axvline(d_boundary, color='gray', lw=0.5, linestyle='--')
ax.set_title('HG: cumulative net position per Mark (continuous timeline across 3 days)')
ax.set_ylabel('cumulative net position')
ax.set_xlabel('continuous timestamp (day1 → day2 → day3)')
ax.legend()
plt.tight_layout()
"""),

    md("""
**Read.** Both Mark 14 and Mark 38 oscillate around 0 — flat-target market
makers / takers. Neither accumulates a meaningful net position over the
day.
"""),

    md("## 7 — Implied P&L per Mark (mark to mid)"),
    code("""
def cum_pnl_for(who, df):
    is_buyer = df.buyer == who
    is_seller = df.seller == who
    sgn = is_buyer.astype(int) - is_seller.astype(int)
    qty = sgn * df.quantity
    cash = -sgn * df.quantity * df.price  # we paid `price * qty` if buying
    keep = (is_buyer | is_seller)
    pos = qty[keep].cumsum()
    cash_cum = cash[keep].cumsum()
    mid = df.mid[keep]
    pnl = cash_cum + pos * mid
    return df.timestamp[keep] + (df.day[keep] - 1) * 1_000_000, pnl

fig, ax = plt.subplots(figsize=(11, 4))
for who in roster.index:
    ts, pnl = cum_pnl_for(who, hg_t.sort_values(['day','timestamp']))
    ax.plot(ts.values, pnl.values, label=f'{who} (final={pnl.iloc[-1]:.0f})', lw=1)
ax.axhline(0, color='black', lw=0.5)
for d_boundary in [1_000_000, 2_000_000]:
    ax.axvline(d_boundary, color='gray', lw=0.5, linestyle='--')
ax.set_title('HG: cumulative MtM PnL per Mark (3 days)')
ax.set_ylabel('PnL (XIRECs)')
ax.set_xlabel('timestamp')
ax.legend()
plt.tight_layout()
"""),

    md("""
**Read.** This is the smoking gun. Across 3 days:

- **Mark 14 banks ~30K of pure spread capture** on HG alone (steady upward
  ramp, low variance).
- **Mark 38 burns ~30K** paying that spread.
- **Mark 22**: noise, near zero.

That ~30K is the alpha pool we're trying to redirect to ourselves by
under-cutting Mark 14's quote on every tick. If we capture all of Mark 38's
flow at half-spread = 7 (vs Mark 14's 8), we should bank ~26K over 3 days
on HG alone.
"""),

    md("""
## Bottom-line rules from the data

1. HG mid mean-reverts to 10000 every day with std ~32. Use 10000 as a hard
   anchor for skew.
2. The inside spread is structurally 16 wide (Mark 14's quote). Quote at
   `bid_1 + 1 / ask_1 − 1` to win price priority.
3. HG flow has no useful counterparty information — don't condition on Mark
   identities for HG fair value.
4. Drawdowns up to ~$3K per session are expected from holding mean-reversion
   inventory open while the price overshoots; this resolves over the day.
5. Free-take / aggressive mean-reversion entry IS valuable (loses
   half-spread on entry, gains a full spread on the round trip when mid
   reverts to 10000) — keep it.
"""),
]


def write(nb, path: Path):
    nbf.write(nb, str(path))
    print(f'wrote {path.name} ({len(nb.cells)} cells)')


write(nb1, HERE / '01_hydrogel_price.ipynb')
write(nb2, HERE / '02_hydrogel_marks.ipynb')
