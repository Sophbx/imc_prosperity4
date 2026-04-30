"""Plot Thor backtest PnL curves from the saved log file."""
import io
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).parent
LOG = HERE / "thor_backtest.log"

with open(LOG) as f:
    data = json.load(f)

act = pd.read_csv(io.StringIO(data['activitiesLog']), sep=';')
print(f"Loaded {len(act)} rows, days: {sorted(act['day'].unique())}")

# Per-product PnL trajectory by day
products = sorted(act['product'].unique())
days = sorted(act['day'].unique())

# Total PnL = sum of per-product PnL at each timestamp
total = act.groupby(['day','timestamp'])['profit_and_loss'].sum().reset_index()

# Continuous timeline (days stitched together)
total['cont_ts'] = total['timestamp'] + (total['day'].astype(int) - days[0]) * 1_000_000
print(f"Final PnL: ${total['profit_and_loss'].iloc[-1]:.0f}")

# ----- Plot 1: Total PnL trajectory -----
fig, axes = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={'height_ratios': [2, 1]})

ax = axes[0]
ax.plot(total['cont_ts'] / 1e6, total['profit_and_loss'], color='#1f77b4', lw=1.5, label='Total PnL')

# Day boundaries
for d_idx, d in enumerate(days[:-1]):
    boundary = (d_idx + 1) * 1.0  # in millions of ts
    ax.axvline(boundary, color='gray', lw=0.5, ls='--', alpha=0.6)

# Day labels
for d_idx, d in enumerate(days):
    mid_x = d_idx + 0.5
    day_total = act[act['day']==d].groupby('timestamp')['profit_and_loss'].sum().iloc[-1]
    cumul = total[total['cont_ts'] <= (d_idx+1)*1e6]['profit_and_loss'].iloc[-1]
    ax.text(mid_x, ax.get_ylim()[1]*0.95 if d_idx>0 else 5000, f'Day {d}',
            ha='center', fontsize=11, color='gray', alpha=0.8)

ax.set_title(f'Trader Thor — Backtest PnL on R4 days 1-3 (final ${total["profit_and_loss"].iloc[-1]:,.0f})',
             fontsize=14, fontweight='bold')
ax.set_xlabel('continuous timestamp (× 1M ticks)')
ax.set_ylabel('cumulative PnL (XIRECs)')
ax.grid(alpha=0.3)
ax.legend(loc='upper left')
ax.axhline(0, color='black', lw=0.5)

# ----- Plot 2: Per-product PnL stacked -----
ax2 = axes[1]
key_products = ['HYDROGEL_PACK', 'VELVETFRUIT_EXTRACT', 'VEV_4000',
                'VEV_5100', 'VEV_5200', 'VEV_5300', 'VEV_5400', 'VEV_5500']
colors = plt.cm.tab10.colors

for i, p in enumerate(key_products):
    if p not in products:
        continue
    sub = act[act['product']==p].copy()
    sub['cont_ts'] = sub['timestamp'] + (sub['day'].astype(int) - days[0]) * 1_000_000
    sub = sub.sort_values('cont_ts')
    final_pnl = sub['profit_and_loss'].iloc[-1]
    ax2.plot(sub['cont_ts'] / 1e6, sub['profit_and_loss'],
             color=colors[i % len(colors)], lw=1.0, label=f'{p} (${final_pnl:,.0f})')
for d_idx, d in enumerate(days[:-1]):
    boundary = (d_idx + 1) * 1.0
    ax2.axvline(boundary, color='gray', lw=0.5, ls='--', alpha=0.6)
ax2.set_xlabel('continuous timestamp (× 1M ticks)')
ax2.set_ylabel('per-product PnL')
ax2.set_title('Per-product PnL trajectories')
ax2.grid(alpha=0.3)
ax2.legend(loc='upper left', fontsize=8, ncol=2)
ax2.axhline(0, color='black', lw=0.5)

plt.tight_layout()
out = HERE / "thor_pnl_curves.png"
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f"Saved: {out}")

# ----- Plot 3: Per-day per-product bar chart -----
fig, ax3 = plt.subplots(figsize=(13, 6))
last = act.sort_values('timestamp').groupby(['day','product']).tail(1)
pivot = last.pivot(index='product', columns='day', values='profit_and_loss').fillna(0)
pivot = pivot.loc[[p for p in key_products if p in pivot.index]]
pivot.plot(kind='bar', ax=ax3, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax3.set_title('Trader Thor — Per-day per-product PnL', fontsize=14, fontweight='bold')
ax3.set_ylabel('PnL (XIRECs)')
ax3.set_xlabel('Product')
ax3.axhline(0, color='black', lw=0.5)
ax3.grid(axis='y', alpha=0.3)
ax3.legend(title='Day')
for container in ax3.containers:
    ax3.bar_label(container, fmt='%.0f', fontsize=7, padding=2)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
out2 = HERE / "thor_pnl_per_day.png"
plt.savefig(out2, dpi=120, bbox_inches='tight')
print(f"Saved: {out2}")
plt.close('all')