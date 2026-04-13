# Round 0 (Tutorial) — Writeup

A short note for teammates on where our Round 0 algorithm stands, how we got there, and what to keep in mind before Round 1 kicks off.

## Products

Round 0 hands us two products to trade, both with a position limit of ±80:

- **EMERALDS** — extremely stable asset. True price is essentially locked at 10,000 (96.7% of ticks). Bot order book has a deep wall at 9990/10010 and a thinner noise quote at 9992/10008. There is no directional risk here, only spread capture.
- **TOMATOES** — volatile, drifting fair value (std ≈ 20 over a day). Same wall/noise quote structure but the walls move. This is where every real strategy decision matters.

## Final algorithm: `trader_final_v1.py`

Our current submission. Both products use a two-phase loop each tick:

1. **TAKE** — pick off any mispriced bot quotes (buy below our fair value, sell above).
2. **MAKE** — post resting limit orders inside the bot spread to capture spread passively.

The interesting part is how we pick the quote prices.

### EMERALDS — Wall Mid + split-quote

Fair value is a constant 10,000. We quote dynamically one tick inside whichever is better — the noise quote (9992/10008) or the wall (9990/10010). On each side we actually post **two** layers: one at the noise price itself and one one tick above it. In optimistic backtester matching this captures every historical fill at a slightly better edge; in strict matching the inner layer still catches everything as before. Strictly ≥ the single-layer version, never worse.

On the sample days this saturates the product: **14,945 conservative** (7,182 day-2 + 7,763 day-1). We proved arithmetically this is the ceiling — every single historical EMERALDS trade that could be caught × 7 ticks of edge = exactly 14,945. No room left.

### TOMATOES — adaptive noise-quote tracking

Fair value is `wall_mid = (bid_wall + ask_wall) / 2`, where `bid_wall` is the deepest visible bid and `ask_wall` the deepest ask. We do **not** use an EMA — turns out our original `alpha=0.3` EMA has a worse mean-squared error than just using the raw `wall_mid` directly. TOMATOES has lag-1 autocorrelation of -0.18 (weak mean reversion), and smoothing this with an EMA only lags the signal.

For the MAKE phase we quote at:

```python
bid_price = max(bid_wall + 2, best_bid + 1)
ask_price = min(ask_wall - 2, best_ask - 1)
```

In plain words: **always one tick inside the best visible quote**, floored at the wall edge. There is a 6-tick "virgin zone" between the noise bot (which quotes ~1 tick inside the wall) and the true wall_mid where nobody else is posting — we plant ourselves exactly one tick ahead of the noise bot in queue priority, capture all the flow that was hitting the noise bot, and pocket a bigger edge per fill.

A small library of defensive guards lives on top:

- Refuse to post on a side where the noise bot is absent (`best_bid == bid_wall`), so we don't end up as the only liquidity in the book.
- Pre-clamp combined TAKE + MAKE volume against the position limit, or the exchange silently drops every order for the product.
- Use `math.floor`/`math.ceil` for unwind prices when `wall_mid` is a half-integer.
- Replace every `except: pass` with an error counter written to `traderData` — any failure leaves a visible fingerprint in the logs.

## How we got here (journey)

| Version | Conservative PnL | Key change |
|---|---:|---|
| `trader.py` (baseline) | 11,890 | Fixed spread + EMA + linear skew |
| `trader_v2` to `v4` | — | Early experiments, all worse than baseline |
| `trader_eme6_tomato_v1` | — | Friend's finding: EMERALDS `spread=6` > `spread=2` (partial fix) |
| `trader_wallmid_v1` | 14,945 | Full Hedgehogs port. EMERALDS fixed, TOMATOES broke (0) |
| `trader_hybrid_v1` | 23,116 | Wall Mid for EMERALDS + original EMA for TOMATOES |
| `trader_phase1_B` | 28,584 | Noise-quote inner quoting on TOMATOES |
| `trader_phase2_D` | 31,657 | Adaptive noise tracking |
| **`trader_final_v1`** | **31,671** | D + E split-quote + defensive guards |

All numbers are from [Jmerle's Prosperity 3 backtester](https://github.com/jmerle/imc-prosperity-3-backtester) (last year's version — we haven't found a Prosperity 4 equivalent yet, and we locally patched this one to accept our products) run on the day -1 + day -2 sample CSVs with `--match-trades worse` (conservative matching). Optimistic matching peaks at 33,950. **Treat these as reference numbers only** — see the "Backtester and expected live PnL" section below for the full caveats.

Two external things shaped the breakthrough:

1. **Frankfurt Hedgehogs' Prosperity 3 writeup** (`timodiehm/imc-prosperity-3`). Their Wall Mid concept and the "quote one tick inside the best price" mechanic are the entire foundation of our EMERALDS and TOMATOES strategies.
2. **A multi-agent debate run via Claude Code.** After hitting the baseline ceiling we dispatched parallel agents with deliberately different philosophies — a rigorous statistician (made the "delete the EMA" call), a microstructure purist (found the noise-quote virgin zone), a signal hunter (proved there are no Olivia-style bots in our Round 0 data), an attacker (read the final code line-by-line and surfaced the bugs we eventually patched), and two focused improvers. A synthesis pass merged the survivors into `trader_final_v1.py`. Internal design docs for every agent live in `Round0/agents/` (gitignored, local only — ask Nick if you want to read them).

## Backtester and expected live PnL

We are currently running [Jmerle's **Prosperity 3** backtester](https://github.com/jmerle/imc-prosperity-3-backtester) — the same tool last year's top teams (including Frankfurt Hedgehogs) used. There is no official Prosperity 4 equivalent yet as far as we can tell, so this is a stop-gap. **We locally patched it** to accept EMERALDS and TOMATOES in its hardcoded `LIMITS` dict; those two lines aside, the tool is unchanged from upstream. Please treat every backtester number in this repo as a **rough reference**, not ground truth — the P3 simulation mechanics may differ from P4 in ways we haven't verified, and our patch was deliberately minimal so the core matching logic is still whatever Jmerle wrote for last year.

Beyond the P3-vs-P4 mismatch, two more reasons to distrust absolute numbers:

1. **Tick count**. Sample data is two full days (~20,000 ticks total). The live Prosperity server only runs ~2,000 ticks per submission, so divide backtester numbers by ~10 for a first-order live estimate.
2. **Optimistic matching**. The `--match-trades worse` mode still uses strict `<` price comparison and does **not** model queue priority. It's approximately conservative but still ~30-50% optimistic relative to realistic live fills.

So for `trader_final_v1`:

- Backtester conservative: **31,671**
- Divided by 10 (tick count): ~3,167
- After realism discount (×0.6-0.7): **~1,900-2,200**

For comparison, the original `trader.py` was scoring ~1,000-1,200 live. That suggests a rough **2× lift** _if_ the live market behaves anything like the sample data — which we can't fully verify until we actually upload. Use backtester deltas for **relative** comparisons between trader versions, not as a live PnL forecast.

## Known residual risks

Worth mentioning before we upload:

- **Noise bot assumption.** If the live noise bot behaves differently (disappears, becomes informed flow, or quotes at a different offset), our TOMATOES edge could evaporate. The current adaptive logic handles "different offset" gracefully, but a disappearing noise bot or an informed one would hurt.
- **Adverse selection.** Measured in backtest: wall_mid drifts -0.21 ticks over 20 ticks after we fill a MAKE BUY (n=406). Eats 3-4% of per-trade edge. Could be worse live.
- **Backtester optimism / P3-vs-P4 gap.** See above — we're using last year's backtester with a local patch, so absolute numbers shouldn't be trusted. Treat them as a relative comparison between trader versions.
- **Adverse-selection brake (skipped).** We wrote it, measured it, and removed it — it hurt conservative PnL by 341 points (whipsaw). State variables are still persisted so we can turn it back on with a tighter trigger if live logs suggest it.

## What to try for Round 1

When Round 1 data arrives:

1. Re-run the EDA from `phase1_A` on the new products. The lag-1 autocorrelation test + cross-day consistency check is a useful starting template.
2. Port the Wall Mid + adaptive noise-quote tracking to any new product that has the same book structure.
3. Watch the trades CSV for non-null `buyer`/`seller` IDs — Round 0's were all NaN, so no Olivia-style signals were possible. If Round 1 has IDs, there's a ~5-8k/day alpha waiting (per Hedgehogs' Squid Ink section).
4. Keep the `Round0/agents/` workflow for brainstorming — running parallel agents with adversarial prompts surfaced insights we wouldn't have found manually.

## Files worth knowing about

- `trader_final_v1.py` — the current submission. All future edits go here or on a new branch.
- `trader_phase1_B.py` — the strategy inflection point. Read this to understand the noise-quote virgin zone mechanic.
- `trader_phase2_D.py` — the adaptive tracking refinement.
- `trader_hybrid_v1.py` — last pre-debate baseline, useful as a rollback target.
- `Round0/Data/` — sample CSVs for backtesting.

Anything else, ping Nick.
