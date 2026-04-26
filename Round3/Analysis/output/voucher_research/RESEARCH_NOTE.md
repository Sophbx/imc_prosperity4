# Voucher Book Trading Strategy — Research Note

**Round 3, Prosperity 4. Archetype B: IV scalping (Hedgehogs-style).**

This note is the deliverable for the deep-research phase. It records what
worked, what didn't, why, and the recommended design for the implementation
phase.

---

## TL;DR

- **Trade these strikes**: VEV_5000, VEV_5100, VEV_5200, VEV_5300, VEV_5400.
  Optionally include 4500 / 5500 with 0 expected contribution but no harm.
- **Skip definitively**: VEV_4000 (intrinsic-dominated, dead signal),
  VEV_6000, VEV_6500 (dead OTM, mid pinned at 0.5).
- **Smile fit**: parabola in (moneyness, IV) space, refit every tick from the
  4 inner strikes [5000, 5100, 5200, 5300]. Fit is very tight in-sample
  (residual std < 0.02 IV vol units, < 1 tick price units), but extrapolates
  poorly to outer strikes (5400 has structural -2.3 tick bias).
- **Mechanism**: passive maker, post `best_bid+1 / best_ask-1` clamped
  by `floor(fair) / ceil(fair)`. Don't take. Don't hedge.
- **Backtest**: ~$2400 over 3 days (~$800/day), 110 fills. Day-2 (when TTE is
  shortest) carries most of the PnL.
- **Hedging policy: NONE** — empirically destroys PnL because each VELVETFRUIT
  hedge crosses the 1-tick spread at large size.
- **The big honest finding**: the per-tick smile-deviation signal is NOT
  mean-reverting in the actionable sense. The lag-1 negative autocorrelation
  is in voucher mid *returns* (-0.10 to -0.28), not in IV deviations (whose
  lag-1 is *positive*, 0.10–0.61). The Hedgehogs writeup explicitly mentions
  this. Our edge comes from being a quoting market maker, not an IV scalper.

---

## 1. Final Strike Selection

### Method
For each of the 10 voucher strikes I tested:

1. **Smile residual fit quality**: fit a parabola IV ~ a*moneyness² + b*moneyness + c
   per tick on the inner strikes; measure each strike's deviation `dev = iv -
   iv_fit`. Stationarity / variance / autocorrelation of dev.
2. **Counterfactual scalp PnL**: simulate "buy when ask < fair-θ, sell after N
   ticks at then-current bid" for various θ and N. For all out-of-sample
   strikes this loses (cross-spread > follow-through). For in-sample strikes
   the residuals are tiny so few signals.
3. **Negative-return autocorrelation**: lag-1 ACF of `voucher_mid.diff()`
   per strike (Hedgehogs' actual statistical justification).
4. **Bot trade frequency**: how many times per day a market trade actually
   occurs at each strike (caps maker fill rate).
5. **Backtester**: brute-force run the prototype trader with each strike
   set and read the per-strike PnL.

### Findings per strike

| Strike | spread | mean IV | empir δ | mid lag1-ACF | bot trades/day | verdict |
|--------|--------|---------|---------|--------------|----------------|---------|
| 4000   | 21     | 0.09    | 1.00    | -0.28 | 172 | dead — mid≈intrinsic, IV super noisy. SKIP. |
| 4500   | 16     | 0.19    | 1.00    | -0.22 | n/a | wide spread but the bot rarely hits inside-quotes; backtest shows 0 fills. SKIP. |
| 5000   | 6      | 0.24    | 0.92    | -0.10 | n/a | trade. Backtest -$31 over 3 days (essentially flat). |
| 5100   | 4      | 0.24    | 0.78    | -0.09 | n/a | trade. Backtest minimal — 5100 doesn't fill much against our maker because spread too tight, but smile fit anchors fair value for the smile so it's still useful as a fit point. |
| 5200   | 3      | 0.24    | 0.57    | -0.14 | 3-8 | trade. **+$270 over 3 days**, 14 fills. |
| 5300   | 2      | 0.25    | 0.33    | -0.21 | 37-45 | trade. **+$1717 over 3 days**, the dominant source of PnL. |
| 5400   | 1.4    | 0.23    | 0.13    | -0.25 | 64-81 | trade with caution. **+$418 over 3 days** but day-0 was -$59. |
| 5500   | 1.1    | 0.25    | 0.05    | -0.24 | 81 | small +$24, can include but unlikely to matter. |
| 6000   | 1      | 0.40    | 0.00    | n/a (no moves) | 91 | dead, mid stuck at 0.5. SKIP. |
| 6500   | 1      | 0.60    | 0.00    | n/a | 91 | dead. SKIP. |

### Recommendation

**Trade {5000, 5100, 5200, 5300, 5400}.** Strike 5300 is the goldmine
(~80% of total PnL); 5400 contributes meaningfully but with day-to-day
variance; 5200 is a small steady contributor; 5000/5100 are essentially
break-even but valuable as smile-fit inputs. Strikes 4500 and 5500 can be
included for completeness with minor expected impact.

The Hedgehogs strategy on Prosperity 3 made most of its money on the most
liquid ATM strike (10000-strike, with underlying mean ~10000). Our analogue
is 5300 (underlying mean 5267). This pattern holds — most edge concentrates
at one or two ATM-ish strikes.

### The ITM-noise check

Per the brief: confirm deep-ITM IV is unactionable. From `dev_acf.csv`:

- VEV_4000: dev_std = 0.71 (vol units), but mean = -0.07 — pure noise around intrinsic.
- VEV_4500: dev_std = 0.0016, virtually zero — mid is pinned near intrinsic.
- The taker scalp simulation `scalp_pnl_sim.csv` shows VEV_4000 at every theta: pnl_taker_per_trade ≈ -20 (full bid-ask cost lost on every trade). **Confirmed dead.**

---

## 2. Smile Fit Method

### Final choice

**Parabola in (S/K, IV) space, ordinary least squares, refit every tick on
strikes [5000, 5100, 5200, 5300].**

### Why this fit set?

I tested 5 candidate sets (`smile_fit_comparison.csv`):

| Set | strikes | in-sample dev_std (price) | extrapolation health |
|-----|---------|---------------------------|----------------------|
| S1_inner3 | 5100,5200,5300 | ~1e-5 (perfect 3-pt fit) | terrible: 5400 mispx -3.4, 5500 -0.25 |
| **S2_inner4** | **5000,5100,5200,5300** | 0.07-0.65 | acceptable: 5400 mispx -3.9, 5500 -0.91 |
| S3_inner5 | + 5400 | 0.14-0.79 | mostly OK but adds noise |
| S4_inner7 | full set | 0.07-2.0 | wide residuals; 5400 is biased -2.3 |
| S5_skip5400 | drop 5400 | similar to S2 | similar |

S2 (inner 4) is the best compromise between fit quality and stability. Adding
5400 (S3) only marginally widens residuals on 5000-5300 but doesn't help.

### Stability across days

**Smile coefs are NOT stable across days.** From `voucher_research4.py`:

```
day 0: a_mean = -1.92, b_mean = +3.97, c_mean = -1.81
day 1: a_mean = +4.10, b_mean = -8.49, c_mean = +4.64
day 2: a_mean = +7.30, b_mean = -14.99, c_mean = +7.94
```

The parabola literally inverts between days. **Implication**: do NOT calibrate
smile parameters offline once and use them — they have to be refit every
tick. Also: do NOT extrapolate the smile to strikes far outside the fit range.
Within [5000, 5300] the parabola is well-pinned by 4 data points, but
predicting IV at 5400 or 5500 is unreliable.

### Refit cadence

**Every tick.** The fit is cheap (4-point quadratic OLS, closed-form via
3x3 linear solve). Doing it less often loses information when the underlying
moves.

### Alternative forms tested

- Parabola in (log(S/K), IV) — comparable performance, no advantage.
- SVI / SABR — overkill, only 4 fit points; would overfit.

---

## 3. Trading Mechanism

### Architecture

**Pure passive maker.** No taker leg. No hedge.

```python
# Per tick, per tradable strike:
fair = bs_call_price(S, K, T, iv_fit_smile(moneyness))
buy_price = min(best_bid + 1, floor(fair))
sell_price = max(best_ask - 1, ceil(fair))
# (then sanitize: not crossing, valid prices)
```

### Why this works

- The **`floor(fair)`** clamp prevents posting buys above what the smile thinks
  is fair value. So when smile says voucher is rich, buy_price gets clamped
  *down* below the spread → no fill. We don't pay to accumulate longs in a
  rich market.
- Symmetrically for the sell side.
- When smile agrees with the market (fair ≈ touch_mid), our quote is just
  inside the touch (best_bid+1 / best_ask-1) and we MM normally.
- The voucher mid mean-reverts (lag-1 ACF -0.1 to -0.28), so getting filled at
  best_bid+1 typically gives us a small profit on the next move. This is the
  actual edge.

### Maker vs taker — empirical comparison

I tested adding a stat-arb taker leg (buy when best_ask < fair - 3 ticks,
sell when best_bid > fair + 3 ticks):

| variant | total PnL | n_trades | comment |
|---------|-----------|----------|---------|
| v4 baseline (maker only) | $2374 | 105 | clean |
| v6 + taker thr=3 | $2625 | 158 | marginal +$250 vs maker-only, MORE risk |
| v6 + taker thr=5 | $1103 | 147 | LOSES on 5400 (-$853), structural smile bias trips taker |

**Decision: NO taker.** The marginal upside is small and the smile fit error
on outer strikes (5400) systematically tricks the taker into lifting offers
at a "perceived" -3 tick discount that is actually a structural bias. This is
fragile and dangerous.

### Sizing

- `QUOTE_SIZE = 30` per side baseline.
- Inventory skew: when |position| > 80, reduce same-side quote, increase
  opposite-side. Standard MM hygiene.
- `MAX_POS = 250` (limit is 300; we leave 50 of safety margin).
- Per-strike limit is 300, we never approach it on these strikes (positions
  rarely exceed 100 in backtest).

### Inventory rules

- Hard cap: when |pos| ≥ MAX_POS, kill the same-side quote.
- Soft skew: shrink the buy quote linearly when long; vice versa.
- No end-of-day flatten (rounds run independently, position carries over).
  This is consistent with HYDROGEL v2b and standard P4 convention.

---

## 4. Hedging Decision: NONE

### Empirical test

I implemented v7 with a delta hedger: aggregate position-weighted BS delta,
hedge with VELVETFRUIT taker orders when |aggregate_delta| > 30.

| variant | voucher PnL | VELVETFRUIT PnL | total |
|---------|-------------|-----------------|-------|
| v4 (no hedge) | $2374 | $0 | $2374 |
| v7 (with hedge, threshold=30) | ~$2374 | -$2572 | -$198 |

**Hedging destroys PnL by ~$2.5k.** Each hedge crosses the underlying
spread (typically 1-2 ticks at large size, 1 tick at touch). The voucher
positions are inherently delta-bounded (they're calls on a finite move) and
the gamma exposure is small at TTE 5-8 days. Cost > benefit.

This matches the Hedgehogs writeup's explicit observation:
> "explicit delta hedging would have been prohibitively expensive bid-ask
> spreads. It was rather a hedge against bad luck."

### The "lightweight" alternative

Hedgehogs used the deepest ITM call as a side-trade for delta exposure
(not a hedge). I considered using VEV_4500 (deepest tradable ITM) for that
purpose, but:
- VEV_4500 has 16-tick spread → high entry/exit cost
- Spread cost dominates any expected directional return
- Implementing it adds complexity for unclear benefit

**Recommendation**: leave VELVETFRUIT untouched. The strategy is naturally
delta-light because ATM voucher deltas average to ~0.4 across 5000-5400 and
positions remain small.

---

## 5. Counterfactual / Prototype Backtest Numbers

### Final config benchmark

```
Strikes traded: {5000, 5100, 5200, 5300, 5400}
Smile fit: parabola in (S/K, IV), refit per tick on [5000-5300]
Quote: best_bid+1 / best_ask-1 clamped by floor(fair) / ceil(fair)
Quote size: 30, inventory skew above |pos|=80
Position cap: 250 (limit is 300)
Take threshold: disabled
Hedge: disabled

3-day backtest result:
  Day 0 (TTE 8): -$21.5  (20 trades)
  Day 1 (TTE 7): +$685.5 (44 trades)
  Day 2 (TTE 6): +$1710  (41 trades)
  Total:         +$2374  (105 trades)

Per-strike PnL:
  VEV_5000: -$31.5  (in-sample fit anchor; spread too tight)
  VEV_5100:  $0     (in-sample fit anchor; few fills)
  VEV_5200: +$270.5 (occasional MM)
  VEV_5300: +$1717  (workhorse)
  VEV_5400: +$418   (volatile but contributing)
```

### Sanity check

- $2374 / 30000 ticks = $0.08 per tick — meager but consistent with passive
  MM in low-trade-frequency markets.
- Per-fill PnL = $2374 / 105 = $22.6 — reasonable. Our quotes are at
  best_bid+1 / best_ask-1, so the round-trip captures spread - 2 ticks
  ≈ 1-3 ticks in a 3-5 tick spread market, plus some directional benefit
  from lag-1 mean reversion.

### Caveats

- 105 trades is a small sample. PnL is sensitive to a small number of
  good/bad fills. The std of daily PnL across days is ~$700.
- Live round 3 has TTE 5d (vs backtest 6-8d). Shorter TTE → higher gamma →
  more dramatic price moves → potentially more volatile PnL but also more
  fills.
- The Round 3 LIVE day will be a single 1M-tick day with TTE 5→4. The PnL
  pattern (day 0 < day 1 < day 2 in backtest) suggests we want to wait for
  TTE to shrink, which it does naturally.

---

## 6. Risks and Unknowns

### Smile-fit fragility

The fit coefs invert sign between days. If on the live day the smile shape
is unlike anything in 8/7/6-day TTE history, the floor(fair) clamp may
mis-direct quotes. Mitigation: don't trust the smile beyond the fit set.
Skip a side if the implied IV is outside [0.05, 1.5] (sanity bounds).

### Adverse selection

Bots trade only ~37/day on VEV_5300 and ~80/day on VEV_5400. That's tiny.
Maker fills are mostly when our limit catches an "informed" trade. Our
positive PnL says the smile-floor effectively filters out the worst adverse
selection (we don't accumulate the wrong side of an informed flow). But
there's no guarantee live bots will behave like backtest bots.

### Position carry across days

We don't flatten EOD. Day 2 backtest ended with non-zero positions on
5300/5400. In live, we'll carry positions into the live day. This means the
live trader needs to start trading with whatever inventory accumulated from
the historical replay (or be initialized fresh — depending on the
competition rules). Verify with the platform before submission.

### TTE schedule edge cases

Day inference uses `state.timestamp` rollover detection. If the round
starts mid-tick or the platform's day numbering differs, our TTE
computation is wrong → smile and fair price are wrong → quotes are wrong.
The Trader class must be tested across the day boundary explicitly in the
implementation phase.

### IV solver robustness

`implied_vol` uses Brent-style bisection with bounds [1e-3, 2.5]. On the
live day, if a voucher mid drops below intrinsic momentarily, IV returns
NaN and that strike is dropped from the fit. Need to verify the smile fit
is robust to occasional 1-strike dropouts.

### Liquidity asymmetry

The 5300 trades come almost entirely from one or two large bot trades per
day. If the live day has fewer of those, the strategy makes less. If the
live day has more or larger trades against our improved quotes (more
adverse selection), we could lose more.

### Parameter sensitivity

I did not exhaustively grid-search parameters. The current settings are
reasonable but not provably optimal. The implementation phase should run
parameter sweeps on:
- `FAIR_GUARD` ∈ {-0.5, 0, 0.5, 1.0}: I used 0; raising it tightens
  quote-aggression and may lose fills, lowering it lets us quote above fair.
- Strike set: {5000-5300} is conservative ($1956), {5000-5400} is mid
  ($2374), {4500-5500} is loose ($2398). Marginal impact.
- Quote size: 30 was fine; the bot fill rate caps total volume, so larger
  sizes don't help.

---

## 7. Files Added

In `Round3/Analysis/output/voucher_research/`:

- `iv_full_resolution.csv` — full per-tick (day, ts, strike) IV computed
  from voucher mid against underlying mid. ~300k rows. Replaces the
  existing `iv_timeseries.csv` (which was 10x downsampled).
- `smile_fit_residuals_A.csv` — per-strike in-sample residual stats for
  the [4500, 5000, 5100, 5200, 5300, 5400, 5500] fit (S4).
- `smile_fit_A_coefs.csv` — per-tick (day, ts) parabola coefficients for
  the S4 fit. ~30k rows.
- `smile_fit_comparison.csv` — per-strike residual stats for 5 alternative
  fit sets (S1 through S5). Used to choose S2_inner4.
- `dev_acf.csv` — per-strike autocorrelation of dev (with S4 fit).
  Confirms dev is positively autocorrelated (NOT mean-reverting) at lags
  1, 5, 10, 50.
- `scalp_pnl_sim.csv` — per-strike taker-style scalp PnL across thresholds
  / hold horizons. Confirms naive taker is universally losing.
- `scalp_inventory_sim.csv` — inventory-aware scalp PnL across thresholds /
  modes. Confirms inventory-based scalp on smile signal is losing.
- `taker_scalp_sim.csv` — refined taker scalp using S2 fit. Marginally
  positive on 5100/5300 only at very long holds.
- `backtest_results.csv` — summary of all prototype trader runs.
- `RESEARCH_NOTE.md` — this document.

Test traders (in `/tmp/`, NOT committed):
- `trader_voucher_proto.py` — initial naive smile-margin maker (failed).
- `trader_voucher_v2.py` — first working version with FAIR_GUARD=0.5 ($182).
- `trader_voucher_v4.py` — **best clean version**, FAIR_GUARD=0 ($2374).
- `trader_voucher_v5.py` — asymmetric quote tests (no improvement).
- `trader_voucher_v6.py` — taker leg added (marginal but risky).
- `trader_voucher_v7.py` — delta hedge added (destructive, -$2572 on
  VELVETFRUIT).

The implementation phase should start from the v4 architecture but with
re-vetted day-tracking in `traderData` and explicit unit tests at the
day-boundary timestamp rollover.

---

## Appendix: Why the smile-as-edge story doesn't work

The Hedgehogs writeup describes IV scalping as: fit smile → detrend →
scalp deviations from the fitted curve. This sounds clean, but on close
inspection of P4 data:

1. **In-sample residuals are tiny** (< 1 tick after S2 fit on
   5000–5300). With 4-point parabola, the fit is nearly perfect; there's
   no actionable deviation to scalp on the in-sample strikes.
2. **Out-of-sample (5400, 5500) residuals are LARGE but BIASED, not
   mean-reverting.** Strike 5400 has -2.3 mean residual that persists
   across all 30k ticks — that's the smile fit failing to extrapolate,
   not the market mispricing the voucher. Trading against this bias
   loses money.
3. **The dev autocorrelation at lag 1 is POSITIVE** (0.10 on most strikes,
   up to 0.61 on VEV_5100). Positive ACF means the deviation
   persists/trends, not mean-reverts. So scalping deviations is exactly
   the WRONG strategy.
4. **The actual mean reversion is in voucher mid RETURNS.** Lag-1 ACF of
   `mid.diff()` is -0.10 to -0.28 across all strikes. This is just generic
   short-term price mean reversion (microstructure-level), not anything
   IV-specific. It's captured naturally by passive market making, not by
   smile-deviation scalping.

So our strategy is: **maker MM, with smile as a sanity guard, exploiting
generic short-term mid mean reversion.** The "IV scalping" label is a
misnomer; we should mentally re-frame it as "smile-anchored market making."

---

## Decision summary for implementation phase

| Decision | Final answer |
|----------|--------------|
| Strikes traded | {5000, 5100, 5200, 5300, 5400} |
| Smile fit | parabola in (S/K, IV); refit every tick on [5000, 5100, 5200, 5300] |
| Quote logic | passive maker `best_bid+1 / best_ask-1` clamped by `floor(fair) / ceil(fair)` |
| Taker logic | NONE |
| Hedging | NONE — VELVETFRUIT not used |
| Quote size | 30 per side |
| Position cap | 250 (hard) |
| Inventory skew | shrink same-side quote when |pos| > 80 |
| Day inference | timestamp rollover detection in traderData |
| IV bounds | [0.05, 1.5] sanity check |
| Min spread to quote | 2 |
| Expected PnL | ~$700-1000/round (from $2374 over 3 backtest days, scaled) |
