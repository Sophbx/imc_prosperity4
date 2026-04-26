# Frankfurt Hedgehogs (P3, 2nd place) — Round 3 Code Review

Source: `/Users/nickzhu/Desktop/NYU/2026 Spring/IMC Prosperity/imc-prosperity-3/FrankfurtHedgehogs_polished.py`, class `OptionTrader` (lines 559-771), inheriting from `ProductTrader` (line 101).

This review is grounded in the **code** (not the writeup). Every formula and threshold quoted below is verbatim from the source.

---

## 1. Executive summary

After reading every method, the Hedgehogs option engine is conceptually simple and very tightly coded. Most of it is already in our `trader_combined_v1.py` in some form, but **three techniques are clearly portable and not yet in our code**:

| Rank | Technique | Estimated lift |
|------|-----------|----------------|
| 1 | **EMA-of-residual fair-value correction** (`mean_theo_diffs`) | Modest but real. Removes the structural -2.3-tick bias on VEV_5400 that we currently work around by disabling the taker on extrapolated strikes. Re-enables a low-edge taker on 5400 safely. Plausibly +$200-500/day on day 2 of the backtest. |
| 2 | **EMA-of-absolute-residual gate** (`switch_means` >= 0.7) — only run scalping when recent dev volatility justifies it | Risk-control. Auto-disables IV scalping during quiet stretches and re-enables when dev volatility picks up. Probably zero PnL impact in our setting (we're not really IV-scalping anyway), but cheap insurance for a v3 if we re-enable any taker logic. |
| 3 | **Underlying EMA mean-reversion overlay** (their `get_mr_orders` for the 9500 strike) — combine `ema_o_dev` of the underlying with per-option `theo_diff - mean_theo_diff` to drive a directional position on the deep ITM call | Could be additive if VELVETFRUIT shows mean reversion at our 1-tick spread. Empirically uncertain; needs a quick ACF check on VELVETFRUIT mid before committing. Best case +$300-600/day from a small directional flow on VEV_4000. |

Everything else they do is either redundant (we have it), inapplicable (different bot population, different smile shape), or actively rejected by our prior research (e.g., aggressive delta hedging).

**Recommended action**: Build a `trader_combined_v2.py` adding only Technique 1 (residual EMA correction on the 5400 fair price, gated on 5300/5400 only). Skip 2 and 3 unless time permits.

---

## 2. Technique-by-technique

For each technique I describe (a) **what the code does mechanically**, (b) **what we have**, (c) **portability**, and (d) **implementation outline + lift**.

### 2.1 Smile fit: hard-coded coefficients, not refit per tick

**Code** (`get_option_values`, lines 583-587):
```python
def get_iv(St, K, TTE):
    m_t_k = np.log(K/St) / TTE**0.5
    coeffs = [0.27362531, 0.01007566, 0.14876677]   # from the fitted vol smile
    iv = np.poly1d(coeffs)(m_t_k)
    return iv
```
The smile is a fixed parabola in `log(K/S) / sqrt(TTE)` space (sometimes called "log-moneyness over sqrt-time"). Coefficients were calibrated offline once across history and **never refit during the round**. There is no rolling fit, no per-tick OLS — just `a*x^2 + b*x + c` evaluated each tick.

**Our code** fits a fresh parabola in `(S/K, IV)` space every tick using OLS over the inner 4 strikes (`fit_parabola` in `trader_voucher_v1.py:88`, called from `_fit_smile`).

**Portable?** No — and not in the direction they did it. Our prior research (`Round3/Analysis/output/voucher_research/RESEARCH_NOTE.md` § 2) documented that **smile coefs invert sign between days** in our data:

```
day 0: a=-1.92, b=+3.97, c=-1.81
day 1: a=+4.10, b=-8.49, c=+4.64
day 2: a=+7.30, b=-14.99, c=+7.94
```

So a fixed offline-calibrated parabola would mis-direct quotes badly on at least one of the 3 days. The Hedgehogs got away with hard-coding because their smile was relatively stable across rounds. **Keep our per-tick refit.** This is one place we are correctly more sophisticated than they were.

The minor cosmetic borrow: their parameterization in `log(K/S)/sqrt(T)` is the conventional "Black total variance" axis and is more numerically stable than `S/K` near ATM. Worth trying as a one-line refactor if v3 needs it, but on its own not a PnL move.

### 2.2 EMA-of-residual fair-value correction (`mean_theo_diffs`)

**Code** (`calculate_indicators`, lines 647-655):
```python
option_theo_diff = option.wall_mid - option_theo                     # raw residual
indicators['current_theo_diffs'][option.name] = option_theo_diff
new_mean_diff = self.calculate_ema(f'{option.name}_theo_diff',
                                   THEO_NORM_WINDOW, option_theo_diff)
indicators['mean_theo_diffs'][option.name] = new_mean_diff
```
where `THEO_NORM_WINDOW = 20` (line 83) and the EMA alpha is `2/(20+1) = 0.0952`.

The trading condition (lines 682-692, simplified) is:
```
edge_at_bid  = best_bid - (option_theo + mean_theo_diff)
edge_at_ask  = (option_theo + mean_theo_diff) - best_ask

# Open short when bid is above corrected-fair by THR_OPEN+low_vega_adj
if edge_at_bid >= 0.5 (or 1.0 if vega<=1) and ask_room > 0:
    sell at best_bid (cross the bid)

# Close existing long when bid is above corrected-fair by THR_CLOSE=0
if edge_at_bid >= 0 and pos > 0:
    sell at best_bid
# (symmetric on the ask side)
```

The crucial substitution is **`option_theo + mean_theo_diff`**, not `option_theo`. The EMA of the residual absorbs whatever **systematic** mispricing exists between the parametric smile and the real market. It is a per-strike, online-updating bias correction.

**Our code** uses raw `bs_call_price(u_mid, K, T, fair_iv)` as fair, with no EMA correction. This is exactly why our voucher_research note flagged that **VEV_5400 has a structural -2.3-tick price bias** when the smile is fit on 5000-5300 — we don't subtract it out. That's why our `TAKER_STRIKES = set(VOUCHER_STRIKES_FIT)` excludes 5400: the bias would systematically trick a taker into lifting offers that aren't actually rich.

**Portable?** Yes — and this is the highest-value finding in the Hedgehogs code. The mechanism is explicitly designed to fix exactly the failure mode we observed. The +0.5 / +1.0 tick `THR_OPEN` is a mild edge requirement that shrinks adverse fills.

**Critical caveat**: Hedgehogs use it to drive an **aggressive crossing taker** (sell at the bid_wall, lift the ask_wall). Our voucher_research v6 prototype already tested a taker with `TAKER_EDGE=3` and found marginal +$250 vs MM-only at the cost of more risk. If we add the residual EMA, we should keep the taker conservative (edge=2 like `trader_combined_v1.py`, NOT the full-cross they do) and treat it as **bias-correction for our existing taker on 5400**, not as a new aggressive scalper.

**Implementation outline:**
```python
# In Trader, add per-strike traderData state:
#   "mean_theo_diff": {symbol: float}  (one EMA per voucher strike)
# Per tick, after computing fair_px from smile:
raw_diff = touch_mid - fair_px
new_mean = (2/21) * raw_diff + (19/21) * old_mean
state.traderData["mean_theo_diff"][sym] = new_mean
fair_px_corrected = fair_px + new_mean
# Use fair_px_corrected for both the maker clamp AND the taker check.
# Then enable taker on K=5400 (currently excluded as TAKER_STRIKES = FIT_STRIKES).
```

Window choice: their 20-tick window is 2 seconds of game time and is aggressive (very fast adaptation). For our setting where bot trades on 5300/5400 are 30-80/day not per-tick, a slower window like 100-300 ticks may be more stable. Sweep this.

**Expected lift**: based on voucher_research, VEV_5400 contributed +$418 over 3 days with the taker disabled. Re-enabling it with bias correction could plausibly ~double that to $800-900, so **+$120-150/day** on the lower bound, more on a good day-2.

### 2.3 EMA-of-absolute-residual gate (`switch_means` >= 0.7)

**Code** (`calculate_indicators` line 658, used in `get_iv_scalping_orders` line 672):
```python
new_mean_avg_dev = self.calculate_ema(f'{option.name}_avg_devs',
                                      IV_SCALPING_WINDOW,
                                      abs(option_theo_diff - new_mean_diff))
indicators['switch_means'][option.name] = new_mean_avg_dev
...
if self.new_switch_mean[option.name] >= IV_SCALPING_THR:    # 0.7
    # run scalping logic
else:
    # flatten any open position, do NOT post new scalp orders
```
where `IV_SCALPING_WINDOW = 100` (line 86) and `IV_SCALPING_THR = 0.7` (line 85).

This is the **mean absolute deviation** of `theo_diff` from its EMA, in price ticks. When this MAD drops below 0.7 ticks, the option's residual signal is too quiet to generate edge larger than half a tick (their `THR_OPEN`). They turn off the strategy AND flatten existing positions until volatility returns.

**Our code** has no equivalent gate. We always quote.

**Portable?** Conceptually yes, but **probably not useful as a PnL driver** in our setting because we're not actually IV-scalping — our edge is generic short-term mid mean reversion, captured by being a passive maker. Gating on residual MAD doesn't help a maker (you still want to be in the book to capture the reversion).

It does become relevant if Technique 2.2 turns on a taker on VEV_5400 — then we'd want a gate to disable the taker when residuals are too small to be a real signal. In that scenario, copy the EMA(|dev|) construct with the same 0.7-tick threshold as a guard in front of the taker.

**Implementation outline (only as a guard for the bias-corrected taker):**
```python
mad_window = 100
mad_threshold = 0.7
mean_abs_dev = ema(abs(raw_diff - new_mean), window=mad_window)
if mean_abs_dev < mad_threshold:
    skip_taker = True   # only post passive maker quotes
```

**Expected lift**: zero on its own, but it lets us run the bias-corrected taker (Technique 2.2) more confidently.

### 2.4 Low-vega adjustment (`LOW_VEGA_THR_ADJ`)

**Code** (lines 81, 678-679):
```python
LOW_VEGA_THR_ADJ = 0.5
...
low_vega_adj = 0
if self.vegas.get(option.name, 0) <= 1:
    low_vega_adj = LOW_VEGA_THR_ADJ
# ... THR_OPEN + low_vega_adj used in the open conditions
```

Vega is the option's price sensitivity to vol. When vega ≤ 1, a 1-vol-point smile-fit error translates to less than 1 price tick — i.e., the IV signal is barely visible above tick noise. They demand an extra half-tick of edge.

**Our code** doesn't compute vega and doesn't gate on it.

**Portable?** Yes, very cheap. We already have `bs_call_vega` in `trader_combined_v1.py:47`. Wire it into the taker condition.

In our P4 R3 universe, vega for an ATM 5d-TTE voucher is roughly `S * pdf(d1) * sqrt(T)` ≈ `5250 * 0.4 * sqrt(5/365)` ≈ 246. So vega is large and this adjustment basically never triggers for ATM strikes. Where it matters: deep OTM/ITM strikes (VEV_4000, 6000, 6500) where vega is near zero. We don't trade 6000/6500, and 4000 is delta-1 ITM with `bs_vega ≈ 0` — exactly where we should require more edge before any taker activity. Useful as a sanity rail if we ever add a taker on 4000/4500.

**Expected lift**: probably zero in current regime, useful if we expand strike set in v3.

### 2.5 Underlying EMA mean reversion (`ema_u`, `get_underlying_orders`)

**Code** (lines 620, 750-758):
```python
# In calculate_indicators
new_mean_price = self.calculate_ema('ema_u', underlying_mean_reversion_window=10,
                                    self.underlying.wall_mid)
indicators['ema_u_dev'] = self.underlying.wall_mid - new_mean_price
...
# In get_underlying_orders
if self.indicators['ema_o_dev'] > 15 and ask_room > 0:
    underlying.ask(bid_wall + 1, all_remaining)   # aggressive sell
elif ema_o_dev < -15 and bid_room > 0:
    underlying.bid(ask_wall - 1, all_remaining)   # aggressive buy
```

A 10-tick EMA on the underlying (VOLCANIC_ROCK). When the rock is more than 15 ticks above its short EMA, sell aggressively; when more than 15 below, buy aggressively. They explicitly chose **fixed thresholds, not vol-scaled**, "to keep the model simple and robust" (writeup line 577).

**Note**: there is a code bug in the file — line 752 reads `current_deviation = self.indicators['ema_o_dev']` (options window) but it's used in the underlying orders block. It's almost certainly a typo for `ema_u_dev`. Either way, the technique is clear.

**Our code** does not run a directional model on VELVETFRUIT_EXTRACT. We MM it passively in `trader_combined_v1.py` with the VELVET_CONFIG block.

**Portable?** Maybe — but contingent on a quick check we have not done. The Hedgehogs justified mean reversion on Volcanic Rock with autocorrelation analysis (writeup figure 8). We need the same check for VELVETFRUIT_EXTRACT before adding any directional model. If VELVETFRUIT mid returns show negative lag-1 ACF at a 10-tick window with magnitude > the 1-tick spread cost, this technique is a small additive PnL source. If not, it would lose money the same way Hedgehogs' P3-R5 mean reversion lost ~50k for them.

The asymmetry matters: Hedgehogs trade on a 1-tick aggressive edge over a 1-2 tick rock spread; we'd be doing the same on VELVETFRUIT which has a similar 1-tick spread. Tight margin.

**Implementation outline (only after ACF check):**
```python
# In TraderState carry { "ema_u": float }
# Per tick: ema_u = (2/(W+1))*u_mid + (1 - 2/(W+1))*ema_u_prev
# If u_mid > ema_u + thr: sell VELVETFRUIT at best_bid+1, sized by inventory room.
# If u_mid < ema_u - thr: buy VELVETFRUIT at best_ask-1.
# Apply ONLY when our passive MM has spare room (do NOT crowd the maker).
```

**Expected lift**: highly conditional. If VELVETFRUIT mean-reverts at lag 10 with `|acf| > 0.15`, plausibly +$200-400/day. If not, will lose. Run the check first.

### 2.6 ITM mean-reversion combined signal (`get_mr_orders`)

**Code** (lines 706-727):
```python
for option in options:    # only the 9500 strike (deepest ITM)
    current_deviation = self.indicators['ema_o_dev']    # underlying long-EMA dev
    iv_deviation = current_theo_diffs[name] - mean_theo_diffs[name]
    current_deviation += iv_deviation

    if current_deviation > 5 and ask_room > 0:
        ask(best_bid, max_allowed_sell)
    elif current_deviation < -5 and bid_room > 0:
        bid(best_ask, max_allowed_buy)
```

A second underlying-EMA (window 30, `ema_o`) added to the IV residual gives a single signal that says "underlying is rich AND/OR voucher is rich". Threshold 5. They run this **only on the 9500 strike** — the deepest ITM, near delta 1 — using it as a directional underlying play with leverage.

**Our code** has no equivalent. We do quote VEV_4000 (deepest ITM) passively but with no directional overlay.

**Portable?** Conditionally yes. VEV_4000 in our market has ~21-tick spread (per voucher_research) and `bs_vega ≈ 0` — IV residuals there are pure noise per our existing analysis. So the IV part of the signal contributes nothing. The pure underlying EMA dev would have to do all the work, which collapses this back into Technique 2.5.

The only reason to keep the ITM strike rather than VELVETFRUIT for the directional play is **leverage and position-limit headroom**: VEV_4000's limit is 300 vs VELVETFRUIT's 200, and VEV_4000 is delta ≈ 1 so 300 contracts is ~equivalent to 300 underlying. Net: same delta, higher cap. But ITM voucher has a 21-tick spread vs VELVETFRUIT's ~5-tick spread, so per-trade cost is 4x higher.

**Math: don't bother for VEV_4000.** The wider spread eats any mean-reversion edge. If you do an underlying mean reversion overlay, do it on VELVETFRUIT directly (Technique 2.5).

### 2.7 Wall-mid price fallback for one-sided books

**Code** (lines 631-639):
```python
if option.wall_mid is None:
    if option.ask_wall is not None:
        option.wall_mid = option.ask_wall - 0.5
        option.bid_wall = option.ask_wall - 1
        option.best_bid = option.ask_wall - 1
    elif option.bid_wall is not None:
        option.wall_mid = option.bid_wall + 0.5
        option.ask_wall = option.bid_wall + 1
        option.best_ask = option.bid_wall + 1
```

When one side of the book is empty, fabricate a synthetic mid 0.5 below ask or 0.5 above bid. This keeps the strike in the smile fit and IV computation rather than dropping it.

**Our code** drops the strike entirely if either side is missing (`_fit_smile` line 245-246: `if depth is None or not depth.buy_orders or not depth.sell_orders: continue`).

**Portable?** Yes, free. In our setting one-sided books happen on VEV_5500, 6000, 6500 and would let those still feed into a smile fit. But: our research already disqualified those strikes for trading and the smile fit on inner 4 is by design extrapolation-light, so adding noisy outer points hurts more than it helps. **Skip.**

### 2.8 Warmup gate

**Code** (line 732):
```python
if self.state.timestamp / 100 < min([THEO_NORM_WINDOW, ...]): return {}
```

Skip all option orders for the first `min(window) * 100` timestamps so the EMAs have data. With windows 20/10/30, that's 1000-3000 timestamps (= 1-3 seconds of game time).

**Our code** doesn't need this because we don't use EMAs. If we adopt Technique 2.2 (residual EMA correction), we MUST add a warmup gate or the first ticks will use a stale `mean_theo_diff = 0` that misleads the corrected fair value. Trivial to add.

### 2.9 IV solver: closed-form polynomial vs Newton-Raphson

**Code** (line 587): `iv = np.poly1d(coeffs)(m_t_k)` — no solver, just plug in moneyness.

**Our code** runs Newton-Raphson per tick per strike (`implied_vol_call`).

**Portable?** Direction reversed: their approach is faster but useless to us because their polynomial coefficients aren't ours, and our voucher_research showed our parabola coefs flip sign daily. Keep the Newton solver. Our solver already has all the fail modes covered (return NaN on bad inputs / non-convergence / out-of-bounds sigma).

### 2.10 Position sizing — "fire all at once"

**Code** (lines 683, 689 etc.): `option.ask(option.best_bid, option.max_allowed_sell_volume)` — when the signal fires, post the **full remaining position room** at the touch. No fractional sizing, no QUOTE_SIZE concept. They size the bet as large as the position limit allows.

**Our code** (combined v1, line 130): `VOUCHER_QUOTE_SIZE = 30` per side, with linear shrinkage above |pos|=80.

**Portable?** No. Their full-firing-on-edge style works because they have a real IV-residual signal with positive expected per-trade PnL (they backtested ~100k per round). We've shown that on our data the per-tick residual signal is **positively autocorrelated** (deviation persists, doesn't revert) — firing the full limit on a "rich" reading means you get caught on the wrong side as the deviation grows. Our 30-lot quote with inventory-skew is the right design for our regime.

### 2.11 Informed-trader following (Olivia)

**Code** in the `ProductTrader` base `check_for_informed` (line 225) and the wiring in `DynamicTrader`/`InkTrader`/`EtfTrader`.

**Crucially the `OptionTrader` class does NOT call `check_for_informed`.** They never look for Olivia in the voucher books. So no informed-trader signal applies to options in their P3 code. Closed.

**Our equivalent**: P4 has its own informed traders (`Penelope`, `Camilla`, etc. — listed in the wiki). We don't currently use them on vouchers, and the Hedgehogs precedent suggests we shouldn't either.

---

## 3. What we already have

Confirming redundancies — these techniques in their code are present in some form in `trader_combined_v1.py`:

- **Per-tick smile fit, BS pricing, fair-clamp on maker quotes** — we have it (and ours is more flexible: we refit instead of using fixed coefs).
- **Inventory skew on quote sizing** — we have it (`_voucher_sized_quotes` linear shrink above |pos|=80).
- **Position limit guardrails** — we have it (`VOUCHER_MAX_POSITION = 250` of the 300 limit).
- **Multi-strike strategy with different logic per strike** — we have it (4000 ITM gets `ITM_CONFIG` delta-1 MM; 5000-5400 get smile MM).
- **Cautious taker on safe strikes** — we have it (`TAKER_EDGE = 2`, `TAKER_STRIKES = FIT_STRIKES`).
- **End-of-day flatten** — we have it (`FLATTEN_WINDOW_START_TS = 980_000`); they don't.
- **Skip when smile undefined** — we have it (`fair_px is None` fallback to plain MM).
- **Skip when one side of the book missing** — we have it (their fabrication logic is more aggressive but worse for our outer strikes).

---

## 4. What is NOT portable and why

Be explicit so we don't relitigate this:

**a) Hard-coded smile coefficients** — our smile parameters invert across days; theirs were stable. Verified in `voucher_research/RESEARCH_NOTE.md` § 2.

**b) Aggressive crossing IV-scalping (lift the entire ask wall on a 0.5-tick edge)** — our IV deviations have **positive** lag-1 autocorrelation across all relevant strikes (per `dev_acf.csv`: 0.10 to 0.61). The Hedgehogs setup had NEGATIVE autocorrelation in IV deviations, which is why scalping worked for them. Crossing the spread on a "rich" reading on our data systematically catches the wrong side of a continuing trend. **Do not implement their `get_iv_scalping_orders` directly.**

**c) Underlying mean reversion on VOLCANIC_ROCK with thresholds 15 / window 10** — these are calibrated to a different underlying with very different volatility. VELVETFRUIT_EXTRACT may or may not mean-revert; needs an ACF check before any directional overlay.

**d) ITM 9500-strike directional flow** — wider voucher spread + zero IV vega contribution makes this a worse bet than just trading the underlying directly. Do it on VELVETFRUIT or skip.

**e) Aggressive delta hedging** — they didn't do it explicitly and their writeup says it would be too costly. Our voucher_research v7 prototype confirmed (-$2572 on VELVETFRUIT hedger). Don't add a hedger.

**f) Fire-the-whole-limit sizing** — only works with a high-confidence signal. Our regime doesn't have one.

**g) Olivia-style informed follow** — their option code doesn't use it. Their delta-1 traders do but those are different products with different bot structures.

---

## 5. Recommended next step

**Build `trader_combined_v2.py`** from `trader_combined_v1.py` adding only **Technique 2.2 (residual EMA fair-value correction)**. Concretely:

1. Add `mean_theo_diff: dict[str, float]` to `traderData`.
2. After computing `fair_px` in `_quote_smile_voucher`, compute `raw_diff = touch_mid - fair_px`, EMA-update `mean_theo_diff[sym]` with window 100 (slower than their 20 to suit our trade frequency; sweep 50/100/200 in backtest).
3. Define `fair_px_corrected = fair_px + mean_theo_diff[sym]` and use it in **both** the maker clamp (`min(buy_price, int(fair_px_corrected))` etc.) and the taker check (`ba <= fair_px_corrected - TAKER_EDGE` etc.).
4. With the corrected fair, **expand `TAKER_STRIKES = set(VOUCHER_STRIKES_SMILE)`** (add 5400) so the 5400 strike's taker leg can fire safely.
5. Add a warmup guard: skip taker (but not maker) when `state.timestamp < WARMUP_TICKS` (e.g., 100*100 = 10k).

Backtest comparison metrics: total PnL across 3 days, per-day PnL, max DD, per-strike contribution. Decision rule from the user's stated criteria: keep v2 only if **per-day variance does not increase** while total PnL improves.

If v2 doesn't beat v1 on combined PnL+stability, **don't ship it** — the existing combined v1 is profitable and adding moving parts has real downside risk. Time spent on the residual-EMA correction is bounded: ~1-2 hours to implement, 30 min to backtest. If it doesn't pass, freeze v1.

**Skip techniques 2.3 (MAD gate), 2.5 (underlying MR), and 2.6 (ITM MR) for now** unless v2 demonstrates that the residual-correction story works AND the user has bandwidth.

---

## 6. Files added

- `Round3/Analysis/output/hedgehogs_r3_review/HEDGEHOGS_R3_REVIEW.md` — this file.

No code changes. Implementation deferred until the user explicitly wants v2.
