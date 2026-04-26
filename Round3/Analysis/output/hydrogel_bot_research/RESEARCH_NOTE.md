# HYDROGEL_PACK informed-bot hunt — research note

## Headline answer

**No exploitable informed-bot pattern.** Trades that print near rolling extremes do precede a measurable forward reversion (t-stat 5–6 for fwd-500, mean PnL ~$15–20 per long signal at HYDROGEL_PACK), but the same reversion is **stronger and cleaner from extreme MID moments where no trade occurred at all** (fwd-100 t=+11.97, mean=+7.69). Once we condition on the rolling-mid extreme, the trade event itself adds essentially zero incremental signal (delta R² ≈ 0.0001 for HYDROGEL, 0.0002 for VELVETFRUIT). The "near-extreme print" effect is just mean-reversion of the underlying mid, not a bot fingerprint we can identify by trade-tape characteristics.

That doesn't mean the underlying mean-reversion is unprofitable — it's worth the existing v2 hydrogel MM trader's time — it just means there is **no Olivia analogue** in HYDROGEL trades to exploit beyond what the order book itself already tells us.

## Method

For each HYDROGEL_PACK and (as benchmark) VELVETFRUIT_EXTRACT trade across days 0–2 (1,010 + 1,372 trades respectively):

1. Built rolling 200/500/1000-tick min/max of the touch-mid per (product, day).
2. Flagged trades within delta = 0,1,2 of the rolling extreme as "near-min" or "near-max".
3. Computed forward mid changes at horizons 50/100/500 ticks (1 tick = 100 timestamp units).
4. Stratified by quantity quartile, trade location class (`at_or_above_ask_touch` etc.), trade aggression direction (price vs touch_mid), and timestamp modular bucket.
5. **Placebo**: ran the same forward-return analysis on book rows where the mid hit the rolling extreme but **no trade occurred** in that timestamp.
6. Counterfactual fade-the-extreme PnL: long 1 lot at trade price when "near-min", short 1 lot when "near-max", exit at fwd_mid_h.
7. Multivariate OLS: `fwd_ret_100 ~ dist_signed_to_extreme + had_trade + trade_qty` to measure incremental lift from trade features.

## Key statistics

### A. Near-extreme trades do show forward reversion (looks promising at first)

HYDROGEL_PACK, w=500 rolling window, delta=1:

| bucket   | n  | mean fwd_50 | t fwd_50 | mean fwd_100 | t fwd_100 | mean fwd_500 | t fwd_500 |
|----------|----|-------------|----------|--------------|-----------|--------------|-----------|
| near_min | 76 | +4.73       | +2.87    | +8.98        | +3.99     | +14.23       | +4.28     |
| near_max | 97 | -1.43       | -1.03    | -3.89        | -2.14     | -8.97        | -3.00     |

Three-day cumulative fade-the-extreme PnL (long-near-min + short-near-max, exit at fwd_500): **$3,150** for HYDROGEL_PACK, **$1,447** for VELVETFRUIT_EXTRACT.

### B. But all "near-min" trades are aggressive sells (price < mid), and all "near-max" trades are aggressive buys

| product             | bucket   | n  | frac_below_mid | frac_above_mid | mean_price_minus_mid |
|---------------------|----------|----|----------------|----------------|----------------------|
| HYDROGEL_PACK       | near_min | 76 | 1.000          | 0.000          | -7.91                |
| HYDROGEL_PACK       | near_max | 97 | 0.000          | 1.000          | +7.90                |
| VELVETFRUIT_EXTRACT | near_min | 93 | 0.774          | 0.226          | -1.74                |
| VELVETFRUIT_EXTRACT | near_max | 88 | 0.000          | 1.000          | +2.46                |

So "near-min" is mechanically defined to coincide with a sell-aggressed print — these are the same set of trades. The "informed counterparty" framing inverts: the *seller* (counterparty) just sold at a local low and we'd want to fade them. But the question is whether observing the *trade* gives us information beyond what we already see in the book.

### C. Placebo kills the bot hypothesis

| product       | direction          | horizon | n    | mean fwd_ret | t      |
|---------------|--------------------|---------|------|--------------|--------|
| HYDROGEL_PACK | near_min, **no trade** | 100 | 928  | **+7.69**    | **+11.97** |
| HYDROGEL_PACK | near_min, with trade | 100 | 33   | +1.11        | +0.30  |
| HYDROGEL_PACK | near_max, **no trade** | 100 | 1047 | **-5.03**    | **-10.52** |
| HYDROGEL_PACK | near_max, with trade | 100 | 42   | -4.14        | -1.71  |

The reversion is **stronger** when no trade occurred. This is the opposite of what an "informed bot" model predicts — if a bot were buying at the local minimum, we would expect *with-trade* extremes to forward-revert *more* than no-trade extremes. They forward-revert *less*, and only marginally.

### D. Trade aggression direction at extremes shows no differential signal

| product       | context                  | horizon | n  | mean_ret | t     |
|---------------|--------------------------|---------|----|----------|-------|
| HYDROGEL_PACK | near_min_sell_aggressed  | 500     | 13 | +12.12   | +1.80 |
| HYDROGEL_PACK | near_min_buy_aggressed   | 500     | 20 | +12.33   | +1.85 |
| HYDROGEL_PACK | near_max_sell_aggressed  | 500     | 21 | -8.38    | -1.75 |
| HYDROGEL_PACK | near_max_buy_aggressed   | 500     | 21 | -9.60    | -1.58 |

Whether the print is buy-aggressed or sell-aggressed at the extreme, the forward return is roughly equal (and the t-stats are weak). No "informed buy at the bottom" effect.

### E. Quantity / timing fingerprints

- HYDROGEL quantity range: 2–6, mean 4.04. Distribution at near-extremes is broadly the same shape as overall — no anomalous quantity signature (e.g., no recurring "always 2 lots" pattern that would mark a distinct trader).
- All HYDROGEL trades occur at timestamps that are multiples of 100 (the data tick). 21% land on multiples of 500, 11% on multiples of 1000 — exactly what you'd expect from uniform spacing. No mod-cadence fingerprint.
- High-qty vs low-qty splits at the extreme: t-stats are similar / noisier for high-qty. Quantity is not a reliable filter.

### F. Multivariate OLS, fwd_ret_100 ~ dist_signed + had_trade + trade_qty (whole 30k-row book panel)

| product             | n      | R² (dist only) | R² (+ trade features) | Delta R² | t(had_trade) | t(trade_qty) |
|---------------------|--------|----------------|------------------------|----------|--------------|--------------|
| HYDROGEL_PACK       | 29,673 | 0.04040        | 0.04047                | +0.000067 | -1.06        | +1.33        |
| VELVETFRUIT_EXTRACT | 29,673 | 0.02280        | 0.02298                | +0.000183 | -1.86        | +2.26        |

The `dist_signed_to_extreme` slope (t = +35) absorbs essentially all of the predictive content. Adding `had_trade` and `trade_qty` lifts R² by 67 ppm on HYDROGEL — irrelevant.

### G. Per-day stability of the fade-the-extreme strategy (w=500, horizon=500)

| product       | day | direction | n  | pnl_total | pnl_mean | t     |
|---------------|-----|-----------|----|-----------|----------|-------|
| HYDROGEL_PACK | 0   | near_min  | 22 | $635      | +28.86   | +6.00 |
| HYDROGEL_PACK | 1   | near_min  | 24 | $556      | +23.15   | +4.41 |
| HYDROGEL_PACK | 2   | near_min  | 27 | $426      | +15.76   | +2.40 |
| HYDROGEL_PACK | 0   | near_max  | 35 | $706      | +20.16   | -4.31 |
| HYDROGEL_PACK | 1   | near_max  | 38 | $529      | +13.91   | -2.84 |
| HYDROGEL_PACK | 2   | near_max  | 18 | $300      | +16.67   | -2.54 |

Day-by-day signs are consistent. But again, this is a re-statement of mean-reversion, not a trade-specific edge.

## Counterfactual PnL

If we did fade the trade-extreme without considering positions / inventory / overlap with our existing market-maker, the headline 3-day PnL is **$3,150 for HYDROGEL** and **$1,447 for VELVETFRUIT** (w=500, horizon=500). For comparison, a passive 25%-fill MM on HYDROGEL is estimated at **$1,985** for 3 days (using avg half-spread ~7.86), and the existing v2/v2b hydrogel trader earns much more from continuous quoting. The fade-the-extreme strategy:

- Triggers only ~50 times per day on HYDROGEL.
- Carries 500-tick mid-mark risk.
- Will be drowned by the same MM PnL it competes for if both run on the same book.

More importantly, the placebo result shows that the trade is **not** the informative event. The book-state signal (mid touching its rolling extreme) is. We can already use that signal in any market-making logic via the touch_mid / rolling-min features without watching the trade tape at all.

## Recommendation

**Do not build an "informed-bot" v3 hydrogel trader.** The trade tape contains no fingerprint that distinguishes a smart from a noise counterparty:

- No quantity anomaly,
- No timing modular pattern,
- No trade-location pattern beyond the mechanically tautological "near-min ⇒ at_or_below_bid_touch",
- No incremental forward-return predictability beyond what the rolling mid distance gives us.

What is real and exploitable is the **mean-reversion of the HYDROGEL_PACK mid itself**: when the touch_mid prints a rolling-500 extreme, the next 100 ticks reverse by ~$5–8. That signal is already cleanly observable from the order book; if v2/v2b doesn't yet weight quoting against rolling-extreme distance, that's a small but worthwhile tweak. But it's a feature for the existing MM logic, not a separate informed-bot trader.

Same conclusion holds for VELVETFRUIT_EXTRACT — weaker reversion (mean-revert ~$2 over 100 ticks), same lack of trade-tape fingerprint.

If a future round publishes counterparty IDs, this analysis should be re-run with the buyer/seller tags. With anonymous trade tape, the answer is firmly **no**.

## Files added

Under `Round3/Analysis/output/hydrogel_bot_research/`:

- `RESEARCH_NOTE.md` — this note.
- `trades_enriched.csv` — every HYDROGEL/VELVETFRUIT trade with rolling-extreme flags + forward returns at 50/100/500.
- `bucket_forward_returns.csv` — mean/t fwd return by (product, near_min/near_max, w, delta).
- `exploit_pnl.csv` and `exploit_fade_pnl.csv` — fade-the-extreme PnL, multiple windows / horizons.
- `per_day_signal_stability.csv` — same PnL split by day (stability check).
- `placebo_no_trade_extremes.csv` — **the decisive table**: extreme-mid moments with vs without a trade, fwd-return.
- `conditional_reversion.csv` — fwd-return conditional on (near-extreme, had_trade) — same data, cleaner cut.
- `qty_conditional.csv` — fwd-return at extremes split by trade quantity bucket.
- `trade_aggression_at_extremes.csv` — fwd-return at extremes split by trade aggression direction.
- `trade_location_breakdown.csv` — counts and fwd-returns by `trade_location` class.
- `near_extreme_signed.csv` — confirms that "near-min" trades are 100% sell-aggressed, "near-max" 100% buy-aggressed.
- `near_extreme_x_location.csv` — same cross-tabulation.
- `quantity_distributions.csv` — qty histogram overall vs in near-extreme buckets.
- `timing_modular.csv` — timestamp mod cadence check.
- `passive_mm_baseline.csv` — half-spread × n_trades baseline for context.
- `signal_frame.csv` — slim per-trade frame (price, mid, fwd returns, near-extreme flags) for any downstream prototyping.
