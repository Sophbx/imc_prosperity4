# Round 3 Analysis — Overview

Two notebooks looking at Round 3 data. Split by asset class because options
need totally different analysis (implied vol, smile, etc.) than the regular
delta-1 products.

## Files

```
Round3/Analysis/
├── helpers.py              shared utilities (data loading, BS, IV, book/trade features)
├── test_helpers.py         pytest tests for helpers (30 passing)
├── analysis_delta1.ipynb   HYDROGEL_PACK + VELVETFRUIT_EXTRACT
├── analysis_options.ipynb  10 VEV vouchers + their underlying
├── build_*.py              scripts that regenerate the notebooks (don't hand-edit ipynbs)
└── output/                 CSVs the strategy code can consume
```

## What's in each notebook

**`analysis_delta1.ipynb`** — for the two non-option products:
- mid-price trajectories, drift, returns, autocorrelation, rolling vol
- per-product spread / fair-value estimators / predictor ranking against future price change
- VELVETFRUIT realized volatility (this feeds the options view)

**`analysis_options.ipynb`** — for the 10 VEV vouchers:
- price trajectories per strike + spread/quoted-size summary
- voucher vs underlying scatter (slope = empirical delta)
- implied volatility per row, IV time series, smile snapshots, rich/cheap rank

## Output CSVs

| File | What's in it |
|---|---|
| `delta1_feature_summary.csv` | per-product spread/width/drift summary |
| `delta1_predictor_ranking.csv` | per-feature correlation with forward change/direction |
| `velvetfruit_realized_vol.csv` | tick-level rolling vol of VELVETFRUIT |
| `iv_timeseries.csv` | IV per (day, ts, strike) — 30k rows |
| `voucher_price_summary.csv` | per-strike: spread, size, mean IV, delta, moneyness |
| `voucher_underlying_corr.csv` | per-strike: OLS slope, R², rolling correlation stats |

## Running it

```bash
cd Round3/Analysis
python3 -m pytest test_helpers.py    # sanity check
jupyter lab analysis_delta1.ipynb    # or analysis_options.ipynb
```

To regenerate a notebook from scratch (e.g., after editing the build script):

```bash
python3 build_delta1_notebook.py
jupyter nbconvert --to notebook --execute --inplace analysis_delta1.ipynb
```

## Conventions

- `r = 0` for Black-Scholes (Prosperity has no carry).
- TTE annualized by 365 calendar days.
- TTE schedule: 8d at start of day 0, 7d at day 1, 6d at day 2, 5d live in Round 3.
- Timestamps run 0..999_900 step 100 (10k ticks per day).

## Quick findings from the data

- Smile is steep, not flat — IV ranges 0.09 deep ITM → 0.60 deep OTM.
- VEV_4000 empirical delta ≈ 1.0 (deep ITM, basically the underlying).
- VEV_6000 / VEV_6500 have constant mids — IV is noisy/undefined for those.
- IV is finite for ~92% of rows.
