# Round 3 Analysis Overview

Two notebooks looking at Round 3 data. Split by asset class because options
need totally different analysis compared to the regular
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

**`analysis_delta1.ipynb`** :
- mid-price trajectories, drift, returns, autocorrelation, rolling vol
- per-product spread / fair-value estimators / predictor ranking against future price change
- VELVETFRUIT realized volatility (this feeds the options view)

**`analysis_options.ipynb`** :
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
