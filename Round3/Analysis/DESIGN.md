# Round 3 Analysis — Design

Date: 2026-04-24
Status: approved (2026-04-24), implementation pending

## Goal

Build two Jupyter notebooks that turn the three days of Round 3 historical data
(`Round3/Data/`) into an actionable understanding of the 13 products we will
trade: 2 delta-1 commodities (`HYDROGEL_PACK`, `VELVETFRUIT_EXTRACT`) and 10
call-option vouchers on `VELVETFRUIT_EXTRACT` (`VEV_4000`, `VEV_4500`, `VEV_5000`,
`VEV_5100`, `VEV_5200`, `VEV_5300`, `VEV_5400`, `VEV_5500`, `VEV_6000`,
`VEV_6500`).

The notebooks are research tools, not strategy code. They should:

1. Let a reader understand each product's behaviour by reading top-to-bottom.
2. Produce a small curated set of CSV artifacts that later strategy code can
   load without re-running the notebooks.

Delta-1 = price moves 1-for-1 with itself (no optionality). Option delta is the
rate of change of option price per unit change of the underlying.

## Scope decisions (agreed during brainstorming)

| Decision | Choice |
|---|---|
| Structure | Two notebooks split by asset class |
| Options depth | First-look + implied-volatility core (no greeks timeseries, no IV-vs-RV rich/cheap, no cross-strike arbitrage checks) |
| Delta-1 depth | Focused pass (~22 cells); skips detrending, trade-impact alignment, OHLC candles, break-vs-bounce probes, size-wall tercile matrix |
| Outputs | Inline plots + curated CSV exports under `Analysis/output/` |
| Helper sharing | Hybrid: `helpers.py` for expensive/reusable code (book features, Black-Scholes); inline for trivial glue |

## File layout

```
imc_prosperity4/Round3/
├── Data/                              (already present)
│   ├── prices_round_3_day_{0,1,2}.csv
│   ├── trades_round_3_day_{0,1,2}.csv
│   └── La_trahison_des_images.png
└── Analysis/                          (new)
    ├── DESIGN.md                      this file
    ├── helpers.py                     shared helpers
    ├── analysis_delta1.ipynb          HYDROGEL_PACK + VELVETFRUIT_EXTRACT
    ├── analysis_options.ipynb         10 VEV vouchers + underlying
    └── output/
        ├── delta1_feature_summary.csv
        ├── delta1_predictor_ranking.csv
        ├── velvetfruit_realized_vol.csv
        ├── iv_timeseries.csv
        ├── voucher_price_summary.csv
        └── voucher_underlying_corr.csv
```

## `helpers.py`

```python
# ---------- data loading ----------
extract_day_from_filename(path)              -> int
load_prices(data_dir, price_files)           -> DataFrame
load_trades(data_dir, trade_files)           -> DataFrame

# ---------- tiny math utilities ----------
safe_corr(x, y)                              -> float
weighted_avg(values, weights)                -> float

# ---------- book feature engineering (ported from Round 1) ----------
build_book_features(prices, visible_levels=3) -> DataFrame
#   touch_mid, boundary_mid, size_wall_mid, side_vwap_mid, full_book_vwap_center,
#   touch_spread, boundary_width, size_wall_width,
#   frontier_imbalance, depth_imbalance, size_wall_vol_imbalance,
#   fwd_touch_mid_change_{HORIZON}, fwd_direction_{HORIZON}, next_trade_sign_proxy
build_trade_features(trades, book_df)        -> DataFrame

# ---------- Black–Scholes (new in Round 3) ----------
bs_call_price(S, K, T, sigma, r=0.0)         -> float
bs_call_delta(S, K, T, sigma, r=0.0)         -> float
bs_call_vega (S, K, T, sigma, r=0.0)         -> float
bs_call_gamma(S, K, T, sigma, r=0.0)         -> float
implied_vol_call(price, S, K, T, r=0.0,
                 lo=1e-4, hi=5.0, tol=1e-6)  -> float   # NaN outside bounds

# ---------- convention constants ----------
TIMESTAMPS_PER_DAY = 10_000
YEAR_DAYS = 365
TTE_DAYS_AT_DAY = {0: 8, 1: 7, 2: 6}          # per wiki
tte_years(day, timestamp)                    -> float
```

Conventions:

- **r = 0** — standard for Prosperity; no cash-carry.
- **Annualize by 365 calendar days** — arbitrary scale; IV time-series shape is
  the only thing strategy signals care about.

## `analysis_delta1.ipynb` (~22 cells)

```
A. Setup & load
   A.1 (md) title + what this notebook answers
   A.2 (py) imports, plot theme, from helpers import ...
   A.3 (py) config: DATA_DIR, PRICE_FILES, TRADE_FILES, VISIBLE_LEVELS=3,
          HORIZON=10, OUTPUT_DIR
   A.4 (py) load prices + trades, concat days, call build_book_features and
          build_trade_features
   A.5 (md) data-quality summary
   A.6 (py) rows per product × day, NaN counts, mid_price==0 rows; drop bad
          rows into df_clean

B. Cross-product overview
   B.1 (md) section intro
   B.2 (py) 3-day mid-price trajectory per product, colour-coded by day
   B.3 (py) per-day OLS drift slope table (mid ~ timestamp)
   B.4 (py) tick-return distribution + lag-1..20 autocorrelation + rolling vol
          per product
   B.5 (py) cross-product tick-return correlation per day (flags VELVETFRUIT
          ↔ HYDROGEL coupling)

C. Per-product drill-down (loop over both products)
   C.1 (md) section intro; PRODUCT is loop-driven, not fixed
   C.2 (py) static geometry — touch_spread & size_wall_width mean/median +
          depth histogram
   C.3 (py) five fair-value estimators — correlation matrix + dislocation
          time series
   C.4 (py) predictors for fwd_touch_mid_change_{HORIZON}: rank 14 features
          by |corr|
   C.5 (py) predictors for next_trade_sign_proxy: rank features by |corr|
   C.6 (py) trade-location breakdown (at touch / at size wall / inside /
          outside)

D. VELVETFRUIT-specific
   D.1 (md) underlying-of-vouchers note, pointer to options notebook
   D.2 (py) rolling realized vol of VELVETFRUIT_EXTRACT mid-returns in
          multiple windows; save to output/velvetfruit_realized_vol.csv

E. Exports
   E.1 (py) save delta1_feature_summary.csv, delta1_predictor_ranking.csv,
          velvetfruit_realized_vol.csv
   E.2 (md) "Next: analysis_options.ipynb"
```

## `analysis_options.ipynb` (~25 cells)

```
A. Setup & load
   A.1 (md) title + what this notebook answers
   A.2 (py) imports, theme, from helpers import ...
   A.3 (py) config: DATA_DIR, PRICE_FILES, TRADE_FILES,
          STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500],
          UNDERLYING = "VELVETFRUIT_EXTRACT",
          TTE_DAYS_AT_DAY = {0: 8, 1: 7, 2: 6}, OUTPUT_DIR
   A.4 (py) load prices + trades; filter to VEV_* and VELVETFRUIT_EXTRACT;
          attach strike and tte_years per row
   A.5 (md) data-quality summary
   A.6 (py) rows per strike × day, NaN counts, zero-mid / zero-quote sanity,
          quoted-size distribution

B. Price behaviour of underlying + vouchers
   B.1 (md) section intro
   B.2 (py) VELVETFRUIT_EXTRACT mid trajectory (repeated from delta-1 nb for
          self-containment)
   B.3 (py) mid trajectory for each of 10 vouchers (3×4 grid, shared x-axis)
   B.4 (py) per-voucher spread + quoted size summary table

C. Voucher ↔ underlying relationship
   C.1 (md) section intro — option slope ≈ delta
   C.2 (py) scatter voucher_mid vs underlying_mid per strike (3×4 grid); OLS
          slope = empirical delta; annotate moneyness S/K
   C.3 (py) rolling-window correlation (voucher-return vs underlying-return)
          per strike
   C.4 (py) intrinsic-value floor check: voucher_mid >= max(0, S - K)?
          flag violations

D. Implied volatility core
   D.1 (md) section intro — IV, per strike, smile
   D.2 (py) compute IV via helpers.implied_vol_call, downsampled to every
          10th row to keep runtime ~30k solves rather than 300k; attach as iv
   D.3 (py) IV time series per strike (overlay, colour by strike)
   D.4 (py) volatility smile at 3 snapshots (start of day 0, end of day 1,
          end of day 2); IV vs strike; annotate ATM strike
   D.5 (py) IV stability: distribution + mean ± std per strike
   D.6 (py) "rich / cheap" view — rank strikes by (IV − cross-strike median IV);
          flag persistent outliers

E. Summary tables
   E.1 (py) save iv_timeseries.csv (long: day, timestamp, strike, mid, iv)
   E.2 (py) save voucher_price_summary.csv
   E.3 (py) save voucher_underlying_corr.csv
   E.4 (md) free-form findings summary — liquidity, smile regime, suspicious
          strikes, strategy recommendations
```

## Artifact exports

All written to `imc_prosperity4/Round3/Analysis/output/`.

| File | Source notebook | Schema | Row count (rough) |
|---|---|---|---|
| `delta1_feature_summary.csv` | delta1 | product, spread/size-wall/dislocation stats, per-day drift | 6 |
| `delta1_predictor_ranking.csv` | delta1 | product, feature, corr_with_fwd_change, corr_with_fwd_direction, corr_with_next_trade_sign | ~60 |
| `velvetfruit_realized_vol.csv` | delta1 | day, timestamp, rv_100, rv_500, rv_2000 | ~30k |
| `iv_timeseries.csv` | options | day, timestamp, strike, voucher_mid, underlying_mid, moneyness, tte_years, iv | ~30k |
| `voucher_price_summary.csv` | options | strike, n_rows, mean_spread, mean_quoted_size, mean_iv, iv_std, empirical_delta, mean_moneyness | 10 |
| `voucher_underlying_corr.csv` | options | strike, ols_slope, r_squared, rolling_corr_mean, rolling_corr_std | 10 |

## Verification plan

1. Execute both notebooks end-to-end from a fresh kernel via
   `jupyter nbconvert --to notebook --execute` — must succeed with no errors.
2. Confirm `output/` contains all 6 CSVs with the expected schema and non-trivial
   row counts.
3. Sanity checks:
   - Deep-ITM voucher (strike 4000) empirical delta ≈ 1 in the scatter.
   - Smile plot is not flat — some curvature expected.
   - IV is finite for ≥ 80% of downsampled rows.
4. Spot-check one `safe_corr` value in the delta-1 notebook vs a manual
   `np.corrcoef` call to confirm `helpers.py` math is correct.

## Explicit non-goals

- No strategy code.
- No greeks time series beyond the empirical delta in section C.2 of options.
- No cross-strike no-arbitrage checks (butterfly / call-spread).
- No vol term structure (only one expiry).
- No IV-vs-realized-vol rich/cheap scoring beyond section D.6's cross-strike
  relative view.
- No detrending layer, no trade-impact alignment, no OHLC candles, no break-
  vs-bounce probes, no size-wall tercile matrix in delta-1 (these were Round 1
  layers; marginal value doesn't justify cells here).
