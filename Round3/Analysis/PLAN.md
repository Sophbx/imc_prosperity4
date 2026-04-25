# Round 3 Analysis Notebooks — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `helpers.py` + `analysis_delta1.ipynb` + `analysis_options.ipynb` per `Round3/Analysis/DESIGN.md`, producing 6 CSV artifacts under `Round3/Analysis/output/`.

**Architecture:** Hybrid helper sharing — one `helpers.py` module holds the expensive/reusable functions (data loading, book features, Black–Scholes, implied vol), and each notebook imports from it. Notebooks are created programmatically via `nbformat` from Python build scripts so cell content is diff-friendly.

**Tech Stack:** Python 3 · pandas · numpy · scipy.optimize (Brent) · matplotlib · seaborn · nbformat · jupyter nbconvert · pytest

---

## Conventions for this plan

- **Working directory for all commands:** `imc_prosperity4/Round3/Analysis/` unless a command explicitly uses `cd` elsewhere.
- **Python interpreter:** `python3` (system Python has pandas/numpy/scipy/matplotlib/seaborn installed; `python3 -m pytest` works; `jupyter` CLI is on PATH).
- **Tests live beside `helpers.py`** — file is `test_helpers.py` in the same directory, discovered by pytest with no `__init__.py` or sys.path tweaks needed.
- **Commits go on the current branch (`main`).** Never push in the middle of the plan; only push after Task 14 if the user asks.

---

## Task 1: Scaffold `helpers.py` + math utilities (`safe_corr`, `weighted_avg`, `extract_day_from_filename`)

**Files:**
- Create: `imc_prosperity4/Round3/Analysis/helpers.py`
- Create: `imc_prosperity4/Round3/Analysis/test_helpers.py`

- [ ] **Step 1: Write the failing test file**

Write `Round3/Analysis/test_helpers.py`:

```python
import math
import numpy as np
import pandas as pd
import pytest

import helpers


def test_extract_day_from_filename_positive():
    assert helpers.extract_day_from_filename("prices_round_3_day_2.csv") == 2


def test_extract_day_from_filename_negative():
    assert helpers.extract_day_from_filename("prices_round_1_day_-2.csv") == -2


def test_extract_day_from_filename_no_day_defaults_zero():
    assert helpers.extract_day_from_filename("randomfile.csv") == 0


def test_safe_corr_perfect_positive():
    x = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert helpers.safe_corr(x, x) == pytest.approx(1.0)


def test_safe_corr_constant_returns_nan():
    x = pd.Series([1.0, 2.0, 3.0, 4.0])
    y = pd.Series([5.0, 5.0, 5.0, 5.0])
    assert math.isnan(helpers.safe_corr(x, y))


def test_safe_corr_with_nans_drops_them():
    x = pd.Series([1.0, 2.0, np.nan, 4.0])
    y = pd.Series([2.0, 4.0, 6.0, 8.0])
    assert helpers.safe_corr(x, y) == pytest.approx(1.0)


def test_weighted_avg_basic():
    values = [1.0, 2.0, 3.0]
    weights = [1.0, 1.0, 1.0]
    assert helpers.weighted_avg(values, weights) == pytest.approx(2.0)


def test_weighted_avg_weighted():
    values = [1.0, 10.0]
    weights = [9.0, 1.0]
    assert helpers.weighted_avg(values, weights) == pytest.approx(1.9)


def test_weighted_avg_zero_weights_returns_nan():
    assert math.isnan(helpers.weighted_avg([1.0, 2.0], [0.0, 0.0]))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "imc_prosperity4/Round3/Analysis" && python3 -m pytest test_helpers.py -v
```

Expected: `ModuleNotFoundError: No module named 'helpers'` (or similar import error).

- [ ] **Step 3: Create minimal `helpers.py`**

Write `Round3/Analysis/helpers.py`:

```python
"""Shared helpers for Round 3 analysis notebooks.

See DESIGN.md for the full specification.
"""
from __future__ import annotations

import math
import os
import re
from typing import Iterable

import numpy as np
import pandas as pd


# ---------- data loading ----------

def extract_day_from_filename(path: str) -> int:
    """Pull the integer day N out of a filename like 'prices_round_3_day_-1.csv'.

    Returns 0 when no 'day_<int>' segment is present.
    """
    m = re.search(r"day_(-?\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


# ---------- tiny math utilities ----------

def safe_corr(x, y) -> float:
    """Pearson correlation that returns NaN on zero-variance or empty input.

    Aligns x and y, drops any row where either is NaN, then returns NaN if
    fewer than 2 rows remain or either side is constant.
    """
    z = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(z) < 2:
        return float("nan")
    if z["x"].std(ddof=0) == 0 or z["y"].std(ddof=0) == 0:
        return float("nan")
    return float(z["x"].corr(z["y"]))


def weighted_avg(values: Iterable[float], weights: Iterable[float]) -> float:
    """Volume-weighted average that returns NaN when all weights are zero/NaN."""
    values = np.asarray(list(values), dtype=float)
    weights = np.asarray(list(weights), dtype=float)
    mask = ~(np.isnan(values) | np.isnan(weights))
    values = values[mask]
    weights = weights[mask]
    total = weights.sum()
    if total == 0:
        return float("nan")
    return float((values * weights).sum() / total)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd "imc_prosperity4/Round3/Analysis" && python3 -m pytest test_helpers.py -v
```

Expected: 9 tests passed.

- [ ] **Step 5: Commit**

```bash
cd "imc_prosperity4" && \
git add Round3/Analysis/helpers.py Round3/Analysis/test_helpers.py && \
git commit -m "$(cat <<'EOF'
Round 3 helpers: math utilities + filename parser

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Data loading (`load_prices`, `load_trades`)

**Files:**
- Modify: `imc_prosperity4/Round3/Analysis/helpers.py`
- Modify: `imc_prosperity4/Round3/Analysis/test_helpers.py`

- [ ] **Step 1: Append failing tests to `test_helpers.py`**

Append (do not overwrite) to `Round3/Analysis/test_helpers.py`:

```python
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")


def test_load_prices_reads_all_days():
    df = helpers.load_prices(
        DATA_DIR,
        ["prices_round_3_day_0.csv",
         "prices_round_3_day_1.csv",
         "prices_round_3_day_2.csv"],
    )
    assert {"day", "timestamp", "product", "bid_price_1", "ask_price_1",
            "mid_price"}.issubset(df.columns)
    assert set(df["day"].unique()) == {0, 1, 2}
    assert "VELVETFRUIT_EXTRACT" in df["product"].unique()
    assert "VEV_5000" in df["product"].unique()


def test_load_trades_reads_and_renames_symbol():
    df = helpers.load_trades(
        DATA_DIR,
        ["trades_round_3_day_0.csv",
         "trades_round_3_day_1.csv",
         "trades_round_3_day_2.csv"],
    )
    assert {"day", "timestamp", "product", "price", "quantity"}.issubset(df.columns)
    assert set(df["day"].unique()) == {0, 1, 2}
    assert "symbol" not in df.columns, "load_trades should rename symbol -> product"
```

Also add `import os` at the top of the file if not already there (it is, from Task 1).

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "imc_prosperity4/Round3/Analysis" && python3 -m pytest test_helpers.py -v
```

Expected: 2 new tests fail with `AttributeError: module 'helpers' has no attribute 'load_prices'`.

- [ ] **Step 3: Add loaders to `helpers.py`**

Append to `Round3/Analysis/helpers.py`:

```python
def load_prices(data_dir: str, price_files: list[str]) -> pd.DataFrame:
    """Load and concatenate Prosperity price CSVs.

    CSVs are ';'-separated. Each row is augmented with an integer 'day' column
    parsed from the filename if the column is missing.
    """
    frames = []
    for fn in price_files:
        df = pd.read_csv(os.path.join(data_dir, fn), sep=";")
        if "day" not in df.columns:
            df["day"] = extract_day_from_filename(fn)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_trades(data_dir: str, trade_files: list[str]) -> pd.DataFrame:
    """Load and concatenate Prosperity trade CSVs.

    Renames the 'symbol' column to 'product' for consistency with prices, and
    tags each row with an integer 'day' from the filename.
    """
    frames = []
    for fn in trade_files:
        df = pd.read_csv(os.path.join(data_dir, fn), sep=";")
        df = df.rename(columns={"symbol": "product"})
        df["day"] = extract_day_from_filename(fn)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "imc_prosperity4/Round3/Analysis" && python3 -m pytest test_helpers.py -v
```

Expected: 11 tests passed.

- [ ] **Step 5: Commit**

```bash
cd "imc_prosperity4" && \
git add Round3/Analysis/helpers.py Round3/Analysis/test_helpers.py && \
git commit -m "$(cat <<'EOF'
Round 3 helpers: load_prices / load_trades

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Black–Scholes primitives (`bs_call_price`, `bs_call_delta`, `bs_call_vega`, `bs_call_gamma`)

Why these specific reference values: for an at-the-money European call with `S=K=100, T=1, r=0, σ=0.20`, Hull's *Options, Futures, and Other Derivatives* gives price ≈ 7.9656, delta ≈ 0.5398, vega ≈ 37.524, gamma ≈ 0.01988. These are the canonical cross-checks for a BS implementation.

**Files:**
- Modify: `imc_prosperity4/Round3/Analysis/helpers.py`
- Modify: `imc_prosperity4/Round3/Analysis/test_helpers.py`

- [ ] **Step 1: Append failing tests**

Append to `Round3/Analysis/test_helpers.py`:

```python
def test_bs_call_price_atm_reference():
    # S=K=100, T=1, r=0, sigma=0.20 -> 7.9656 per Hull ch.15
    assert helpers.bs_call_price(100, 100, 1.0, 0.20) == pytest.approx(7.9656, abs=1e-3)


def test_bs_call_price_intrinsic_when_expired():
    # T=0 -> price == max(S-K, 0)
    assert helpers.bs_call_price(110, 100, 0.0, 0.20) == pytest.approx(10.0)
    assert helpers.bs_call_price(90, 100, 0.0, 0.20) == pytest.approx(0.0)


def test_bs_call_delta_atm_around_half():
    assert helpers.bs_call_delta(100, 100, 1.0, 0.20) == pytest.approx(0.5398, abs=1e-3)


def test_bs_call_delta_deep_itm_near_one():
    assert helpers.bs_call_delta(200, 100, 1.0, 0.20) == pytest.approx(1.0, abs=1e-3)


def test_bs_call_delta_deep_otm_near_zero():
    assert helpers.bs_call_delta(50, 100, 1.0, 0.20) == pytest.approx(0.0, abs=1e-3)


def test_bs_call_vega_atm_reference():
    assert helpers.bs_call_vega(100, 100, 1.0, 0.20) == pytest.approx(37.524, abs=1e-2)


def test_bs_call_gamma_atm_reference():
    assert helpers.bs_call_gamma(100, 100, 1.0, 0.20) == pytest.approx(0.01988, abs=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "imc_prosperity4/Round3/Analysis" && python3 -m pytest test_helpers.py -v
```

Expected: 7 new tests fail with `AttributeError`.

- [ ] **Step 3: Add Black–Scholes primitives**

Append to `Round3/Analysis/helpers.py`:

```python
# ---------- Black-Scholes (r=0 by default for Prosperity) ----------

from math import log, sqrt, exp, pi


def _d1(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    return (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """European call price. T in years, r continuously compounded, sigma annualized."""
    if T <= 0 or sigma <= 0:
        return max(S - K * exp(-r * T), 0.0)
    d1 = _d1(S, K, T, sigma, r)
    d2 = d1 - sigma * sqrt(T)
    return S * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)


def bs_call_delta(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else (0.5 if S == K else 0.0)
    return _norm_cdf(_d1(S, K, T, sigma, r))


def bs_call_vega(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Vega per 1.00 change in sigma (i.e. in *absolute* units, not per 1%)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return S * _norm_pdf(_d1(S, K, T, sigma, r)) * sqrt(T)


def bs_call_gamma(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    return _norm_pdf(_d1(S, K, T, sigma, r)) / (S * sigma * sqrt(T))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "imc_prosperity4/Round3/Analysis" && python3 -m pytest test_helpers.py -v
```

Expected: 18 tests passed.

- [ ] **Step 5: Commit**

```bash
cd "imc_prosperity4" && \
git add Round3/Analysis/helpers.py Round3/Analysis/test_helpers.py && \
git commit -m "$(cat <<'EOF'
Round 3 helpers: Black-Scholes call primitives

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Implied-volatility solver (`implied_vol_call`)

**Files:**
- Modify: `imc_prosperity4/Round3/Analysis/helpers.py`
- Modify: `imc_prosperity4/Round3/Analysis/test_helpers.py`

- [ ] **Step 1: Append failing tests**

Append to `Round3/Analysis/test_helpers.py`:

```python
def test_implied_vol_roundtrips():
    # Compute BS price at sigma=0.25, then solve for IV; should recover 0.25.
    price = helpers.bs_call_price(100, 100, 0.5, 0.25)
    iv = helpers.implied_vol_call(price, 100, 100, 0.5)
    assert iv == pytest.approx(0.25, abs=1e-4)


def test_implied_vol_below_intrinsic_returns_nan():
    # Price below max(S-K, 0) is impossible; return NaN.
    iv = helpers.implied_vol_call(1.0, 200, 100, 0.5)  # intrinsic is 100
    assert math.isnan(iv)


def test_implied_vol_above_cap_returns_nan():
    # Price above reasonable S cap (S=100, sigma cap 5.0) returns NaN.
    iv = helpers.implied_vol_call(99.0, 100, 100, 0.5)
    assert math.isnan(iv)


def test_implied_vol_zero_tte_returns_nan():
    assert math.isnan(helpers.implied_vol_call(1.0, 100, 100, 0.0))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "imc_prosperity4/Round3/Analysis" && python3 -m pytest test_helpers.py -v
```

Expected: 4 new tests fail with `AttributeError`.

- [ ] **Step 3: Implement the solver**

Append to `Round3/Analysis/helpers.py`:

```python
from scipy.optimize import brentq


def implied_vol_call(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.0,
    lo: float = 1e-4,
    hi: float = 5.0,
    tol: float = 1e-6,
) -> float:
    """Solve BS implied vol for a European call via Brent's method.

    Returns NaN when:
    - T <= 0 (expired)
    - price is below intrinsic value max(S - K*exp(-rT), 0)
    - price sits above the value implied by sigma = hi (we don't extrapolate)
    """
    if T <= 0 or price <= 0 or S <= 0 or K <= 0:
        return float("nan")
    intrinsic = max(S - K * exp(-r * T), 0.0)
    if price < intrinsic - tol:
        return float("nan")
    f_lo = bs_call_price(S, K, T, lo, r) - price
    f_hi = bs_call_price(S, K, T, hi, r) - price
    if f_lo > 0 or f_hi < 0:
        return float("nan")
    try:
        return float(brentq(
            lambda sigma: bs_call_price(S, K, T, sigma, r) - price,
            lo, hi, xtol=tol,
        ))
    except ValueError:
        return float("nan")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "imc_prosperity4/Round3/Analysis" && python3 -m pytest test_helpers.py -v
```

Expected: 22 tests passed.

- [ ] **Step 5: Commit**

```bash
cd "imc_prosperity4" && \
git add Round3/Analysis/helpers.py Round3/Analysis/test_helpers.py && \
git commit -m "$(cat <<'EOF'
Round 3 helpers: implied volatility solver

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: TTE helper + constants (`tte_years`, `TIMESTAMPS_PER_DAY`, `YEAR_DAYS`, `TTE_DAYS_AT_DAY`)

**Files:**
- Modify: `imc_prosperity4/Round3/Analysis/helpers.py`
- Modify: `imc_prosperity4/Round3/Analysis/test_helpers.py`

- [ ] **Step 1: Append failing tests**

Append to `Round3/Analysis/test_helpers.py`:

```python
def test_tte_years_day0_start():
    # day 0, timestamp 0 -> 8 days / 365
    assert helpers.tte_years(0, 0) == pytest.approx(8 / 365, abs=1e-9)


def test_tte_years_day0_midday():
    # timestamp at half a day on day 0 -> (8 - 0.5) / 365
    assert helpers.tte_years(0, helpers.TIMESTAMPS_PER_DAY // 2) == pytest.approx(
        (8 - 0.5) / 365, abs=1e-6
    )


def test_tte_years_day2_end():
    # day 2, last timestamp before day 3 -> ~(6 - 1) / 365
    last = helpers.TIMESTAMPS_PER_DAY - 1
    assert helpers.tte_years(2, last) == pytest.approx(
        (6 - last / helpers.TIMESTAMPS_PER_DAY) / 365, abs=1e-6
    )


def test_constants_match_spec():
    assert helpers.TIMESTAMPS_PER_DAY == 10_000
    assert helpers.YEAR_DAYS == 365
    assert helpers.TTE_DAYS_AT_DAY == {0: 8, 1: 7, 2: 6}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "imc_prosperity4/Round3/Analysis" && python3 -m pytest test_helpers.py -v
```

Expected: 4 new tests fail with `AttributeError`.

- [ ] **Step 3: Add constants and `tte_years`**

Append to `Round3/Analysis/helpers.py`:

```python
# ---------- Round 3 option conventions ----------

TIMESTAMPS_PER_DAY = 10_000
YEAR_DAYS = 365
TTE_DAYS_AT_DAY = {0: 8, 1: 7, 2: 6}  # per wiki


def tte_years(day: int, timestamp: int) -> float:
    """Time to expiry in years for a Round 3 voucher at (day, timestamp).

    Uses a linear intra-day schedule: TTE at start of day d is TTE_DAYS_AT_DAY[d] days,
    decreasing linearly to TTE_DAYS_AT_DAY[d] - 1 over TIMESTAMPS_PER_DAY ticks.
    """
    days_remaining = TTE_DAYS_AT_DAY[day] - timestamp / TIMESTAMPS_PER_DAY
    return days_remaining / YEAR_DAYS
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "imc_prosperity4/Round3/Analysis" && python3 -m pytest test_helpers.py -v
```

Expected: 26 tests passed.

- [ ] **Step 5: Commit**

```bash
cd "imc_prosperity4" && \
git add Round3/Analysis/helpers.py Round3/Analysis/test_helpers.py && \
git commit -m "$(cat <<'EOF'
Round 3 helpers: TTE schedule and constants

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Port `build_book_features` from Round 1

**Files:**
- Modify: `imc_prosperity4/Round3/Analysis/helpers.py`
- Modify: `imc_prosperity4/Round3/Analysis/test_helpers.py`
- Reference (read-only): `imc_prosperity4/Round1/analysis_v2.py` lines 65-278

- [ ] **Step 1: Append sanity test**

Append to `Round3/Analysis/test_helpers.py`:

```python
EXPECTED_BOOK_COLS = {
    "touch_mid", "boundary_mid", "size_wall_mid", "side_vwap_mid",
    "full_book_vwap_center",
    "touch_spread", "boundary_width", "size_wall_width",
    "frontier_imbalance", "depth_imbalance", "size_wall_vol_imbalance",
    "fwd_touch_mid_change_10", "fwd_direction_10",
}


def test_build_book_features_adds_expected_columns():
    prices = helpers.load_prices(DATA_DIR, ["prices_round_3_day_0.csv"])
    book = helpers.build_book_features(prices)
    missing = EXPECTED_BOOK_COLS - set(book.columns)
    assert not missing, f"missing columns: {missing}"


def test_build_book_features_touch_mid_matches_dataset_mid_for_normal_rows():
    prices = helpers.load_prices(DATA_DIR, ["prices_round_3_day_0.csv"])
    book = helpers.build_book_features(prices)
    # For rows where bid_price_1 and ask_price_1 are both present and mid_price>0
    mask = book["mid_price"] > 0
    diff = (book.loc[mask, "touch_mid"] - book.loc[mask, "mid_price"]).abs()
    assert diff.max() <= 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "imc_prosperity4/Round3/Analysis" && python3 -m pytest test_helpers.py -v
```

Expected: 2 new tests fail with `AttributeError`.

- [ ] **Step 3: Port the function**

Read `imc_prosperity4/Round1/analysis_v2.py` lines 19-278. Port the following names into `Round3/Analysis/helpers.py`, keeping them textually identical except where noted:

1. Module constants (line 24-25): `VISIBLE_LEVELS = 3`, `HORIZON = 10`
2. Row-level helpers (lines 65-116): `row_weighted_avg`, `row_boundary_bid`, `row_boundary_ask`, `row_size_wall_price_and_volume`, `next_trade_direction_from_price_vs_mid`
3. `build_book_features` (lines 148-278)

Do not port `load_prices` / `load_trades` from Round 1 — we have our own in Task 2.
Do not port `print_*` functions — those are report helpers only used in Round 1's CLI.

Append the code verbatim to `Round3/Analysis/helpers.py` under a clearly-commented section header:

```python
# ---------- book feature engineering (ported from Round1/analysis_v2.py) ----------

VISIBLE_LEVELS = 3
HORIZON = 10

# ... (paste row_weighted_avg, row_boundary_bid, row_boundary_ask,
# row_size_wall_price_and_volume, next_trade_direction_from_price_vs_mid
# from Round1/analysis_v2.py lines 65-116 unchanged)

# ... (paste build_book_features from Round1/analysis_v2.py lines 148-278 unchanged)
```

After the paste, grep to confirm the new constants / functions appear exactly once:

```bash
cd "imc_prosperity4/Round3/Analysis" && grep -c "^def build_book_features" helpers.py
```

Expected: `1`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "imc_prosperity4/Round3/Analysis" && python3 -m pytest test_helpers.py -v
```

Expected: 28 tests passed.

- [ ] **Step 5: Commit**

```bash
cd "imc_prosperity4" && \
git add Round3/Analysis/helpers.py Round3/Analysis/test_helpers.py && \
git commit -m "$(cat <<'EOF'
Round 3 helpers: port build_book_features from Round 1

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Port `build_trade_features` + `merge_book_and_trade` from Round 1

**Important contract reminder:** `build_trade_features` returns a tuple
`(agg_df, raw_trades_df)`. `agg_df` is per-`(product, day, timestamp)` and has
`trade_sign_proxy`, `next_trade_sign_proxy`, `frac_*`. `raw_trades_df` is per
trade and has `trade_location`. Both notebooks rely on this shape.
`merge_book_and_trade(book_df, agg_df)` folds `agg_df` into the book frame.

**Files:**
- Modify: `imc_prosperity4/Round3/Analysis/helpers.py`
- Modify: `imc_prosperity4/Round3/Analysis/test_helpers.py`
- Reference (read-only): `imc_prosperity4/Round1/analysis_v2.py` lines 280-354

- [ ] **Step 1: Append sanity tests**

Append to `Round3/Analysis/test_helpers.py`:

```python
def test_build_trade_features_returns_agg_and_raw_trades():
    prices = helpers.load_prices(DATA_DIR, ["prices_round_3_day_0.csv"])
    trades = helpers.load_trades(DATA_DIR, ["trades_round_3_day_0.csv"])
    book = helpers.build_book_features(prices)
    agg, raw_trades = helpers.build_trade_features(trades, book)
    # agg has per-tick aggregates
    assert "trade_sign_proxy" in agg.columns
    assert "next_trade_sign_proxy" in agg.columns
    assert set(agg["trade_sign_proxy"].dropna().unique()).issubset({-1.0, 0.0, 1.0})
    # raw_trades has the per-trade location
    assert "trade_location" in raw_trades.columns


def test_merge_book_and_trade_attaches_agg_columns():
    prices = helpers.load_prices(DATA_DIR, ["prices_round_3_day_0.csv"])
    trades = helpers.load_trades(DATA_DIR, ["trades_round_3_day_0.csv"])
    book = helpers.build_book_features(prices)
    agg, _ = helpers.build_trade_features(trades, book)
    merged = helpers.merge_book_and_trade(book, agg)
    for col in ["trade_count", "trade_volume", "trade_sign_proxy",
                "next_trade_sign_proxy", "frac_at_or_above_ask_touch"]:
        assert col in merged.columns
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "imc_prosperity4/Round3/Analysis" && python3 -m pytest test_helpers.py -v
```

Expected: 2 new tests fail with `AttributeError`.

- [ ] **Step 3: Port both functions verbatim**

Read `Round1/analysis_v2.py` lines 280-354 (the bodies of `build_trade_features`
and `merge_book_and_trade`). Append both functions to `Round3/Analysis/helpers.py`
under a clearly-commented header. Do not modify the function bodies.

```python
# ---------- trade feature engineering (ported from Round1/analysis_v2.py) ----------

# def build_trade_features(trades, book_df) -> (agg_df, raw_trades_df):
#     ... copy lines 280-341 verbatim from Round1/analysis_v2.py ...

# def merge_book_and_trade(book_df, trade_agg) -> book_with_trade_cols:
#     ... copy lines 344-354 verbatim from Round1/analysis_v2.py ...
```

After pasting, verify with grep:

```bash
cd "imc_prosperity4/Round3/Analysis" && \
grep -c "^def build_trade_features\|^def merge_book_and_trade" helpers.py
```

Expected: `2`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "imc_prosperity4/Round3/Analysis" && python3 -m pytest test_helpers.py -v
```

Expected: 30 tests passed.

- [ ] **Step 5: Commit**

```bash
cd "imc_prosperity4" && \
git add Round3/Analysis/helpers.py Round3/Analysis/test_helpers.py && \
git commit -m "$(cat <<'EOF'
Round 3 helpers: port build_trade_features and merge_book_and_trade

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Convention for notebook tasks (8–13)

Each notebook-building task follows the same shape:

1. Write a small `build_<name>.py` script that constructs the notebook via `nbformat` (one cell per logical unit of analysis), putting it into `imc_prosperity4/Round3/Analysis/`.
2. Run the script to produce the `.ipynb`.
3. Execute the notebook end-to-end with `jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 --inplace <notebook>.ipynb` — this is the "test" of the notebook: it must run without error from a fresh kernel.
4. Commit the `.ipynb` and the `build_<name>.py` script.

The build scripts remain in the repo so anyone can regenerate the notebooks deterministically. They also serve as a diff-friendly source of truth for the cell content.

---

## Task 8: `analysis_delta1.ipynb` — Parts A + B (setup, load, cross-product overview)

**Files:**
- Create: `imc_prosperity4/Round3/Analysis/build_delta1_notebook.py`
- Create: `imc_prosperity4/Round3/Analysis/analysis_delta1.ipynb` (generated)

- [ ] **Step 1: Write the build script (Parts A + B only for this task)**

Write `Round3/Analysis/build_delta1_notebook.py`:

```python
"""Generate analysis_delta1.ipynb from a deterministic cell list.

Run me: `python3 build_delta1_notebook.py`
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()

C = []  # list of (kind, source)
md = lambda s: C.append(("md", s))
py = lambda s: C.append(("py", s))

# ---------- Part A: Setup & load ----------
md("""# Round 3 — Delta-1 Analysis

Analysis of the two delta-1 products in Round 3: `HYDROGEL_PACK` and
`VELVETFRUIT_EXTRACT`. VELVETFRUIT_EXTRACT is also the underlying of the 10
VEV vouchers — see `analysis_options.ipynb`.

See `DESIGN.md` for the notebook outline.""")

md("## A. Setup & load")

py("""import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import helpers

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 100)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")

sns.set_theme(context="notebook", style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 4)""")

py("""DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "Data"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.getcwd(), "output"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRICE_FILES = [
    "prices_round_3_day_0.csv",
    "prices_round_3_day_1.csv",
    "prices_round_3_day_2.csv",
]
TRADE_FILES = [
    "trades_round_3_day_0.csv",
    "trades_round_3_day_1.csv",
    "trades_round_3_day_2.csv",
]

DELTA1_PRODUCTS = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]
VISIBLE_LEVELS = 3
HORIZON = 10""")

py("""prices_all = helpers.load_prices(DATA_DIR, PRICE_FILES)
trades_all = helpers.load_trades(DATA_DIR, TRADE_FILES)

prices = prices_all[prices_all["product"].isin(DELTA1_PRODUCTS)].copy()
trades = trades_all[trades_all["product"].isin(DELTA1_PRODUCTS)].copy()

book = helpers.build_book_features(prices)
trade_agg, raw_trades = helpers.build_trade_features(trades, book)
df = helpers.merge_book_and_trade(book, trade_agg)

print(f"book rows       : {len(book):>7,}")
print(f"trade agg rows  : {len(trade_agg):>7,}")
print(f"raw trade rows  : {len(raw_trades):>7,}")
print(f"merged df rows  : {len(df):>7,}")""")

md("### A.5 Data quality summary")

py("""qc = (df.groupby(["product", "day"])
        .agg(rows=("timestamp", "size"),
             nan_mid=("mid_price", lambda s: s.isna().sum()),
             zero_mid=("mid_price", lambda s: (s == 0).sum()))
        .reset_index())
display(qc)

bad_mask = df["mid_price"] == 0
print(f"zero-mid rows: {bad_mask.sum()}  ({100*bad_mask.mean():.3f}%)")
df_clean = df[~bad_mask].copy()""")

# ---------- Part B: Cross-product overview ----------
md("## B. Cross-product overview")

md("""Both products plotted together across the three days to spot regime shifts
and coupling.""")

py("""fig, axes = plt.subplots(len(DELTA1_PRODUCTS), 1,
                         figsize=(12, 3 * len(DELTA1_PRODUCTS)), sharex=False)
if len(DELTA1_PRODUCTS) == 1:
    axes = [axes]
for ax, prod in zip(axes, DELTA1_PRODUCTS):
    for d, color in zip(sorted(df_clean["day"].unique()),
                        sns.color_palette("tab10")):
        s = df_clean[(df_clean["product"] == prod) & (df_clean["day"] == d)]
        ax.plot(s["timestamp"].to_numpy() + d * helpers.TIMESTAMPS_PER_DAY,
                s["mid_price"].to_numpy(), color=color, label=f"day {d}", lw=0.8)
    ax.set_title(f"{prod}: mid_price across 3 days")
    ax.legend(loc="upper right", ncol=3)
plt.tight_layout()
plt.show()""")

md("### B.3 Per-day OLS drift slope")

py("""rows = []
for prod in DELTA1_PRODUCTS:
    for d in sorted(df_clean["day"].unique()):
        s = df_clean[(df_clean["product"] == prod) & (df_clean["day"] == d)].sort_values("timestamp")
        if len(s) < 10:
            continue
        slope, intercept = np.polyfit(s["timestamp"].to_numpy(float),
                                      s["mid_price"].to_numpy(float), 1)
        rows.append({"product": prod, "day": d, "slope_per_tick": slope,
                     "intercept": intercept, "r_points": len(s)})
drift = pd.DataFrame(rows)
display(drift)""")

md("### B.4 Tick-return distribution, autocorrelation, rolling volatility")

py("""MAX_LAG = 20
ROLL_WIN = 200

fig, axes = plt.subplots(len(DELTA1_PRODUCTS), 3,
                         figsize=(15, 3 * len(DELTA1_PRODUCTS)))
if len(DELTA1_PRODUCTS) == 1:
    axes = axes.reshape(1, -1)

for row, prod in enumerate(DELTA1_PRODUCTS):
    s = df_clean[df_clean["product"] == prod].sort_values(["day", "timestamp"])
    rets = s.groupby("day")["mid_price"].diff()
    axes[row, 0].hist(rets.dropna(), bins=80)
    axes[row, 0].set_title(f"{prod}: tick returns")
    # autocorr
    acf = [rets.autocorr(lag) for lag in range(1, MAX_LAG + 1)]
    axes[row, 1].bar(range(1, MAX_LAG + 1), acf)
    axes[row, 1].axhline(0, color="k", lw=0.5)
    axes[row, 1].set_title(f"{prod}: autocorrelation (lag 1..{MAX_LAG})")
    # rolling vol
    rv = rets.rolling(ROLL_WIN).std()
    axes[row, 2].plot(rv.to_numpy())
    axes[row, 2].set_title(f"{prod}: rolling std (win={ROLL_WIN})")
plt.tight_layout()
plt.show()""")

md("### B.5 Cross-product tick-return correlation per day")

py("""rows = []
for d in sorted(df_clean["day"].unique()):
    pivot = (df_clean[df_clean["day"] == d]
             .pivot_table(index="timestamp", columns="product", values="mid_price")
             .sort_index())
    rets = pivot.diff().dropna()
    if rets.shape[1] < 2:
        continue
    rows.append({
        "day": d,
        "corr": helpers.safe_corr(rets[DELTA1_PRODUCTS[0]], rets[DELTA1_PRODUCTS[1]]),
        "n": len(rets),
    })
display(pd.DataFrame(rows))""")

# ---------- emit notebook ----------
for kind, src in C:
    if kind == "md":
        nb.cells.append(nbf.v4.new_markdown_cell(src))
    else:
        nb.cells.append(nbf.v4.new_code_cell(src))

nbf.write(nb, "analysis_delta1.ipynb")
print(f"wrote analysis_delta1.ipynb with {len(nb.cells)} cells")
```

- [ ] **Step 2: Run the build script**

```bash
cd "imc_prosperity4/Round3/Analysis" && python3 build_delta1_notebook.py
```

Expected: `wrote analysis_delta1.ipynb with 16 cells`.

- [ ] **Step 3: Execute the notebook end-to-end**

```bash
cd "imc_prosperity4/Round3/Analysis" && jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 --inplace analysis_delta1.ipynb
```

Expected: finishes with no error. If a cell raises, fix it in the build script, regenerate, and re-execute.

- [ ] **Step 4: Commit**

```bash
cd "imc_prosperity4" && \
git add Round3/Analysis/build_delta1_notebook.py Round3/Analysis/analysis_delta1.ipynb && \
git commit -m "$(cat <<'EOF'
Round 3 delta-1 notebook: setup + cross-product overview

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: `analysis_delta1.ipynb` — Parts C + D + E (per-product drill-down, velvetfruit, exports)

**Files:**
- Modify: `imc_prosperity4/Round3/Analysis/build_delta1_notebook.py`
- Modify: `imc_prosperity4/Round3/Analysis/analysis_delta1.ipynb` (regenerated)

- [ ] **Step 1: Append cells to the build script**

In `build_delta1_notebook.py`, insert the following cells **before** the final `for kind, src in C:` emit loop (i.e. after the Part B cells from Task 8):

```python
# ---------- Part C: Per-product drill-down ----------
md("## C. Per-product drill-down")

md("Each cell loops over both delta-1 products and emits one figure per product.")

py("""def describe_static_geometry(x, prod):
    summary = {
        "touch_spread_mean":    x["touch_spread"].mean(),
        "touch_spread_median":  x["touch_spread"].median(),
        "boundary_width_mean":  x["boundary_width"].mean(),
        "size_wall_width_mean": x["size_wall_width"].mean(),
        "rows":                 len(x),
    }
    return summary

geom_rows = []
for prod in DELTA1_PRODUCTS:
    x = df_clean[df_clean["product"] == prod]
    geom_rows.append({"product": prod, **describe_static_geometry(x, prod)})
display(pd.DataFrame(geom_rows))""")

py("""fig, axes = plt.subplots(len(DELTA1_PRODUCTS), 2,
                         figsize=(13, 3.5 * len(DELTA1_PRODUCTS)))
if len(DELTA1_PRODUCTS) == 1:
    axes = axes.reshape(1, -1)
for row, prod in enumerate(DELTA1_PRODUCTS):
    x = df_clean[df_clean["product"] == prod]
    sns.histplot(x["touch_spread"], bins=40, ax=axes[row, 0])
    axes[row, 0].set_title(f"{prod}: touch_spread")
    sns.histplot(x["size_wall_width"], bins=40, ax=axes[row, 1])
    axes[row, 1].set_title(f"{prod}: size_wall_width")
plt.tight_layout()
plt.show()""")

py("""FAIR_VALUE_COLS = [
    "touch_mid", "boundary_mid", "size_wall_mid",
    "side_vwap_mid", "full_book_vwap_center", "mid_price",
]

for prod in DELTA1_PRODUCTS:
    x = df_clean[df_clean["product"] == prod]
    corr = x[FAIR_VALUE_COLS].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title(f"{prod}: fair-value estimator correlation")
    plt.tight_layout()
    plt.show()""")

py("""PREDICTOR_FEATURES = [
    "touch_spread", "boundary_width", "size_wall_width",
    "frontier_imbalance", "depth_imbalance", "size_wall_vol_imbalance",
    "boundary_mid_minus_touch_mid", "size_wall_mid_minus_touch_mid",
    "side_vwap_mid_minus_touch_mid", "full_book_vwap_center_minus_touch_mid",
    "mid_price_minus_touch_mid",
]
target_change = f"fwd_touch_mid_change_{HORIZON}"
target_dir    = f"fwd_direction_{HORIZON}"

pred_rows = []
for prod in DELTA1_PRODUCTS:
    x = df_clean[df_clean["product"] == prod]
    for feat in PREDICTOR_FEATURES:
        if feat not in x.columns:
            continue
        pred_rows.append({
            "product": prod,
            "feature": feat,
            "corr_with_fwd_change":    helpers.safe_corr(x[feat], x[target_change]),
            "corr_with_fwd_direction": helpers.safe_corr(x[feat], x[target_dir]),
        })
pred_df = (pd.DataFrame(pred_rows)
             .assign(abs_corr=lambda d: d["corr_with_fwd_change"].abs())
             .sort_values(["product", "abs_corr"], ascending=[True, False])
             .drop(columns="abs_corr"))
display(pred_df)""")

py("""rows = []
for prod in DELTA1_PRODUCTS:
    t = raw_trades[raw_trades["product"] == prod].copy()
    if "trade_location" not in t.columns:
        continue
    loc = t["trade_location"].value_counts(dropna=False).to_frame("count")
    loc["fraction"] = loc["count"] / loc["count"].sum()
    loc["product"] = prod
    rows.append(loc.reset_index().rename(columns={"index": "trade_location"}))
if rows:
    display(pd.concat(rows, ignore_index=True))
else:
    print("No trade_location column — check build_trade_features output.")""")

py("""next_sign_rows = []
for prod in DELTA1_PRODUCTS:
    x = df_clean[df_clean["product"] == prod]
    if "next_trade_sign_proxy" not in x.columns:
        continue
    for feat in ["frontier_imbalance", "depth_imbalance", "size_wall_vol_imbalance",
                 "touch_spread", "boundary_width"]:
        next_sign_rows.append({
            "product": prod,
            "feature": feat,
            "corr_with_next_trade_sign": helpers.safe_corr(x[feat], x["next_trade_sign_proxy"]),
        })
next_sign_df = pd.DataFrame(next_sign_rows)
display(next_sign_df)""")

# ---------- Part D: VELVETFRUIT-specific ----------
md("## D. VELVETFRUIT_EXTRACT — underlying of vouchers")

md("""`VELVETFRUIT_EXTRACT` is the underlying of the 10 VEV vouchers. Its realized
volatility directly feeds the options book — see `analysis_options.ipynb`.""")

py("""vev = df_clean[df_clean["product"] == "VELVETFRUIT_EXTRACT"].sort_values(["day", "timestamp"]).copy()
vev["ret"] = vev.groupby("day")["mid_price"].diff()

rv_frames = []
for win in (100, 500, 2000):
    rv = vev.groupby("day")["ret"].transform(lambda s: s.rolling(win).std())
    rv_frames.append(rv.rename(f"rv_{win}"))
rv_df = pd.concat([vev[["day", "timestamp"]].reset_index(drop=True)] +
                  [f.reset_index(drop=True) for f in rv_frames], axis=1)
display(rv_df.describe())

fig, ax = plt.subplots(figsize=(12, 4))
for col in [f"rv_{w}" for w in (100, 500, 2000)]:
    ax.plot(rv_df[col].to_numpy(), label=col, lw=0.8)
ax.set_title("VELVETFRUIT_EXTRACT realized volatility of tick returns")
ax.legend()
plt.tight_layout()
plt.show()""")

# ---------- Part E: Exports ----------
md("## E. Artifact exports")

py("""# delta1_feature_summary.csv
summary_rows = []
for prod in DELTA1_PRODUCTS:
    x = df_clean[df_clean["product"] == prod]
    summary_rows.append({
        "product": prod,
        "rows": len(x),
        "touch_spread_mean": x["touch_spread"].mean(),
        "touch_spread_median": x["touch_spread"].median(),
        "boundary_width_mean": x["boundary_width"].mean(),
        "size_wall_width_mean": x["size_wall_width"].mean(),
        "mean_mid_price": x["mid_price"].mean(),
        "per_day_drift_slope_mean": (drift[drift["product"] == prod]["slope_per_tick"].mean()
                                      if not drift.empty else float("nan")),
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "delta1_feature_summary.csv"), index=False)

# delta1_predictor_ranking.csv
ranking = pred_df.merge(next_sign_df, on=["product", "feature"], how="outer")
ranking.to_csv(os.path.join(OUTPUT_DIR, "delta1_predictor_ranking.csv"), index=False)

# velvetfruit_realized_vol.csv
rv_df.to_csv(os.path.join(OUTPUT_DIR, "velvetfruit_realized_vol.csv"), index=False)

for name in ["delta1_feature_summary.csv",
             "delta1_predictor_ranking.csv",
             "velvetfruit_realized_vol.csv"]:
    path = os.path.join(OUTPUT_DIR, name)
    print(f"{name:40s} {os.path.getsize(path):>8,d} bytes")""")

md("""### Next: `analysis_options.ipynb`

The options notebook reuses `VELVETFRUIT_EXTRACT` as its underlying and pulls
realized vol context from `output/velvetfruit_realized_vol.csv`.""")
```

- [ ] **Step 2: Regenerate the notebook**

```bash
cd "imc_prosperity4/Round3/Analysis" && python3 build_delta1_notebook.py
```

Expected: `wrote analysis_delta1.ipynb with 30 cells` (16 from Task 8 + 14 new).

- [ ] **Step 3: Execute end-to-end**

```bash
cd "imc_prosperity4/Round3/Analysis" && jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 --inplace analysis_delta1.ipynb
```

Expected: no error.

- [ ] **Step 4: Confirm exports exist**

```bash
cd "imc_prosperity4/Round3/Analysis" && ls -la output/
```

Expected: `delta1_feature_summary.csv`, `delta1_predictor_ranking.csv`, `velvetfruit_realized_vol.csv` all present, each non-zero bytes.

- [ ] **Step 5: Commit**

```bash
cd "imc_prosperity4" && \
git add Round3/Analysis/build_delta1_notebook.py Round3/Analysis/analysis_delta1.ipynb Round3/Analysis/output/ && \
git commit -m "$(cat <<'EOF'
Round 3 delta-1 notebook: per-product drill-down + exports

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: `analysis_options.ipynb` — Parts A + B (setup, load, price behaviour)

**Files:**
- Create: `imc_prosperity4/Round3/Analysis/build_options_notebook.py`
- Create: `imc_prosperity4/Round3/Analysis/analysis_options.ipynb` (generated)

- [ ] **Step 1: Write the build script (Parts A + B)**

Write `Round3/Analysis/build_options_notebook.py`:

```python
"""Generate analysis_options.ipynb from a deterministic cell list.

Run me: `python3 build_options_notebook.py`
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(("md", s))
py = lambda s: C.append(("py", s))

# ---------- Part A ----------
md("""# Round 3 — Options Analysis

Analysis of the 10 VEV vouchers and their underlying `VELVETFRUIT_EXTRACT`.

See `DESIGN.md` for the notebook outline.""")

md("## A. Setup & load")

py("""import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import helpers

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 100)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")

sns.set_theme(context="notebook", style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 4)""")

py("""DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "Data"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.getcwd(), "output"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRICE_FILES = [
    "prices_round_3_day_0.csv",
    "prices_round_3_day_1.csv",
    "prices_round_3_day_2.csv",
]
TRADE_FILES = [
    "trades_round_3_day_0.csv",
    "trades_round_3_day_1.csv",
    "trades_round_3_day_2.csv",
]

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{K}" for K in STRIKES]
UNDERLYING = "VELVETFRUIT_EXTRACT"

IV_DOWNSAMPLE = 10  # compute IV on every Nth row""")

py("""prices_all = helpers.load_prices(DATA_DIR, PRICE_FILES)

# Underlying path + voucher rows
u = prices_all[prices_all["product"] == UNDERLYING].copy()
v = prices_all[prices_all["product"].isin(VOUCHERS)].copy()

# Attach strike integer + TTE years
v["strike"] = v["product"].str.replace("VEV_", "", regex=False).astype(int)
v["tte_years"] = [helpers.tte_years(int(d), int(t))
                  for d, t in zip(v["day"], v["timestamp"])]

# Attach underlying mid on (day, timestamp) — prices are tick-aligned
u_key = u[["day", "timestamp", "mid_price"]].rename(columns={"mid_price": "underlying_mid"})
v = v.merge(u_key, on=["day", "timestamp"], how="left")
v["moneyness"] = v["underlying_mid"] / v["strike"]

print(f"underlying rows: {len(u):>7,}")
print(f"voucher rows   : {len(v):>7,}")""")

md("### A.5 Data quality summary")

py("""qc = (v.groupby(["strike", "day"])
        .agg(rows=("timestamp", "size"),
             nan_mid=("mid_price", lambda s: s.isna().sum()),
             zero_mid=("mid_price", lambda s: (s == 0).sum()),
             mean_quoted_size=("bid_volume_1", lambda s: s.fillna(0).mean()))
        .reset_index())
display(qc)""")

# ---------- Part B ----------
md("## B. Price behaviour")

py("""fig, ax = plt.subplots(figsize=(12, 4))
for d, color in zip(sorted(u["day"].unique()), sns.color_palette("tab10")):
    s = u[u["day"] == d].sort_values("timestamp")
    ax.plot(s["timestamp"].to_numpy() + d * helpers.TIMESTAMPS_PER_DAY,
            s["mid_price"].to_numpy(), label=f"day {d}", color=color, lw=0.8)
ax.set_title(f"{UNDERLYING}: mid_price across 3 days")
ax.legend()
plt.tight_layout()
plt.show()""")

py("""fig, axes = plt.subplots(3, 4, figsize=(18, 9), sharex=False)
for ax, K in zip(axes.ravel(), STRIKES):
    s_v = v[v["strike"] == K].sort_values(["day", "timestamp"])
    for d, color in zip(sorted(s_v["day"].unique()), sns.color_palette("tab10")):
        sd = s_v[s_v["day"] == d]
        ax.plot(sd["timestamp"].to_numpy() + d * helpers.TIMESTAMPS_PER_DAY,
                sd["mid_price"].to_numpy(), color=color, lw=0.7, label=f"d{d}")
    ax.set_title(f"VEV_{K}")
# turn off unused axes
for ax in axes.ravel()[len(STRIKES):]:
    ax.axis("off")
plt.tight_layout()
plt.show()""")

py("""per_strike = (v.groupby("strike")
                .agg(rows=("timestamp", "size"),
                     mean_mid=("mid_price", "mean"),
                     mean_spread=("mid_price", lambda s: (s.std() if len(s) else np.nan)),
                     mean_quoted_size_bid=("bid_volume_1", lambda s: s.fillna(0).mean()),
                     mean_quoted_size_ask=("ask_volume_1", lambda s: s.fillna(0).mean()))
                .reset_index())
# Recompute mean_spread more precisely as ask_price_1 - bid_price_1
spread = (v.assign(spread=v["ask_price_1"] - v["bid_price_1"])
            .groupby("strike")["spread"].mean().rename("mean_spread"))
per_strike = per_strike.drop(columns="mean_spread").merge(spread, on="strike")
display(per_strike)""")

# emit
for kind, src in C:
    if kind == "md":
        nb.cells.append(nbf.v4.new_markdown_cell(src))
    else:
        nb.cells.append(nbf.v4.new_code_cell(src))

nbf.write(nb, "analysis_options.ipynb")
print(f"wrote analysis_options.ipynb with {len(nb.cells)} cells")
```

- [ ] **Step 2: Run the build script**

```bash
cd "imc_prosperity4/Round3/Analysis" && python3 build_options_notebook.py
```

Expected: `wrote analysis_options.ipynb with 11 cells`.

- [ ] **Step 3: Execute end-to-end**

```bash
cd "imc_prosperity4/Round3/Analysis" && jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 --inplace analysis_options.ipynb
```

Expected: no error.

- [ ] **Step 4: Commit**

```bash
cd "imc_prosperity4" && \
git add Round3/Analysis/build_options_notebook.py Round3/Analysis/analysis_options.ipynb && \
git commit -m "$(cat <<'EOF'
Round 3 options notebook: setup + price behaviour

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: `analysis_options.ipynb` — Part C (voucher ↔ underlying)

**Files:**
- Modify: `imc_prosperity4/Round3/Analysis/build_options_notebook.py`
- Modify: `imc_prosperity4/Round3/Analysis/analysis_options.ipynb` (regenerated)

- [ ] **Step 1: Append Part C cells to the build script**

In `build_options_notebook.py`, insert the following **before** the emit loop:

```python
# ---------- Part C: Voucher <-> underlying ----------
md("## C. Voucher ↔ underlying relationship")

md("""If a voucher is a real option, its mid should move with the underlying.
The scatter slope per strike is an empirical delta.

Moneyness = underlying / strike. Moneyness < 1 = out-of-money, > 1 = in-the-money.""")

py("""fig, axes = plt.subplots(3, 4, figsize=(18, 11))
slopes = {}
for ax, K in zip(axes.ravel(), STRIKES):
    s = v[v["strike"] == K].dropna(subset=["mid_price", "underlying_mid"])
    if len(s) < 100:
        ax.axis("off"); continue
    ax.scatter(s["underlying_mid"], s["mid_price"], s=2, alpha=0.3)
    slope, intercept = np.polyfit(s["underlying_mid"].to_numpy(float),
                                  s["mid_price"].to_numpy(float), 1)
    slopes[K] = slope
    xs = np.linspace(s["underlying_mid"].min(), s["underlying_mid"].max(), 50)
    ax.plot(xs, slope * xs + intercept, color="red", lw=1.0)
    mean_money = s["moneyness"].mean()
    ax.set_title(f"VEV_{K}: slope={slope:.3f}  moneyness≈{mean_money:.3f}")
for ax in axes.ravel()[len(STRIKES):]:
    ax.axis("off")
plt.tight_layout()
plt.show()
print("Empirical deltas:", {k: round(s, 4) for k, s in slopes.items()})""")

py("""ROLL = 500
fig, ax = plt.subplots(figsize=(13, 4))
for K in STRIKES:
    s = v[v["strike"] == K].sort_values(["day", "timestamp"]).copy()
    s["voucher_ret"] = s.groupby("day")["mid_price"].diff()
    s["underlying_ret"] = s.groupby("day")["underlying_mid"].diff()
    rc = (s[["voucher_ret", "underlying_ret"]]
            .rolling(ROLL).corr().unstack()["voucher_ret"]["underlying_ret"])
    ax.plot(rc.to_numpy(), label=f"VEV_{K}", lw=0.8)
ax.set_title(f"Rolling-{ROLL} correlation of tick returns vs underlying")
ax.legend(ncol=5, fontsize=8, loc="lower center")
plt.tight_layout()
plt.show()""")

py("""viols = v.assign(intrinsic=np.maximum(v["underlying_mid"] - v["strike"], 0))
viols = viols[viols["mid_price"] < viols["intrinsic"] - 1e-6]
print(f"intrinsic-floor violations: {len(viols)} rows "
      f"({100*len(viols)/max(len(v), 1):.3f}%)")
if len(viols):
    display(viols.groupby("strike").size().rename("violations"))""")
```

- [ ] **Step 2: Regenerate and execute**

```bash
cd "imc_prosperity4/Round3/Analysis" && \
python3 build_options_notebook.py && \
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 --inplace analysis_options.ipynb
```

Expected: build script prints `wrote analysis_options.ipynb with 16 cells`; nbconvert finishes without error.

- [ ] **Step 3: Commit**

```bash
cd "imc_prosperity4" && \
git add Round3/Analysis/build_options_notebook.py Round3/Analysis/analysis_options.ipynb && \
git commit -m "$(cat <<'EOF'
Round 3 options notebook: voucher <-> underlying relationship

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: `analysis_options.ipynb` — Part D (implied volatility core)

**Files:**
- Modify: `imc_prosperity4/Round3/Analysis/build_options_notebook.py`
- Modify: `imc_prosperity4/Round3/Analysis/analysis_options.ipynb` (regenerated)

- [ ] **Step 1: Append Part D cells**

In `build_options_notebook.py`, insert the following before the emit loop:

```python
# ---------- Part D: Implied volatility ----------
md("## D. Implied volatility core")

md("""Implied volatility (IV) is the sigma that makes the BS price match the
observed voucher mid. Plotted vs strike at a snapshot, the U-shape is the
**smile** — far-from-money options are richer on a vol basis than at-the-money.""")

py("""# Downsample for IV solves. Keep a copy of raw 'v' for plots that don't need IV.
iv_rows = v.iloc[::IV_DOWNSAMPLE].copy().reset_index(drop=True)
print(f"IV solve rows: {len(iv_rows):,}")

def _iv(row):
    return helpers.implied_vol_call(
        price=float(row["mid_price"]),
        S=float(row["underlying_mid"]),
        K=float(row["strike"]),
        T=float(row["tte_years"]),
    )

iv_rows["iv"] = iv_rows.apply(_iv, axis=1)
finite = iv_rows["iv"].notna().mean()
print(f"finite IV fraction: {finite:.3f}")""")

py("""fig, ax = plt.subplots(figsize=(13, 5))
for K in STRIKES:
    s = iv_rows[iv_rows["strike"] == K].sort_values(["day", "timestamp"])
    if s["iv"].notna().sum() == 0:
        continue
    x = s["timestamp"].to_numpy() + s["day"].to_numpy() * helpers.TIMESTAMPS_PER_DAY
    ax.plot(x, s["iv"].to_numpy(), label=f"VEV_{K}", lw=0.6)
ax.set_title("Implied volatility time series per strike")
ax.legend(ncol=5, fontsize=8, loc="upper right")
plt.tight_layout()
plt.show()""")

py("""SNAPSHOTS = [
    (0, 0),
    (1, helpers.TIMESTAMPS_PER_DAY - 1),
    (2, helpers.TIMESTAMPS_PER_DAY - 1),
]

def nearest_snapshot(day, ts, window=1000):
    mask = (iv_rows["day"] == day) & (iv_rows["timestamp"].between(ts - window, ts + window))
    return iv_rows[mask]

fig, ax = plt.subplots(figsize=(11, 5))
for (d, ts), color in zip(SNAPSHOTS, sns.color_palette("tab10")):
    snap = nearest_snapshot(d, ts).dropna(subset=["iv"])
    if snap.empty:
        continue
    smile = snap.groupby("strike")["iv"].mean().reset_index()
    ax.plot(smile["strike"], smile["iv"], marker="o",
            label=f"day {d}, ts≈{ts}", color=color)
ax.set_xlabel("strike")
ax.set_ylabel("implied vol")
ax.set_title("Volatility smile at 3 snapshots")
ax.legend()
plt.tight_layout()
plt.show()""")

py("""fig, ax = plt.subplots(figsize=(12, 5))
stats = (iv_rows.dropna(subset=["iv"])
                .groupby("strike")["iv"]
                .agg(["mean", "std", "min", "max", "count"])
                .reset_index())
display(stats)
for K in STRIKES:
    s = iv_rows[(iv_rows["strike"] == K) & iv_rows["iv"].notna()]
    if len(s) < 10:
        continue
    ax.hist(s["iv"], bins=40, alpha=0.3, label=f"VEV_{K}")
ax.set_title("IV distribution per strike")
ax.legend(ncol=5, fontsize=8)
plt.tight_layout()
plt.show()""")

py("""# Rich/cheap: iv - median_iv_across_strikes at the same (day, timestamp)
piv = (iv_rows.dropna(subset=["iv"])
              .pivot_table(index=["day", "timestamp"], columns="strike", values="iv"))
median_iv = piv.median(axis=1)
rich_cheap = piv.subtract(median_iv, axis=0)
rc_summary = rich_cheap.mean().rename("mean_iv_vs_median").to_frame()
rc_summary["abs"] = rc_summary["mean_iv_vs_median"].abs()
rc_summary = rc_summary.sort_values("abs", ascending=False).drop(columns="abs")
display(rc_summary)""")
```

- [ ] **Step 2: Regenerate and execute**

```bash
cd "imc_prosperity4/Round3/Analysis" && \
python3 build_options_notebook.py && \
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 --inplace analysis_options.ipynb
```

Expected: `wrote analysis_options.ipynb with 23 cells`; nbconvert succeeds.

- [ ] **Step 3: Spot-check the finite-IV fraction**

```bash
cd "imc_prosperity4/Round3/Analysis" && \
python3 -c "
import nbformat, json
nb = nbformat.read('analysis_options.ipynb', as_version=4)
for c in nb.cells:
    if c.cell_type == 'code':
        for out in c.get('outputs', []):
            txt = out.get('text') or out.get('data', {}).get('text/plain', '')
            if 'finite IV fraction' in (txt if isinstance(txt, str) else ''.join(txt)):
                print(txt if isinstance(txt, str) else ''.join(txt))"
```

Expected: a line like `finite IV fraction: 0.NNN` with value ≥ 0.80. If it is below 0.80, investigate (probably an intrinsic-floor issue, bad underlying alignment, or too-narrow brent brackets).

- [ ] **Step 4: Commit**

```bash
cd "imc_prosperity4" && \
git add Round3/Analysis/build_options_notebook.py Round3/Analysis/analysis_options.ipynb && \
git commit -m "$(cat <<'EOF'
Round 3 options notebook: implied volatility core

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: `analysis_options.ipynb` — Part E (exports + findings)

**Files:**
- Modify: `imc_prosperity4/Round3/Analysis/build_options_notebook.py`
- Modify: `imc_prosperity4/Round3/Analysis/analysis_options.ipynb` (regenerated)

- [ ] **Step 1: Append Part E cells**

In `build_options_notebook.py`, insert before the emit loop:

```python
# ---------- Part E: Exports + findings ----------
md("## E. Exports")

py("""iv_export = iv_rows[["day", "timestamp", "strike", "mid_price",
                      "underlying_mid", "moneyness", "tte_years", "iv"]].copy()
iv_export = iv_export.rename(columns={"mid_price": "voucher_mid"})
iv_export.to_csv(os.path.join(OUTPUT_DIR, "iv_timeseries.csv"), index=False)
print("iv_timeseries.csv  rows:", len(iv_export))""")

py("""summary_rows = []
for K in STRIKES:
    s = iv_rows[iv_rows["strike"] == K]
    s_raw = v[v["strike"] == K]
    summary_rows.append({
        "strike": K,
        "n_rows": len(s_raw),
        "mean_spread": (s_raw["ask_price_1"] - s_raw["bid_price_1"]).mean(),
        "mean_quoted_size_bid": s_raw["bid_volume_1"].fillna(0).mean(),
        "mean_quoted_size_ask": s_raw["ask_volume_1"].fillna(0).mean(),
        "mean_iv": s["iv"].mean(),
        "iv_std": s["iv"].std(),
        "empirical_delta": slopes.get(K, float("nan")),
        "mean_moneyness": s_raw["moneyness"].mean(),
    })
voucher_summary = pd.DataFrame(summary_rows)
voucher_summary.to_csv(os.path.join(OUTPUT_DIR, "voucher_price_summary.csv"), index=False)
display(voucher_summary)""")

py("""# Per-strike OLS slope, R^2, rolling correlation stats vs underlying
rows = []
ROLL = 500
for K in STRIKES:
    s = v[v["strike"] == K].sort_values(["day", "timestamp"]).copy()
    if len(s) < 50:
        continue
    s["voucher_ret"] = s.groupby("day")["mid_price"].diff()
    s["underlying_ret"] = s.groupby("day")["underlying_mid"].diff()
    rc = (s[["voucher_ret", "underlying_ret"]]
            .rolling(ROLL).corr().unstack()["voucher_ret"]["underlying_ret"])
    ols_slope, ols_intercept = np.polyfit(
        s["underlying_mid"].dropna().to_numpy(float),
        s["mid_price"].dropna().to_numpy(float), 1,
    ) if s[["underlying_mid", "mid_price"]].dropna().shape[0] > 1 else (float("nan"), float("nan"))
    preds = ols_slope * s["underlying_mid"] + ols_intercept
    ss_res = float(((s["mid_price"] - preds) ** 2).sum())
    ss_tot = float(((s["mid_price"] - s["mid_price"].mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rows.append({
        "strike": K,
        "ols_slope": ols_slope,
        "r_squared": r2,
        "rolling_corr_mean": rc.mean(),
        "rolling_corr_std": rc.std(),
    })
corr_summary = pd.DataFrame(rows)
corr_summary.to_csv(os.path.join(OUTPUT_DIR, "voucher_underlying_corr.csv"), index=False)
display(corr_summary)""")

md("""### Findings

_Fill in after running. The notebook is set up to answer:_

- Which strikes are most/least liquid (see `voucher_price_summary.csv`).
- Smile regime: classic smile, skew, or flat (see Part D smile plot).
- Which strikes are persistently rich or cheap relative to the cross-strike
  median IV (see Part D.6 table).
- Empirical delta vs strike monotonicity check (Part C.2 scatters; deep-ITM
  should approach 1).
- Any intrinsic-floor violations (Part C.4).""")
```

- [ ] **Step 2: Regenerate and execute**

```bash
cd "imc_prosperity4/Round3/Analysis" && \
python3 build_options_notebook.py && \
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 --inplace analysis_options.ipynb
```

Expected: `wrote analysis_options.ipynb with 28 cells`; nbconvert succeeds.

- [ ] **Step 3: Confirm exports**

```bash
cd "imc_prosperity4/Round3/Analysis" && ls -la output/
```

Expected: 6 CSVs present — three from delta-1 (Task 9) plus `iv_timeseries.csv`, `voucher_price_summary.csv`, `voucher_underlying_corr.csv`.

- [ ] **Step 4: Commit**

```bash
cd "imc_prosperity4" && \
git add Round3/Analysis/build_options_notebook.py Round3/Analysis/analysis_options.ipynb Round3/Analysis/output/ && \
git commit -m "$(cat <<'EOF'
Round 3 options notebook: exports and findings section

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: End-to-end verification

**Files:**
- None modified (verification only)

- [ ] **Step 1: Full test suite**

```bash
cd "imc_prosperity4/Round3/Analysis" && python3 -m pytest test_helpers.py -v
```

Expected: 30 tests passed.

- [ ] **Step 2: Fresh-kernel re-run of both notebooks**

```bash
cd "imc_prosperity4/Round3/Analysis" && \
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 --inplace analysis_delta1.ipynb && \
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 --inplace analysis_options.ipynb
```

Expected: both finish without error.

- [ ] **Step 3: Artifact inventory**

```bash
cd "imc_prosperity4/Round3/Analysis" && \
for f in delta1_feature_summary.csv delta1_predictor_ranking.csv \
         velvetfruit_realized_vol.csv iv_timeseries.csv \
         voucher_price_summary.csv voucher_underlying_corr.csv; do
  if [ -f "output/$f" ]; then
    bytes=$(wc -c < "output/$f")
    rows=$(wc -l < "output/$f")
    echo "$f  ${bytes} bytes  ${rows} lines"
  else
    echo "MISSING: output/$f"
  fi
done
```

Expected: all 6 present, each ≥ 2 lines (header + data), no MISSING entries.

- [ ] **Step 4: Sanity checks from DESIGN.md**

```bash
cd "imc_prosperity4/Round3/Analysis" && python3 -c "
import pandas as pd
vp = pd.read_csv('output/voucher_price_summary.csv')
iv = pd.read_csv('output/iv_timeseries.csv')

# deep-ITM strike (4000) empirical delta close to 1
delta_4000 = vp[vp['strike'] == 4000]['empirical_delta'].iloc[0]
print(f'empirical delta VEV_4000: {delta_4000:.4f}  (expect ~1.0)')
assert delta_4000 > 0.85, f'deep-ITM delta too low: {delta_4000}'

# IV finite fraction
finite = iv['iv'].notna().mean()
print(f'iv_timeseries finite fraction: {finite:.3f}  (expect >= 0.80)')
assert finite >= 0.80, f'too many NaN IVs: {finite}'

# smile not flat: max-min mean_iv across strikes > small threshold
iv_range = vp['mean_iv'].max() - vp['mean_iv'].min()
print(f'mean_iv range across strikes: {iv_range:.4f}  (expect > 0.01)')
assert iv_range > 0.01, f'smile suspiciously flat: {iv_range}'
print('OK')"
```

Expected: all three asserts pass, final line `OK`.

- [ ] **Step 5: Final git status + commit if clean**

```bash
cd "imc_prosperity4" && git status && git log --oneline -20
```

Expected: `nothing to commit, working tree clean` plus a visible chain of ~12-13 Round-3-analysis commits since the DESIGN.md commit.

- [ ] **Step 6: Report back to user with a summary**

Summarize to the user in one short paragraph: 29 tests passing, both notebooks executing end-to-end, 6 CSV artifacts written, the sanity checks that passed, and any findings worth flagging (e.g., smile shape, liquidity outliers). Do not push to origin unless the user asks.

---

## Self-review checklist

- **Spec coverage:**
  - File layout matches DESIGN.md: Tasks 1-2 create helpers.py + tests; Tasks 6-7 port book/trade feature engineering; Tasks 8-9 build analysis_delta1.ipynb (Parts A-E); Tasks 10-13 build analysis_options.ipynb (Parts A-E); Task 14 verifies. ✓
  - All 6 artifact CSVs are produced: `delta1_feature_summary.csv` (Task 9), `delta1_predictor_ranking.csv` (Task 9), `velvetfruit_realized_vol.csv` (Task 9), `iv_timeseries.csv` (Task 13), `voucher_price_summary.csv` (Task 13), `voucher_underlying_corr.csv` (Task 13). ✓
  - Conventions from DESIGN.md: r=0 (defaulted in BS functions), annualized by 365 (YEAR_DAYS constant), TTE schedule {0:8, 1:7, 2:6}. ✓
  - Explicit non-goals respected: no greeks time series, no cross-strike arb checks, no IV-vs-RV scoring, no detrending layer in delta-1. ✓

- **Placeholder scan:**
  - "Findings" markdown cell in Task 13 is intentionally left for the human to fill in after running — not a plan placeholder, it's the notebook's output section.
  - Otherwise no TBD / TODO / "add appropriate X" / "similar to above" patterns.

- **Type consistency:**
  - `build_book_features`, `build_trade_features`, `safe_corr`, `implied_vol_call` signatures referenced across tasks match their definitions.
  - Artifact column lists in DESIGN.md's table match the CSV writes in Tasks 9 and 13.
  - `HORIZON` is defined once in Task 6 and used in Task 9's predictor target columns.
