import math
import os
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
    assert helpers.bs_call_vega(100, 100, 1.0, 0.20) == pytest.approx(39.695, abs=1e-2)


def test_bs_call_gamma_atm_reference():
    assert helpers.bs_call_gamma(100, 100, 1.0, 0.20) == pytest.approx(0.01988, abs=1e-4)


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
