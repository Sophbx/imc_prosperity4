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
