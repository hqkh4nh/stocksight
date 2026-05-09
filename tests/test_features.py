import numpy as np
import pandas as pd
import pytest

from src.features import compute_features, FEATURE_COLUMNS_BASE


@pytest.fixture
def stock_df():
    n = 200
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    close = 100 + rng.standard_normal(n).cumsum()
    return pd.DataFrame({
        "date": dates,
        "open": close + rng.standard_normal(n) * 0.5,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, size=n),
    })


@pytest.fixture
def macro_df():
    n = 200
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "date": dates,
        "vix_change": rng.standard_normal(n) * 0.05,
        "tnx_change": rng.standard_normal(n) * 0.02,
        "oil_change": rng.standard_normal(n) * 0.03,
        "usd_change": rng.standard_normal(n) * 0.01,
    })


def test_compute_features_for_oil_sector(stock_df, macro_df):
    df = compute_features(stock_df, macro_df, ticker="BKR")
    df = df.dropna()
    # All Group A + Group B + Group C (oil) should be present
    for col in FEATURE_COLUMNS_BASE:
        assert col in df.columns, f"missing {col}"
    for col in ["vix_change", "tnx_change", "oil_change", "usd_change"]:
        assert col in df.columns
    assert "stock_oil_corr_21" in df.columns
    assert "vol_x_oil" in df.columns
    assert "stock_rate_corr_21" not in df.columns


def test_compute_features_for_rate_sector(stock_df, macro_df):
    df = compute_features(stock_df, macro_df, ticker="FITB")
    df = df.dropna()
    assert "stock_rate_corr_21" in df.columns
    assert "stock_oil_corr_21" not in df.columns


def test_compute_features_no_lookahead(stock_df, macro_df):
    df = compute_features(stock_df, macro_df, ticker="BKR")
    # last value of returns should equal last close / second-to-last close - 1
    expected = stock_df["close"].iloc[-1] / stock_df["close"].iloc[-2] - 1
    assert abs(df["returns"].iloc[-1] - expected) < 1e-9
