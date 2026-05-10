import numpy as np
import pandas as pd
import pytest

from src.preprocessing import prepare_dl_pipeline, build_macro_df


@pytest.fixture(scope="module")
def aapl_pipeline():
    # Use BKR (already cached by Task 2.2 smoke run)
    macro_df = build_macro_df()
    return prepare_dl_pipeline("BKR", macro_df, window=60, horizon=14, train_ratio=0.85)


def test_window_horizon_shapes(aapl_pipeline):
    dl, split = aapl_pipeline
    assert dl.X_train_seq.shape[1] == 60
    assert dl.y_return_seq_train.shape[1] == 14
    assert dl.y_return_seq_train.shape[2] == 1


def test_chronological_no_overlap(aapl_pipeline):
    dl, split = aapl_pipeline
    # last train target date < first test target date
    assert dl.train_target_dates.iloc[-1] < dl.test_target_dates.iloc[0]


def test_scaler_fit_train_only(aapl_pipeline):
    dl, split = aapl_pipeline
    # X_train should be in [0, 1] approximately (MinMaxScaler fitted on train)
    assert 0.0 - 1e-6 <= dl.X_train_seq.min() <= dl.X_train_seq.max() <= 1.0 + 1e-6
    # X_test may exceed [0,1] (regime shift); just check it's not all zeros
    assert dl.X_test_seq.std() > 0


def test_85_15_split_ratio(aapl_pipeline):
    """ML train+val should be ~85% of (train+val+test); val is ~10% of train+val."""
    dl, split = aapl_pipeline
    n_train = len(split.X_train)
    n_val = len(split.X_val)
    n_test = len(split.X_test)
    train_plus_val = n_train + n_val
    ratio = train_plus_val / (train_plus_val + n_test)
    assert 0.83 < ratio < 0.87
    val_ratio = n_val / train_plus_val
    assert 0.08 < val_ratio < 0.12


def test_dl_targets_are_raw_returns(aapl_pipeline):
    """Targets must remain raw returns (negatives included), not scaled into [0,1]."""
    dl, _ = aapl_pipeline
    y_train = dl.y_return_seq_train
    assert (y_train < 0).any(), "Train targets must contain negative returns"
    assert (y_train > 0).any(), "Train targets must contain positive returns"
    assert abs(y_train).max() < 1.0, "Returns should be small (<100% per day)"


def test_dl_data_no_scaler_y(aapl_pipeline):
    """DLData must not carry scaler_y once raw returns are used."""
    dl, _ = aapl_pipeline
    assert not hasattr(dl, "scaler_y") or dl.scaler_y is None
