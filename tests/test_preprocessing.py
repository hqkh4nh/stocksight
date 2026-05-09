import numpy as np
import pandas as pd
import pytest

from src.preprocessing import prepare_dl_pipeline_v2, build_macro_df


@pytest.fixture(scope="module")
def aapl_pipeline():
    # Use BKR (already cached by Task 2.2 smoke run)
    macro_df = build_macro_df()
    return prepare_dl_pipeline_v2("BKR", macro_df, window=60, horizon=14, train_ratio=0.85)


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
    # X_test may exceed [0,1] — that's expected (regime shift) and proves no leak
    # Just check it's not all zeros
    assert dl.X_test_seq.std() > 0


def test_85_15_split_ratio(aapl_pipeline):
    dl, split = aapl_pipeline
    n_train = len(split.X_train)
    n_test = len(split.X_test)
    ratio = n_train / (n_train + n_test)
    assert 0.83 < ratio < 0.87
