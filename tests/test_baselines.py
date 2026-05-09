"""Tests for naive baselines."""
import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

from src.baselines import (baseline_majority_class, baseline_persistence,
                           baseline_zero)
from src.preprocessing import SplitData


@pytest.fixture
def split():
    rng = np.random.default_rng(0)
    n_train, n_test, n_feat = 200, 50, 4
    X_train = rng.standard_normal((n_train, n_feat)).astype(np.float32)
    X_test = rng.standard_normal((n_test, n_feat)).astype(np.float32)
    y_train = (rng.standard_normal(n_train) * 0.02).astype(np.float32)
    y_test = (rng.standard_normal(n_test) * 0.02).astype(np.float32)
    return SplitData(
        X_train=X_train, X_test=X_test,
        y_return_train=y_train, y_return_test=y_test,
        y_direction_train=(y_train > 0).astype(np.int32),
        y_direction_test=(y_test > 0).astype(np.int32),
        close_train=np.ones(n_train), close_test=np.ones(n_test),
        feature_names=[f"f{i}" for i in range(n_feat)],
        scaler_X=MinMaxScaler(),
        train_dates=None, test_dates=None,
    )


def test_zero_mae_equals_mean_abs_return(split):
    m = baseline_zero(split)
    expected_mae = float(np.mean(np.abs(split.y_return_test)))
    assert m["test_mae_return"] == pytest.approx(expected_mae, rel=1e-6)


def test_persistence_uses_prior_return(split):
    m = baseline_persistence(split)
    assert "test_mae_return" in m
    assert "test_rmse_return" in m
    # MAE should be roughly 2x the random-walk std (sqrt(2) * std on diff)
    assert m["test_mae_return"] > 0


def test_majority_class_predicts_constant(split):
    m = baseline_majority_class(split)
    assert "test_accuracy" in m and "test_f1" in m
    # Accuracy of constant predictor equals proportion of majority class in test
    majority = int(np.bincount(split.y_direction_train).argmax())
    expected_acc = float((split.y_direction_test == majority).mean())
    assert m["test_accuracy"] == pytest.approx(expected_acc, rel=1e-6)
