"""Regression test: RF/XGB regression trainers return MAE/RMSE keys."""
import numpy as np
import pytest

from src.models.ml_model import train_rf_reg, train_xgb_reg
from src.preprocessing import SplitData
from sklearn.preprocessing import MinMaxScaler


@pytest.fixture
def tiny_split():
    rng = np.random.default_rng(0)
    n_train, n_test, n_feat = 200, 50, 8
    X_train = rng.standard_normal((n_train, n_feat)).astype(np.float32)
    X_test = rng.standard_normal((n_test, n_feat)).astype(np.float32)
    y_train = rng.standard_normal(n_train).astype(np.float32) * 0.02
    y_test = rng.standard_normal(n_test).astype(np.float32) * 0.02
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


def test_train_rf_reg_returns_metrics(tiny_split):
    model, y_pred, m = train_rf_reg(tiny_split)
    assert set(m.keys()) == {"test_mae_return", "test_rmse_return"}
    assert m["test_mae_return"] > 0
    assert y_pred.shape == (50,)


def test_train_xgb_reg_returns_metrics(tiny_split):
    model, y_pred, m = train_xgb_reg(tiny_split)
    assert set(m.keys()) == {"test_mae_return", "test_rmse_return"}
    assert m["test_mae_return"] > 0
