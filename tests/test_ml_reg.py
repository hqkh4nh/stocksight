"""Regression test: Ridge/RF/XGB regressors return (model, y_pred_test, y_pred_val)."""
from src.models.ml_model import train_rf_reg, train_ridge, train_xgb_reg


def test_train_ridge_returns_predictions(tiny_split):
    model, y_test, y_val = train_ridge(tiny_split)
    assert y_test.shape == tiny_split.y_return_test.shape
    assert y_val.shape == tiny_split.y_return_val.shape


def test_train_rf_reg_returns_predictions(tiny_split):
    model, y_test, y_val = train_rf_reg(tiny_split)
    assert y_test.shape == tiny_split.y_return_test.shape
    assert y_val.shape == tiny_split.y_return_val.shape


def test_train_xgb_reg_returns_predictions(tiny_split):
    model, y_test, y_val = train_xgb_reg(tiny_split)
    assert y_test.shape == tiny_split.y_return_test.shape
    assert y_val.shape == tiny_split.y_return_val.shape
