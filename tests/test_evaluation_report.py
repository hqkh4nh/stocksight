"""Unit tests for src.evaluation_report."""
import numpy as np
import pytest

from src.evaluation_report import (extended_reg_metrics,
                                   per_horizon_price_mae)


def test_extended_reg_metrics_keys():
    rng = np.random.default_rng(0)
    y_true = rng.standard_normal(200) * 0.02
    y_pred = y_true * 0.5 + rng.standard_normal(200) * 0.005
    m = extended_reg_metrics(y_true, y_pred, naive_zero_mae=0.015)
    assert {"test_mae_return", "test_rmse_return", "mae_vs_naive_ratio",
            "r2_score", "directional_accuracy", "residual_std",
            "residual_skew", "residual_kurt"} <= set(m.keys())


def test_mae_vs_naive_ratio_is_none_when_zero_mae_missing():
    y = np.array([0.01, -0.01, 0.005])
    m = extended_reg_metrics(y, y, naive_zero_mae=None)
    assert m["mae_vs_naive_ratio"] is None


def test_directional_accuracy_perfect():
    y_true = np.array([0.01, -0.02, 0.005, -0.001])
    y_pred = np.array([0.05, -0.03, 0.001, -0.0001])
    m = extended_reg_metrics(y_true, y_pred)
    assert m["directional_accuracy"] == pytest.approx(1.0)


def test_directional_accuracy_zero_when_all_signs_flipped():
    y_true = np.array([0.01, -0.02, 0.005])
    y_pred = -y_true
    m = extended_reg_metrics(y_true, y_pred)
    assert m["directional_accuracy"] == pytest.approx(0.0)


def test_per_horizon_price_mae_d1_d7_d14():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(50, 100, (30, 14))
    pred_p = true_p + rng.normal(0, 1, (30, 14))
    out = per_horizon_price_mae(true_p, pred_p)
    assert {"test_mae_price_d1", "test_mae_price_d7", "test_mae_price_d14"} <= set(out.keys())
    for k in out:
        assert out[k] > 0
