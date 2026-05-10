"""Tests for naive regression baselines."""
import numpy as np
import pytest

from src.baselines import baseline_persistence, baseline_zero


def test_zero_mae_equals_mean_abs_return(tiny_split):
    m = baseline_zero(tiny_split)
    expected_mae = float(np.mean(np.abs(tiny_split.y_return_test)))
    assert m["test_mae_return"] == pytest.approx(expected_mae, rel=1e-6)


def test_persistence_returns_metrics(tiny_split):
    m = baseline_persistence(tiny_split)
    assert "test_mae_return" in m
    assert "test_rmse_return" in m
    assert m["test_mae_return"] > 0
