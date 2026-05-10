import numpy as np

from src.models.ml_model import recursive_forecast_14


class _DummyModel:
    def predict(self, X):
        return np.array([0.01])  # always predict +1% return


class _LagSensitiveModel:
    """Predicts 0.5 * close_pct_lag_1 - output depends on prior step's update."""
    def __init__(self, lag_idx):
        self.lag_idx = lag_idx

    def predict(self, X):
        return np.array([0.5 * X[0, self.lag_idx]])


def test_recursive_returns_horizon_length():
    out = recursive_forecast_14(_DummyModel(), np.zeros(10), horizon=14)
    assert out.shape == (14,)
    assert np.allclose(out, 0.01)


def test_recursive_custom_horizon():
    out = recursive_forecast_14(_DummyModel(), np.zeros(5), horizon=7)
    assert out.shape == (7,)


def test_recursive_updates_lag_features():
    """Regression: feature row must mutate between steps, not stay constant."""
    feat_cols = ["returns", "log_returns", "close_pct_lag_1", "close_pct_lag_5", "rsi_14"]
    last_row = np.array([0.02, 0.0198, 0.02, 0.10, 50.0])
    lag_idx = feat_cols.index("close_pct_lag_1")
    model = _LagSensitiveModel(lag_idx=lag_idx)

    out = recursive_forecast_14(model, last_row, feat_cols=feat_cols, horizon=5)
    # Each step halves the prior pred: 0.5 * 0.02 = 0.01, 0.5 * 0.01 = 0.005, ...
    expected = [0.01, 0.005, 0.0025, 0.00125, 0.000625]
    assert np.allclose(out, expected, atol=1e-9), f"got {out}, want {expected}"
    # Predictions must NOT all be equal
    assert len(set(out.round(6))) == 5
