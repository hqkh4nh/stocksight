import numpy as np

from src.models.ml_model import recursive_forecast_14


class _DummyModel:
    def predict(self, X):
        return np.array([0.01])  # always predict +1% return


def test_recursive_returns_horizon_length():
    out = recursive_forecast_14(_DummyModel(), np.zeros(10), horizon=14)
    assert out.shape == (14,)
    assert np.allclose(out, 0.01)


def test_recursive_custom_horizon():
    out = recursive_forecast_14(_DummyModel(), np.zeros(5), horizon=7)
    assert out.shape == (7,)
