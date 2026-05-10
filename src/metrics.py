"""Shared regression metrics for ML, DL, and baselines."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def reg_metrics(y_true, y_pred) -> dict:
    return {
        "test_mae_return": float(mean_absolute_error(y_true, y_pred)),
        "test_rmse_return": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
