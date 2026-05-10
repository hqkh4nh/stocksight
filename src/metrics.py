"""Shared regression + classification metrics for ML, DL, and baselines."""
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, roc_auc_score)


def reg_metrics(y_true, y_pred) -> dict:
    return {
        "test_mae_return": float(mean_absolute_error(y_true, y_pred)),
        "test_rmse_return": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def clf_metrics(y_true, y_pred, y_proba=None) -> dict:
    out = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        out["test_roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return out
