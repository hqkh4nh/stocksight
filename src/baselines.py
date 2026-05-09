"""Naive baselines used as a sanity floor for ML/DL models.

A model that beats a naive baseline by a meaningful margin actually learns; a model
that loses to it has no business being shipped to a defence committee.

Baselines:
- naive_zero        : predict 0 return tomorrow (drift-free random walk)
- naive_persistence : predict tomorrow == today's return (momentum-of-1)
- majority_class    : predict the most common direction in train (drift-aware)
"""
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, roc_auc_score)

from src.preprocessing import SplitData


def _reg_metrics(y_true, y_pred):
    return {
        "test_mae_return": float(mean_absolute_error(y_true, y_pred)),
        "test_rmse_return": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _clf_metrics(y_true, y_pred, y_proba=None):
    out = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        out["test_roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return out


def baseline_zero(split: SplitData) -> dict:
    """Predict 0 return for every test day."""
    y_pred = np.zeros_like(split.y_return_test)
    return _reg_metrics(split.y_return_test, y_pred)


def baseline_persistence(split: SplitData) -> dict:
    """Predict tomorrow's return == today's return.

    Build today's return for each test row from the close series. The first test row
    uses the last training return as its 'today'.
    """
    last_train_return = float(split.y_return_train[-1])
    today_returns = np.concatenate([[last_train_return], split.y_return_test[:-1]])
    return _reg_metrics(split.y_return_test, today_returns)


def baseline_majority_class(split: SplitData) -> dict:
    """Predict the most common direction observed in training."""
    majority = int(np.bincount(split.y_direction_train).argmax())
    y_pred = np.full_like(split.y_direction_test, fill_value=majority)
    y_proba = np.full(split.y_direction_test.shape, fill_value=float(majority), dtype=float)
    return _clf_metrics(split.y_direction_test, y_pred, y_proba)


def all_baselines(split: SplitData) -> dict:
    return {
        "naive_zero": baseline_zero(split),
        "naive_persistence": baseline_persistence(split),
        "majority": baseline_majority_class(split),
    }
