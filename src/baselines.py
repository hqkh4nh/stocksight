"""Naive baselines used as a sanity floor for ML/DL models.

A model that beats a naive baseline by a meaningful margin actually learns; a model
that loses to it has no business being shipped to a defence committee.

Baselines:
- naive_zero        : predict 0 return tomorrow (drift-free random walk)
- naive_persistence : predict tomorrow == today's return (momentum-of-1)
- majority_class    : predict the most common direction in train (drift-aware)
"""
import numpy as np

from src.metrics import clf_metrics, reg_metrics
from src.preprocessing import SplitData


def baseline_zero(split: SplitData) -> dict:
    """Predict 0 return for every test day."""
    y_pred = np.zeros_like(split.y_return_test)
    return reg_metrics(split.y_return_test, y_pred)


def baseline_persistence(split: SplitData) -> dict:
    """Predict tomorrow's return == today's return.

    Build today's return for each test row from the close series. The first test row
    uses the last training return as its 'today'.
    """
    last_train_return = float(split.y_return_train[-1])
    today_returns = np.concatenate([[last_train_return], split.y_return_test[:-1]])
    return reg_metrics(split.y_return_test, today_returns)


def baseline_majority_class(split: SplitData) -> dict:
    """Predict the most common direction observed in training."""
    majority = int(np.bincount(split.y_direction_train).argmax())
    y_pred = np.full_like(split.y_direction_test, fill_value=majority)
    y_proba = np.full(split.y_direction_test.shape, fill_value=float(majority), dtype=float)
    return clf_metrics(split.y_direction_test, y_pred, y_proba)


def all_baselines(split: SplitData) -> dict:
    return {
        "naive_zero": baseline_zero(split),
        "naive_persistence": baseline_persistence(split),
        "majority": baseline_majority_class(split),
    }
