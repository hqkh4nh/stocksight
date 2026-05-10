"""Naive regression baselines as sanity floor for ML/DL.

- naive_zero        : predict 0 return tomorrow
- naive_persistence : predict tomorrow == today's return
"""
import numpy as np

from src.metrics import reg_metrics
from src.preprocessing import SplitData


def baseline_zero(split: SplitData) -> dict:
    """Predict 0 return for every test day."""
    y_pred = np.zeros_like(split.y_return_test)
    return reg_metrics(split.y_return_test, y_pred)


def baseline_persistence(split: SplitData) -> dict:
    """Predict tomorrow's return == today's return.

    First test row uses the last training return as its 'today'.
    """
    last_train_return = float(split.y_return_train[-1])
    today_returns = np.concatenate([[last_train_return], split.y_return_test[:-1]])
    return reg_metrics(split.y_return_test, today_returns)


def all_baselines(split: SplitData) -> dict:
    return {
        "naive_zero": baseline_zero(split),
        "naive_persistence": baseline_persistence(split),
    }
