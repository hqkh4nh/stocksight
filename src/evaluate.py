"""Evaluation: 14-day metrics for DL.

DL outputs are RAW returns (no scaler_y) — convert to prices via
close_anchor x cumprod(1 + returns).
"""
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error)

from src.preprocessing import DLData


def dl_metrics_14d(model, dl: DLData) -> dict:
    """Compute MAE/RMSE/MAPE on price + directional acc/f1 from sign of returns.

    All metrics aggregated over the 14-day horizon (per-day flattened, plus a t+14 long-horizon dir_acc).
    """
    pred_returns = model.predict(dl.X_test_seq, verbose=0)            # (N, 14, 1) raw
    true_returns = dl.y_return_seq_test                               # (N, 14, 1) raw

    close_anchor = dl.close_anchor_test                               # (N,)
    pred_prices = _returns_to_prices(close_anchor, pred_returns[..., 0])
    true_prices = _returns_to_prices(close_anchor, true_returns[..., 0])

    mae = float(mean_absolute_error(true_prices.ravel(), pred_prices.ravel()))
    rmse = float(np.sqrt(mean_squared_error(true_prices.ravel(), pred_prices.ravel())))
    mape = float(np.mean(np.abs((true_prices - pred_prices) / true_prices)) * 100)

    pred_dir = (pred_returns[..., 0] > 0).astype(int).ravel()
    true_dir = (true_returns[..., 0] > 0).astype(int).ravel()
    acc = float(accuracy_score(true_dir, pred_dir))
    f1 = float(f1_score(true_dir, pred_dir, zero_division=0))

    cum_pred = np.prod(1 + pred_returns[..., 0], axis=1) - 1
    cum_true = np.prod(1 + true_returns[..., 0], axis=1) - 1
    long_dir_acc = float(((cum_pred > 0) == (cum_true > 0)).mean())

    return {
        "test_mae_price_14d": mae,
        "test_rmse_price_14d": rmse,
        "test_mape_price_14d": mape,
        "test_dir_accuracy_perday": acc,
        "test_dir_f1_perday": f1,
        "test_dir_accuracy_t14": long_dir_acc,
    }


def _returns_to_prices(close_anchor: np.ndarray, returns_seq: np.ndarray) -> np.ndarray:
    """Convert (N, H) returns to (N, H) prices given anchor close[N]."""
    cum = np.cumprod(1 + returns_seq, axis=1)
    return close_anchor[:, None] * cum
