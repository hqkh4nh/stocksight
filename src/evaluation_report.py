"""Extended evaluation: regression + DL metrics beyond MAE/RMSE.

Goal: produce backtest-ready signals (directional accuracy from regressor sign),
and richer reporting (residual stats, per-horizon DL MAE, val vs test gap).
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score)


def extended_reg_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                         naive_zero_mae: float | None = None) -> dict:
    """Regression metrics for ML 1-day returns.

    Returns
    -------
    dict with keys:
      test_mae_return, test_rmse_return,
      mae_vs_naive_ratio (None if naive_zero_mae is None or 0),
      r2_score, directional_accuracy,
      residual_std, residual_skew, residual_kurt
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    dir_acc = float((np.sign(y_pred) == np.sign(y_true)).mean())

    residuals = y_true - y_pred
    res_std = float(residuals.std())
    res_skew = float(stats.skew(residuals)) if residuals.size > 2 else 0.0
    res_kurt = float(stats.kurtosis(residuals)) if residuals.size > 3 else 0.0

    ratio = None
    if naive_zero_mae is not None and naive_zero_mae > 0:
        ratio = float(mae / naive_zero_mae)

    return {
        "test_mae_return": mae,
        "test_rmse_return": rmse,
        "mae_vs_naive_ratio": ratio,
        "r2_score": r2,
        "directional_accuracy": dir_acc,
        "residual_std": res_std,
        "residual_skew": res_skew,
        "residual_kurt": res_kurt,
    }


def per_horizon_price_mae(true_prices: np.ndarray, pred_prices: np.ndarray) -> dict:
    """MAE/RMSE at horizon day 1, 7, 14 (1-indexed)."""
    out = {}
    for d in (1, 7, 14):
        idx = d - 1
        if idx >= true_prices.shape[1]:
            continue
        mae = float(mean_absolute_error(true_prices[:, idx], pred_prices[:, idx]))
        rmse = float(np.sqrt(mean_squared_error(true_prices[:, idx], pred_prices[:, idx])))
        out[f"test_mae_price_d{d}"] = mae
        out[f"test_rmse_price_d{d}"] = rmse
    return out


def dl_extended_metrics(model, dl, history) -> dict:
    """Full DL evaluation: aggregate price MAE/RMSE/MAPE, per-horizon, directional, and
    train/val loss snapshot from history.history.
    """
    pred_returns = model.predict(dl.X_test_seq, verbose=0)            # (N, 14, 1)
    true_returns = dl.y_return_seq_test                               # (N, 14, 1)

    close_anchor = dl.close_anchor_test                               # (N,)
    pred_prices = _returns_to_prices(close_anchor, pred_returns[..., 0])
    true_prices = _returns_to_prices(close_anchor, true_returns[..., 0])

    mae = float(mean_absolute_error(true_prices.ravel(), pred_prices.ravel()))
    rmse = float(np.sqrt(mean_squared_error(true_prices.ravel(), pred_prices.ravel())))
    mape = float(np.mean(np.abs((true_prices - pred_prices) / true_prices)) * 100)

    pred_dir = (pred_returns[..., 0] > 0).astype(int).ravel()
    true_dir = (true_returns[..., 0] > 0).astype(int).ravel()
    dir_acc = float((pred_dir == true_dir).mean())

    tp = int(((pred_dir == 1) & (true_dir == 1)).sum())
    fp = int(((pred_dir == 1) & (true_dir == 0)).sum())
    fn = int(((pred_dir == 0) & (true_dir == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    cum_pred = np.prod(1 + pred_returns[..., 0], axis=1) - 1
    cum_true = np.prod(1 + true_returns[..., 0], axis=1) - 1
    long_dir_acc = float(((cum_pred > 0) == (cum_true > 0)).mean())

    out = {
        "test_mae_price_14d": mae,
        "test_rmse_price_14d": rmse,
        "test_mape_price_14d": mape,
        "test_dir_accuracy_perday": dir_acc,
        "test_dir_f1_perday": float(f1),
        "test_dir_accuracy_t14": long_dir_acc,
    }
    out.update(per_horizon_price_mae(true_prices, pred_prices))

    hist = history.history if history is not None else {}
    out["train_loss_final"] = float(hist["loss"][-1]) if hist.get("loss") else None
    out["val_loss_final"] = float(hist["val_loss"][-1]) if hist.get("val_loss") else None
    out["train_mae_final"] = float(hist["mae"][-1]) if hist.get("mae") else None
    out["val_mae_final"] = float(hist["val_mae"][-1]) if hist.get("val_mae") else None

    return out


def dl_test_predictions_long(model, dl) -> dict:
    """Predict on test set and return long-format arrays for parquet persistence.

    Returns dict with arrays of length N*horizon:
      anchor_date, target_date, horizon_step (1..14), close_anchor,
      y_true_return, y_pred_return, y_true_price, y_pred_price.
    """
    pred_returns = model.predict(dl.X_test_seq, verbose=0)[..., 0]    # (N, 14)
    true_returns = dl.y_return_seq_test[..., 0]                       # (N, 14)
    close_anchor = dl.close_anchor_test                               # (N,)

    pred_prices = _returns_to_prices(close_anchor, pred_returns)
    true_prices = _returns_to_prices(close_anchor, true_returns)

    n, h = pred_returns.shape
    anchor_dates = np.repeat(dl.test_anchor_dates.values, h)
    horizon_steps = np.tile(np.arange(1, h + 1), n)
    close_anchor_rep = np.repeat(close_anchor, h)

    return {
        "anchor_date": anchor_dates,
        "horizon_step": horizon_steps,
        "close_anchor": close_anchor_rep,
        "y_true_return": true_returns.ravel(),
        "y_pred_return": pred_returns.ravel(),
        "y_true_price": true_prices.ravel(),
        "y_pred_price": pred_prices.ravel(),
    }


def _returns_to_prices(close_anchor: np.ndarray, returns_seq: np.ndarray) -> np.ndarray:
    """Convert (N, H) returns to (N, H) prices given anchor close[N]."""
    cum = np.cumprod(1 + returns_seq, axis=1)
    return close_anchor[:, None] * cum
