"""Offline pre-compute per-horizon MAE for ML regressors (batched / vectorized).

For each (ticker, model in [ridge, rf_reg, xgb_reg]), all test anchors are
forecasted simultaneously. At each horizon step we run a single
`model.predict(X)` on the full (n_anchors, n_features) batch, then update
lag features in-place before stepping to the next horizon. This collapses
~n_anchors x horizon model.predict calls into `horizon` calls.

Output: results/per_horizon_ml.csv with columns (ticker, model, horizon, mae_price, n).
"""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd

from src.config import CFG, MODELS_DIR, RESULTS_DIR
from src.data_loader import load_stock
from src.features import compute_features, feature_columns
from src.preprocessing import build_macro_df
from src.targets import make_targets


def _batch_recursive_forecast(model, X0: np.ndarray, feat_cols: list[str],
                              scaler, horizon: int) -> np.ndarray:
    """Batched version of recursive_forecast_14.

    X0 shape: (n_anchors, n_features) — unscaled feature rows for each anchor.
    Returns: (n_anchors, horizon) array of predicted returns.
    """
    X = X0.astype(float).copy()
    idx = {c: i for i, c in enumerate(feat_cols)}
    n = X.shape[0]

    has_returns = "returns" in idx
    has_log = "log_returns" in idx
    has_lag1 = "close_pct_lag_1" in idx
    has_lag5 = "close_pct_lag_5" in idx

    recent = np.zeros((n, 5), dtype=float)
    if has_lag1:
        recent[:] = X[:, idx["close_pct_lag_1"]][:, None]

    preds = np.zeros((n, horizon), dtype=float)
    for h in range(horizon):
        Xs = scaler.transform(X) if scaler is not None else X
        r = model.predict(Xs)
        preds[:, h] = r

        recent = np.concatenate([recent[:, 1:], r[:, None]], axis=1)
        if has_returns:
            X[:, idx["returns"]] = r
        if has_log:
            X[:, idx["log_returns"]] = np.log1p(r)
        if has_lag1:
            X[:, idx["close_pct_lag_1"]] = r
        if has_lag5:
            X[:, idx["close_pct_lag_5"]] = np.prod(1.0 + recent, axis=1) - 1.0

    return preds


def per_horizon_for_ticker(ticker: str, macro_df: pd.DataFrame,
                            horizon: int = 14) -> list[dict]:
    stock = load_stock(ticker)
    feats = compute_features(stock, macro_df, ticker=ticker)
    feats = make_targets(feats).dropna().reset_index(drop=True)
    feat_cols = feature_columns(ticker)

    n = len(feats)
    train_end = int(n * CFG["split"]["train_ratio"])
    test_idx = np.arange(train_end, n - horizon)
    if len(test_idx) == 0:
        return []

    scaler = joblib.load(MODELS_DIR / f"{ticker}_scaler_X.joblib")
    closes = feats["close"].values
    X0 = feats[feat_cols].iloc[test_idx].values.astype(float)
    anchors = closes[test_idx]

    real_paths = np.stack([closes[i + 1: i + 1 + horizon] for i in test_idx])

    rows = []
    for model_name in ["ridge", "rf_reg", "xgb_reg"]:
        model = joblib.load(MODELS_DIR / f"{ticker}_{model_name}.pkl")
        rets = _batch_recursive_forecast(model, X0, feat_cols, scaler, horizon)
        pred_paths = anchors[:, None] * np.cumprod(1.0 + rets, axis=1)

        abs_err = np.abs(pred_paths - real_paths)
        for h in range(horizon):
            rows.append({
                "ticker": ticker,
                "model": model_name,
                "horizon": h + 1,
                "mae_price": float(abs_err[:, h].mean()),
                "n": int(len(test_idx)),
            })
    return rows


def main():
    macro_df = build_macro_df()
    tickers = CFG["tickers"]
    all_rows: list[dict] = []
    t0 = time.time()
    for i, t in enumerate(tickers, 1):
        elapsed = time.time() - t0
        print(f"[{i}/{len(tickers)}] {t}  ({elapsed:.0f}s elapsed)", flush=True)
        try:
            all_rows.extend(per_horizon_for_ticker(t, macro_df))
        except FileNotFoundError as e:
            print(f"  skipped: {e}")
    out = RESULTS_DIR / "per_horizon_ml.csv"
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"\nSaved: {out}  ({len(all_rows)} rows, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
