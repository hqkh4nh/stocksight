"""Offline pre-compute per-horizon MAE for ML regressors.

For each (ticker, model in [ridge, rf_reg, xgb_reg], horizon in 1..14):
  - For each anchor index i in the test window,
    run recursive_forecast_14 from features[i],
    compare predicted return path against realized returns[i+1..i+14].
  - Aggregate MAE on the predicted PRICE path (close * cumprod(1 + rets))
    vs the realized close path, to match the units in dl_summary.

Output: results/per_horizon_ml.csv with columns (ticker, model, horizon, mae_price).
"""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd

from src.config import CFG, MODELS_DIR, RESULTS_DIR
from src.features import compute_features, feature_columns
from src.models.ml_model import recursive_forecast_14
from src.preprocessing import build_macro_df
from src.data_loader import load_stock
from src.targets import make_targets


def per_horizon_for_ticker(ticker: str, macro_df: pd.DataFrame,
                            horizon: int = 14) -> list[dict]:
    stock = load_stock(ticker)
    feats = compute_features(stock, macro_df, ticker=ticker)
    feats = make_targets(feats).dropna().reset_index(drop=True)
    feat_cols = feature_columns(ticker)

    n = len(feats)
    train_end = int(n * CFG["split"]["train_ratio"])
    test_idx = np.arange(train_end, n - horizon)

    scaler = joblib.load(MODELS_DIR / f"{ticker}_scaler_X.joblib")
    closes = feats["close"].values

    rows = []
    for model_name in ["ridge", "rf_reg", "xgb_reg"]:
        model = joblib.load(MODELS_DIR / f"{ticker}_{model_name}.pkl")
        h_errors = {h: [] for h in range(1, horizon + 1)}
        for i in test_idx:
            x0 = feats[feat_cols].iloc[i].values.astype(float)
            rets = recursive_forecast_14(model, x0, feat_cols=feat_cols,
                                          scaler_X=scaler, horizon=horizon)
            anchor = closes[i]
            pred_path = anchor * np.cumprod(1.0 + rets)
            real_path = closes[i + 1: i + 1 + horizon]
            for h in range(1, horizon + 1):
                h_errors[h].append(abs(pred_path[h - 1] - real_path[h - 1]))
        for h, errs in h_errors.items():
            rows.append({"ticker": ticker, "model": model_name,
                          "horizon": h,
                          "mae_price": float(np.mean(errs)) if errs else float("nan"),
                          "n": len(errs)})
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
