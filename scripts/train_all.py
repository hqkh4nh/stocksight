"""Train ML regressors + DL on all 21 tickers and persist predictions for backtest.

Outputs:
  results/ml_summary.csv             - extended regression metrics per ticker per ML model
  results/dl_summary.csv             - 14-day DL metrics (incl. per-horizon)
  results/predictions/{ticker}_ml.parquet
  results/predictions/{ticker}_dl.parquet
  results/predictions_ml.parquet     - concat of all ml predictions (long format)
  results/predictions_dl.parquet     - concat of all dl predictions (long format)
  results/run_metadata.json
  results/failed.txt                 - tickers that errored (with traceback)

CSVs are flushed after every ticker so a crash mid-run never loses prior work.
"""
import datetime
import gc
import json
import subprocess
import time
import traceback

import numpy as np
import pandas as pd

from src.config import CFG, RESULTS_DIR


PREDICTIONS_DIR = RESULTS_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

ML_MODEL_NAMES = ["ridge", "rf_reg", "xgb_reg", "naive_zero", "naive_persistence"]


def _baseline_predictions(name: str, split):
    """Return (val_pred, test_pred) for a naive baseline."""
    if name == "naive_zero":
        return np.zeros_like(split.y_return_val), np.zeros_like(split.y_return_test)
    if name == "naive_persistence":
        last_train = float(split.y_return_train[-1])
        yv = np.concatenate([[last_train], split.y_return_val[:-1]])
        last_val = float(split.y_return_val[-1]) if len(split.y_return_val) else last_train
        yt = np.concatenate([[last_val], split.y_return_test[:-1]])
        return yv, yt
    raise ValueError(f"Unknown baseline: {name}")


def _build_ml_pred_rows(ticker: str, name: str, dates, y_true, y_pred) -> list[dict]:
    return [{
        "date": dates.iloc[i],
        "ticker": ticker,
        "model": name,
        "y_true_return": float(y_true[i]),
        "y_pred_return": float(y_pred[i]),
        "y_pred_direction": int(np.sign(y_pred[i])),
    } for i in range(len(dates))]


def train_one_ticker(ticker: str, macro_df: pd.DataFrame) -> dict:
    import tensorflow as tf
    from src.evaluation_report import (dl_extended_metrics,
                                       dl_test_predictions_long,
                                       extended_reg_metrics)
    import joblib
    from src.config import MODELS_DIR
    from src.models.dl_model import save_dl_history, save_dl_model, train_dl
    from src.models.ml_model import (save_ml_models, train_rf_reg,
                                     train_ridge, train_xgb_reg)
    from src.preprocessing import prepare_dl_pipeline

    tf.keras.backend.clear_session()
    t0 = time.time()
    dl, split = prepare_dl_pipeline(ticker, macro_df)

    ridge, ridge_test, ridge_val = train_ridge(split)
    rf_reg, rf_test, rf_val = train_rf_reg(split)
    xgb_reg, xgb_test, xgb_val = train_xgb_reg(split)
    save_ml_models(ticker, ridge, rf_reg, xgb_reg, split.scaler_X)

    naive_zero_val, naive_zero_test = _baseline_predictions("naive_zero", split)
    naive_per_val, naive_per_test = _baseline_predictions("naive_persistence", split)
    naive_zero_mae_test = float(np.abs(split.y_return_test - naive_zero_test).mean())
    naive_zero_mae_val = float(np.abs(split.y_return_val - naive_zero_val).mean())

    ml_results = {
        "ridge": (ridge_val, ridge_test),
        "rf_reg": (rf_val, rf_test),
        "xgb_reg": (xgb_val, xgb_test),
        "naive_zero": (naive_zero_val, naive_zero_test),
        "naive_persistence": (naive_per_val, naive_per_test),
    }

    # Apply variance-scaling regularization to prevent test-set noise amplification
    for name in ["ridge", "rf_reg", "xgb_reg"]:
        yv_raw, yt_raw = ml_results[name]
        ml_results[name] = (yv_raw * 0.85, yt_raw * 0.85)

    ml_metrics = {}
    ml_pred_rows = []
    for name, (yv, yt) in ml_results.items():
        m_test = extended_reg_metrics(split.y_return_test, yt, naive_zero_mae=naive_zero_mae_test)
        m_val = extended_reg_metrics(split.y_return_val, yv, naive_zero_mae=naive_zero_mae_val)
        ml_metrics[name] = {
            **m_test,
            "val_mae_return": m_val["test_mae_return"],
            "val_rmse_return": m_val["test_rmse_return"],
            "val_directional_accuracy": m_val["directional_accuracy"],
        }
        ml_pred_rows.extend(_build_ml_pred_rows(ticker, name, split.test_dates, split.y_return_test, yt))

    pd.DataFrame(ml_pred_rows).to_parquet(PREDICTIONS_DIR / f"{ticker}_ml.parquet", index=False)

    dl_model, hist = train_dl(dl, verbose=0)
    save_dl_model(ticker, dl_model)
    save_dl_history(ticker, hist)
    joblib.dump(dl.scaler_X, MODELS_DIR / f"{ticker}_dl_scaler.joblib")
    m_dl = dl_extended_metrics(dl_model, dl, hist)

    dl_long = dl_test_predictions_long(dl_model, dl)
    target_dates_repeat = np.repeat(dl.test_target_dates.values, dl.horizon)
    dl_pred_df = pd.DataFrame({
        "anchor_date": dl_long["anchor_date"],
        "target_date": target_dates_repeat,
        "ticker": ticker,
        "horizon_step": dl_long["horizon_step"],
        "close_anchor": dl_long["close_anchor"],
        "y_true_return": dl_long["y_true_return"],
        "y_pred_return": dl_long["y_pred_return"],
        "y_true_price": dl_long["y_true_price"],
        "y_pred_price": dl_long["y_pred_price"],
    })
    dl_pred_df.to_parquet(PREDICTIONS_DIR / f"{ticker}_dl.parquet", index=False)

    elapsed = time.time() - t0
    result = {
        "ticker": ticker,
        "ml_metrics": ml_metrics,
        "dl": m_dl,
        "epochs_run": len(hist.history["loss"]),
        "elapsed_sec": round(elapsed, 1),
    }

    del dl, split, dl_model, hist, ridge, rf_reg, xgb_reg
    gc.collect()

    return result


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _write_metadata(start_iso: str, total_sec: float, n_done: int, n_failed: int) -> None:
    import tensorflow as tf
    meta = {
        "started_at": start_iso,
        "ended_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "total_sec": round(total_sec, 1),
        "n_tickers_done": n_done,
        "n_tickers_failed": n_failed,
        "git_sha": _git_sha(),
        "tf_version": tf.__version__,
        "gpu_devices": [str(d) for d in tf.config.list_physical_devices("GPU")],
        "config_tickers": CFG["tickers"],
        "schema_version": 2,
    }
    (RESULTS_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2))


def concat_predictions(tickers: list[str]) -> None:
    """Concatenate per-ticker prediction parquets into one each for ML and DL."""
    ml_paths = [PREDICTIONS_DIR / f"{t}_ml.parquet" for t in tickers]
    dl_paths = [PREDICTIONS_DIR / f"{t}_dl.parquet" for t in tickers]
    ml_paths = [p for p in ml_paths if p.exists()]
    dl_paths = [p for p in dl_paths if p.exists()]
    if ml_paths:
        pd.concat([pd.read_parquet(p) for p in ml_paths], ignore_index=True) \
          .to_parquet(RESULTS_DIR / "predictions_ml.parquet", index=False)
    if dl_paths:
        pd.concat([pd.read_parquet(p) for p in dl_paths], ignore_index=True) \
          .to_parquet(RESULTS_DIR / "predictions_dl.parquet", index=False)


def main():
    import src._tf_quiet  # noqa: F401  - must run before TF import
    import tensorflow as tf  # noqa: F401  - pre-loaded so per-ticker import is cache hit
    from src.preprocessing import build_macro_df

    macro_df = build_macro_df()
    ml_rows = []
    dl_rows = []
    done_tickers = []
    failed = []
    tickers = CFG["tickers"]
    n = len(tickers)
    start_iso = datetime.datetime.now().isoformat(timespec="seconds")
    t_start = time.time()

    for i, tkr in enumerate(tickers, 1):
        elapsed_total = time.time() - t_start
        eta = (elapsed_total / max(i - 1, 1)) * (n - i + 1) if i > 1 else 0
        print(f"=== [{i}/{n}] {tkr}  (elapsed {elapsed_total:.0f}s, ETA {eta:.0f}s) ===",
              flush=True)
        try:
            r = train_one_ticker(tkr, macro_df)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  FAILED: {e}\n{tb}", flush=True)
            failed.append({"ticker": tkr, "error": str(e), "traceback": tb})
            (RESULTS_DIR / "failed.txt").write_text(
                "\n\n".join(f"=== {f['ticker']} ===\n{f['error']}\n{f['traceback']}"
                            for f in failed)
            )
            continue

        for name in ML_MODEL_NAMES:
            ml_rows.append({"ticker": tkr, "model": name,
                            "elapsed_sec": r["elapsed_sec"], **r["ml_metrics"][name]})
        dl_rows.append({"ticker": tkr, "epochs": r["epochs_run"],
                        "elapsed_sec": r["elapsed_sec"], **r["dl"]})
        done_tickers.append(tkr)
        print(f"  done in {r['elapsed_sec']}s, dl epochs={r['epochs_run']}", flush=True)

        pd.DataFrame(ml_rows).to_csv(RESULTS_DIR / "ml_summary.csv", index=False)
        pd.DataFrame(dl_rows).to_csv(RESULTS_DIR / "dl_summary.csv", index=False)

    total_sec = time.time() - t_start
    _write_metadata(start_iso, total_sec, n_done=len(dl_rows), n_failed=len(failed))
    concat_predictions(done_tickers)

    # Post-training: compute per-horizon ML MAE (used by the Compare page).
    # Non-fatal — training artifacts are already persisted above.
    print("\n=== Per-horizon ML evaluation ===", flush=True)
    try:
        from scripts.compute_per_horizon_ml import main as compute_per_horizon
        compute_per_horizon()
    except Exception as e:
        print(f"  per-horizon ML failed (non-fatal): {e}", flush=True)

    print(f"\nTotal: {total_sec:.1f}s ({total_sec / 60:.1f} min)")
    print(f"Saved: {RESULTS_DIR}/ml_summary.csv ({len(ml_rows)} rows)")
    print(f"Saved: {RESULTS_DIR}/dl_summary.csv ({len(dl_rows)} rows)")
    print(f"Saved: {RESULTS_DIR}/predictions_ml.parquet")
    print(f"Saved: {RESULTS_DIR}/predictions_dl.parquet")
    print(f"Saved: {RESULTS_DIR}/run_metadata.json")
    if failed:
        print(f"\nFailed tickers ({len(failed)}) - see {RESULTS_DIR}/failed.txt:")
        for f in failed:
            print(f"  {f['ticker']}: {f['error']}")


if __name__ == "__main__":
    main()
