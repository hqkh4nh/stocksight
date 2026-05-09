"""Train all 5 model types (Ridge, Logistic, RF, XGB, DL) on all 21 tickers.

Outputs:
  results/ml_summary.csv     — 1-day metrics per ticker per ML model (incremental)
  results/dl_summary.csv     — 14-day metrics per ticker (incremental)
  results/run_metadata.json  — timestamp + git SHA + GPU info
  results/failed.txt         — tickers that errored (with traceback)

CSVs are flushed after every ticker so a crash mid-run never loses prior work.
"""
import datetime
import gc
import json
import subprocess
import time
import traceback
from pathlib import Path

import pandas as pd

import src._tf_quiet  # noqa: F401  — must run before TF import
import tensorflow as tf

from src.baselines import all_baselines
from src.config import CFG, RESULTS_DIR
from src.evaluate import dl_metrics_14d
from src.models.dl_model import save_dl_model, train_dl
from src.models.ml_model import (save_ml_models, train_logistic, train_rf,
                                  train_rf_reg, train_ridge, train_xgb,
                                  train_xgb_reg)
from src.preprocessing import build_macro_df, prepare_dl_pipeline_v2


def train_one_ticker(ticker: str, macro_df: pd.DataFrame) -> dict:
    # Release prior ticker's TF graph + variables so GPU memory doesn't accumulate
    tf.keras.backend.clear_session()
    t0 = time.time()
    dl, split = prepare_dl_pipeline_v2(ticker, macro_df)

    ridge, _, m_r = train_ridge(split)
    rf_reg, _, m_rf_reg = train_rf_reg(split)
    xgb_reg, _, m_xgb_reg = train_xgb_reg(split)
    log, _, _, m_l = train_logistic(split)
    rf, _, _, m_rf = train_rf(split)
    xgb, _, _, m_x = train_xgb(split)
    save_ml_models(ticker, ridge, log, rf, xgb, split.scaler_X,
                   rf_reg=rf_reg, xgb_reg=xgb_reg)

    base = all_baselines(split)

    dl_model, hist = train_dl(dl, verbose=0)
    save_dl_model(ticker, dl_model)
    m_dl = dl_metrics_14d(dl_model, dl)

    elapsed = time.time() - t0
    result = {
        "ticker": ticker,
        "ridge": m_r, "rf_reg": m_rf_reg, "xgb_reg": m_xgb_reg,
        "logistic": m_l, "rf": m_rf, "xgb": m_x, "dl": m_dl,
        "naive_zero": base["naive_zero"],
        "naive_persistence": base["naive_persistence"],
        "majority": base["majority"],
        "epochs_run": len(hist.history["loss"]),
        "elapsed_sec": round(elapsed, 1),
    }

    # Free Python refs so the next ticker doesn't accumulate hundreds of MB of arrays
    del dl, split, dl_model, hist, ridge, rf_reg, xgb_reg, log, rf, xgb, base
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
    }
    (RESULTS_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2))


def main():
    macro_df = build_macro_df()
    ml_rows = []
    dl_rows = []
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
            # Persist failures incrementally too
            (RESULTS_DIR / "failed.txt").write_text(
                "\n\n".join(f"=== {f['ticker']} ===\n{f['error']}\n{f['traceback']}"
                            for f in failed)
            )
            continue

        for model_name in ["ridge", "rf_reg", "xgb_reg",
                           "logistic", "rf", "xgb",
                           "naive_zero", "naive_persistence", "majority"]:
            ml_rows.append({"ticker": tkr, "model": model_name,
                            "elapsed_sec": r["elapsed_sec"], **r[model_name]})
        dl_rows.append({"ticker": tkr, "epochs": r["epochs_run"],
                        "elapsed_sec": r["elapsed_sec"], **r["dl"]})
        print(f"  done in {r['elapsed_sec']}s, dl epochs={r['epochs_run']}", flush=True)

        # Incremental flush so a crash never loses prior tickers
        pd.DataFrame(ml_rows).to_csv(RESULTS_DIR / "ml_summary.csv", index=False)
        pd.DataFrame(dl_rows).to_csv(RESULTS_DIR / "dl_summary.csv", index=False)

    total_sec = time.time() - t_start
    _write_metadata(start_iso, total_sec, n_done=len(dl_rows), n_failed=len(failed))

    print(f"\nTotal: {total_sec:.1f}s ({total_sec / 60:.1f} min)")
    print(f"Saved: {RESULTS_DIR}/ml_summary.csv ({len(ml_rows)} rows)")
    print(f"Saved: {RESULTS_DIR}/dl_summary.csv ({len(dl_rows)} rows)")
    print(f"Saved: {RESULTS_DIR}/run_metadata.json")
    if failed:
        print(f"\nFailed tickers ({len(failed)}) — see {RESULTS_DIR}/failed.txt:")
        for f in failed:
            print(f"  {f['ticker']}: {f['error']}")


if __name__ == "__main__":
    main()
