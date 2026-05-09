"""Train all 5 model types (Ridge, Logistic, RF, XGB, DL) on all 21 tickers.

Outputs:
  results/ml_summary.csv  — 1-day metrics per ticker per ML model
  results/dl_summary.csv  — 14-day metrics per ticker
"""
import time
from pathlib import Path

import pandas as pd

from src.config import CFG, RESULTS_DIR
from src.evaluate import dl_metrics_14d
from src.models.dl_model import save_dl_model, train_dl
from src.models.ml_model import (save_ml_models, train_logistic, train_rf,
                                  train_ridge, train_xgb)
from src.preprocessing import build_macro_df, prepare_dl_pipeline_v2


def train_one_ticker(ticker: str, macro_df: pd.DataFrame) -> dict:
    t0 = time.time()
    dl, split = prepare_dl_pipeline_v2(ticker, macro_df)

    ridge, _, m_r = train_ridge(split)
    log, _, _, m_l = train_logistic(split)
    rf, _, _, m_rf = train_rf(split)
    xgb, _, _, m_x = train_xgb(split)
    save_ml_models(ticker, ridge, log, rf, xgb, split.scaler_X)

    dl_model, hist = train_dl(dl, verbose=0)
    save_dl_model(ticker, dl_model)
    m_dl = dl_metrics_14d(dl_model, dl)

    elapsed = time.time() - t0
    return {
        "ticker": ticker,
        "ridge": m_r, "logistic": m_l, "rf": m_rf, "xgb": m_x, "dl": m_dl,
        "epochs_run": len(hist.history["loss"]),
        "elapsed_sec": round(elapsed, 1),
    }


def main():
    macro_df = build_macro_df()
    ml_rows = []
    dl_rows = []
    failed = []
    for tkr in CFG["tickers"]:
        print(f"=== {tkr} ===", flush=True)
        try:
            r = train_one_ticker(tkr, macro_df)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            failed.append((tkr, str(e)))
            continue
        for model_name in ["ridge", "logistic", "rf", "xgb"]:
            row = {"ticker": tkr, "model": model_name, **r[model_name]}
            ml_rows.append(row)
        dl_rows.append({"ticker": tkr, "epochs": r["epochs_run"],
                        "elapsed_sec": r["elapsed_sec"], **r["dl"]})
        print(f"  done in {r['elapsed_sec']}s, dl epochs={r['epochs_run']}", flush=True)

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ml_rows).to_csv(RESULTS_DIR / "ml_summary.csv", index=False)
    pd.DataFrame(dl_rows).to_csv(RESULTS_DIR / "dl_summary.csv", index=False)
    print(f"\nSaved: {RESULTS_DIR}/ml_summary.csv ({len(ml_rows)} rows)")
    print(f"Saved: {RESULTS_DIR}/dl_summary.csv ({len(dl_rows)} rows)")
    if failed:
        print(f"\nFailed tickers ({len(failed)}):")
        for t, e in failed:
            print(f"  {t}: {e}")


if __name__ == "__main__":
    main()
