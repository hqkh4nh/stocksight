"""Smoke-test on 3 diverse tickers after the post-mortem fixes.

Pass criteria:
  - DL test_dir_accuracy_perday > 0.50 on at least 2/3 tickers
  - Pipeline runs end-to-end without exceptions

Outputs results/smoke_summary.csv (ml + baselines) and results/smoke_dl.csv.
"""
import time
from pathlib import Path

import pandas as pd

from scripts.train_all import train_one_ticker
from src.config import RESULTS_DIR
from src.preprocessing import build_macro_df

SMOKE_TICKERS = ["COST", "FITB"]  # consumer disc (was MAPE 130%), financial


def main():
    macro_df = build_macro_df()
    ml_rows = []
    dl_rows = []
    failed = []
    t0 = time.time()
    for tkr in SMOKE_TICKERS:
        print(f"=== {tkr} ===", flush=True)
        try:
            r = train_one_ticker(tkr, macro_df)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            failed.append((tkr, str(e)))
            continue
        for model_name in ["ridge", "rf_reg", "xgb_reg",
                           "logistic", "rf", "xgb",
                           "naive_zero", "naive_persistence", "majority"]:
            ml_rows.append({"ticker": tkr, "model": model_name, **r[model_name]})
        dl_rows.append({"ticker": tkr, "epochs": r["epochs_run"],
                        "elapsed_sec": r["elapsed_sec"], **r["dl"]})
        print(f"  done in {r['elapsed_sec']}s, dl_dir_acc_perday="
              f"{r['dl']['test_dir_accuracy_perday']:.4f}", flush=True)

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ml_rows).to_csv(RESULTS_DIR / "smoke_summary.csv", index=False)
    pd.DataFrame(dl_rows).to_csv(RESULTS_DIR / "smoke_dl.csv", index=False)
    print(f"\nTotal: {round(time.time() - t0, 1)}s")
    print(f"Saved: {RESULTS_DIR}/smoke_summary.csv ({len(ml_rows)} rows)")
    print(f"Saved: {RESULTS_DIR}/smoke_dl.csv ({len(dl_rows)} rows)")
    if failed:
        print(f"\nFailed tickers ({len(failed)}):")
        for t, e in failed:
            print(f"  {t}: {e}")

    # Quick verdict for the smoke gate
    if dl_rows:
        passed = sum(1 for d in dl_rows if d["test_dir_accuracy_perday"] > 0.50)
        print(f"\nSmoke gate: {passed}/{len(dl_rows)} tickers with dl_dir_acc > 0.50")

    print_summary(ml_rows, dl_rows)


def print_summary(ml_rows, dl_rows):
    """Print a side-by-side comparison table per ticker so the smoke run is self-explanatory."""
    if not ml_rows:
        return
    ml_df = pd.DataFrame(ml_rows)
    dl_df = pd.DataFrame(dl_rows) if dl_rows else pd.DataFrame()

    print("\n" + "=" * 78)
    print("REGRESSION (return) — lower MAE/RMSE is better. Compare ML vs naive baselines.")
    print("=" * 78)
    reg = ml_df[ml_df["model"].isin(["ridge", "rf_reg", "xgb_reg",
                                       "naive_zero", "naive_persistence"])]
    reg_pivot = reg.pivot(index="model", columns="ticker", values="test_mae_return")
    reg_pivot = reg_pivot.reindex(["naive_zero", "naive_persistence",
                                    "ridge", "rf_reg", "xgb_reg"])
    print("\nMAE return (1-day):")
    print(reg_pivot.round(5).to_string())

    print("\n" + "=" * 78)
    print("CLASSIFICATION (direction) — higher acc/F1/AUC is better. Compare to majority.")
    print("=" * 78)
    clf = ml_df[ml_df["model"].isin(["logistic", "rf", "xgb", "majority"])]
    for metric in ["test_accuracy", "test_f1", "test_roc_auc"]:
        if metric not in clf.columns:
            continue
        pivot = clf.pivot(index="model", columns="ticker", values=metric)
        pivot = pivot.reindex(["majority", "logistic", "rf", "xgb"])
        print(f"\n{metric}:")
        print(pivot.round(4).to_string())

    if not dl_df.empty:
        print("\n" + "=" * 78)
        print("DL seq2seq — 14-day metrics. dir_acc > 0.50 means it learned direction.")
        print("=" * 78)
        cols = ["ticker", "epochs", "elapsed_sec",
                "test_mae_price_14d", "test_mape_price_14d",
                "test_dir_accuracy_perday", "test_dir_f1_perday",
                "test_dir_accuracy_t14"]
        cols = [c for c in cols if c in dl_df.columns]
        print()
        print(dl_df[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
