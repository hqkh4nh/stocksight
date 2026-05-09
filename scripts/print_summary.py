"""Pretty-print smoke / full-run summary CSVs as comparison tables.

Usage:
  python -m scripts.print_summary smoke      # reads results/smoke_*.csv
  python -m scripts.print_summary full       # reads results/ml_summary.csv + dl_summary.csv
"""
import sys
from pathlib import Path

import pandas as pd

from src.config import RESULTS_DIR


def print_tables(ml_csv: Path, dl_csv: Path):
    if not ml_csv.exists():
        print(f"Missing: {ml_csv}")
        return
    ml_df = pd.read_csv(ml_csv)
    dl_df = pd.read_csv(dl_csv) if dl_csv.exists() else pd.DataFrame()

    print("\n" + "=" * 78)
    print(f"REGRESSION (return) — source: {ml_csv.name}")
    print("Lower MAE/RMSE is better. A model losing to naive_zero learned nothing useful.")
    print("=" * 78)
    reg = ml_df[ml_df["model"].isin(["ridge", "rf_reg", "xgb_reg",
                                       "naive_zero", "naive_persistence"])]
    if not reg.empty:
        for metric in ["test_mae_return", "test_rmse_return"]:
            pivot = reg.pivot(index="model", columns="ticker", values=metric)
            pivot = pivot.reindex(["naive_zero", "naive_persistence",
                                    "ridge", "rf_reg", "xgb_reg"]).dropna(how="all")
            print(f"\n{metric}:")
            print(pivot.round(5).to_string())

    print("\n" + "=" * 78)
    print("CLASSIFICATION (direction) — higher is better. Compare to majority baseline.")
    print("=" * 78)
    clf = ml_df[ml_df["model"].isin(["logistic", "rf", "xgb", "majority"])]
    for metric in ["test_accuracy", "test_f1", "test_roc_auc"]:
        if metric not in clf.columns:
            continue
        pivot = clf.pivot(index="model", columns="ticker", values=metric)
        pivot = pivot.reindex(["majority", "logistic", "rf", "xgb"]).dropna(how="all")
        print(f"\n{metric}:")
        print(pivot.round(4).to_string())

    if not dl_df.empty:
        print("\n" + "=" * 78)
        print(f"DL seq2seq — 14-day metrics — source: {dl_csv.name}")
        print("dir_accuracy > 0.50 means the model learned direction.")
        print("=" * 78)
        cols = ["ticker", "epochs", "elapsed_sec",
                "test_mae_price_14d", "test_mape_price_14d",
                "test_dir_accuracy_perday", "test_dir_f1_perday",
                "test_dir_accuracy_t14"]
        cols = [c for c in cols if c in dl_df.columns]
        print()
        print(dl_df[cols].round(4).to_string(index=False))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        print_tables(RESULTS_DIR / "smoke_summary.csv", RESULTS_DIR / "smoke_dl.csv")
    elif mode == "full":
        print_tables(RESULTS_DIR / "ml_summary.csv", RESULTS_DIR / "dl_summary.csv")
    else:
        print(f"Unknown mode: {mode}. Use 'smoke' or 'full'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
