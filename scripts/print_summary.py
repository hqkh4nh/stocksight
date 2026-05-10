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
    print(f"REGRESSION (return) - source: {ml_csv.name}")
    print("Lower MAE is better. mae_vs_naive_ratio < 1.0 means model beats naive_zero.")
    print("=" * 78)
    if not ml_df.empty:
        for metric in ["test_mae_return", "mae_vs_naive_ratio", "directional_accuracy"]:
            if metric not in ml_df.columns:
                continue
            pivot = ml_df.pivot(index="model", columns="ticker", values=metric)
            order = ["naive_zero", "naive_persistence", "ridge", "rf_reg", "xgb_reg"]
            pivot = pivot.reindex([m for m in order if m in pivot.index])
            print(f"\n{metric}:")
            print(pivot.round(5).to_string())

    if not dl_df.empty:
        print("\n" + "=" * 78)
        print(f"DL seq2seq - 14-day metrics - source: {dl_csv.name}")
        print("dir_accuracy > 0.50 means the model learned direction.")
        print("=" * 78)
        cols = ["ticker", "epochs", "elapsed_sec",
                "test_mae_price_d1", "test_mae_price_d7", "test_mae_price_d14",
                "test_mae_price_14d", "test_mape_price_14d",
                "test_dir_accuracy_perday", "test_dir_f1_perday",
                "test_dir_accuracy_t14",
                "train_loss_final", "val_loss_final"]
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
