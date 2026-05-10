"""Smoke-test 3 tickers across sectors before running train_all on all 21.

Gates (all must pass for exit 0):
  G1 - regressors learn something: ridge mae_vs_naive_ratio < 1.05 on >=2/3
  G2 - DL not collapsed: dl test_dir_accuracy_perday > 0.48 on >=2/3
  G3 - DL not one-sided: dl test_dir_f1_perday > 0.10 on >=2/3
  G4 - predictions persist correctly: each ticker has DL parquet with N*14 rows
"""
import sys
import time
from pathlib import Path

import pandas as pd

from scripts.train_all import (PREDICTIONS_DIR, ML_MODEL_NAMES,
                               concat_predictions, train_one_ticker)
from src.config import RESULTS_DIR
from src.preprocessing import build_macro_df

SMOKE_TICKERS = ["COST", "FITB", "AEP"]   # consumer disc + financial + utility


def _check_gates(ml_rows: list[dict], dl_rows: list[dict]) -> tuple[bool, list[str]]:
    msgs = []
    ok = True

    ridge_rows = [r for r in ml_rows if r["model"] == "ridge"]
    g1_pass = sum(1 for r in ridge_rows
                  if r.get("mae_vs_naive_ratio") is not None
                  and r["mae_vs_naive_ratio"] < 1.05)
    msgs.append(f"  G1 ridge mae_vs_naive_ratio < 1.05 on {g1_pass}/{len(ridge_rows)}")
    if g1_pass < 2:
        ok = False

    g2_pass = sum(1 for r in dl_rows if r["test_dir_accuracy_perday"] > 0.48)
    msgs.append(f"  G2 dl_dir_acc_perday > 0.48 on {g2_pass}/{len(dl_rows)}")
    if g2_pass < 2:
        ok = False

    g3_pass = sum(1 for r in dl_rows if r["test_dir_f1_perday"] > 0.10)
    msgs.append(f"  G3 dl_dir_f1_perday > 0.10 on {g3_pass}/{len(dl_rows)}")
    if g3_pass < 2:
        ok = False

    g4_pass = 0
    for r in dl_rows:
        path = PREDICTIONS_DIR / f"{r['ticker']}_dl.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if len(df) > 0 and len(df) % 14 == 0 and {"anchor_date", "target_date",
                                                  "horizon_step", "y_pred_price"}.issubset(df.columns):
            g4_pass += 1
    msgs.append(f"  G4 predictions parquet OK on {g4_pass}/{len(dl_rows)}")
    if g4_pass < len(dl_rows):
        ok = False

    return ok, msgs


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
        for name in ML_MODEL_NAMES:
            ml_rows.append({"ticker": tkr, "model": name, **r["ml_metrics"][name]})
        dl_rows.append({"ticker": tkr, "epochs": r["epochs_run"],
                        "elapsed_sec": r["elapsed_sec"], **r["dl"]})
        print(f"  done in {r['elapsed_sec']}s  "
              f"dl_dir_acc={r['dl']['test_dir_accuracy_perday']:.4f}  "
              f"dl_f1={r['dl']['test_dir_f1_perday']:.4f}", flush=True)

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ml_rows).to_csv(RESULTS_DIR / "smoke_summary.csv", index=False)
    pd.DataFrame(dl_rows).to_csv(RESULTS_DIR / "smoke_dl.csv", index=False)
    concat_predictions([r["ticker"] for r in dl_rows])
    print(f"\nTotal: {round(time.time() - t0, 1)}s")
    print(f"Saved: {RESULTS_DIR}/smoke_summary.csv ({len(ml_rows)} rows)")
    print(f"Saved: {RESULTS_DIR}/smoke_dl.csv ({len(dl_rows)} rows)")
    if failed:
        print(f"\nFailed tickers ({len(failed)}):")
        for t, e in failed:
            print(f"  {t}: {e}")

    print_summary(ml_rows, dl_rows)

    print("\n=== SMOKE GATES ===")
    ok, msgs = _check_gates(ml_rows, dl_rows)
    for m in msgs:
        print(m)
    print(f"\nResult: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


def print_summary(ml_rows, dl_rows):
    if not ml_rows:
        return
    ml_df = pd.DataFrame(ml_rows)
    dl_df = pd.DataFrame(dl_rows) if dl_rows else pd.DataFrame()

    print("\n" + "=" * 78)
    print("REGRESSION (return) - lower MAE is better; mae_vs_naive_ratio < 1 beats baseline")
    print("=" * 78)
    cols = ["ticker", "model", "test_mae_return", "val_mae_return",
            "mae_vs_naive_ratio", "directional_accuracy", "r2_score"]
    cols = [c for c in cols if c in ml_df.columns]
    print(ml_df[cols].round(5).to_string(index=False))

    if not dl_df.empty:
        print("\n" + "=" * 78)
        print("DL seq2seq - 14-day metrics (per-horizon: d1, d7, d14)")
        print("=" * 78)
        cols = ["ticker", "epochs", "elapsed_sec",
                "test_mae_price_d1", "test_mae_price_d7", "test_mae_price_d14",
                "test_mape_price_14d",
                "test_dir_accuracy_perday", "test_dir_f1_perday",
                "test_dir_accuracy_t14",
                "train_loss_final", "val_loss_final"]
        cols = [c for c in cols if c in dl_df.columns]
        print(dl_df[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
