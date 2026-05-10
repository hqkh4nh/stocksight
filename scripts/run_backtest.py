"""End-to-end backtest: 6 strategies + 3 benchmarks + 2 PNG plots.

Reads results/predictions_dl.parquet + results/predictions_ml.parquet,
loads SPY + per-ticker close, runs Top-K daily rebalance for each model
signal, compares against SPY / EW B&H / Cash, writes:

  results/backtest_summary.csv
  results/backtest_equity.parquet
  results/backtest_equity.png
  results/backtest_drawdown.png
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.backtest import (BacktestConfig, build_dl_signal, build_ml_signal,
                          buy_and_hold, buy_and_hold_equal_weight,
                          compute_metrics, load_close_panel, run_topk_daily)
from src.config import CFG, RESULTS_DIR
from src.data_loader import load_spy

ML_MODELS = ["ridge", "rf_reg", "xgb_reg", "naive_zero", "naive_persistence"]


def _plot_equity(strats: dict, bench: dict, out_path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, r in strats.items():
        ax.plot(r.equity.index, r.equity.values, label=name, linewidth=1.2)
    for name, eq in bench.items():
        ax.plot(eq.index, eq.values, label=name, linewidth=1.6, linestyle="--")
    ax.set_yscale("log")
    ax.set_title("Equity curves (log scale, start=1.0)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_drawdown(strats: dict, bench: dict, out_path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, r in strats.items():
        dd = r.equity / r.equity.cummax() - 1.0
        ax.plot(dd.index, dd.values, label=name, linewidth=1.0)
    for name, eq in bench.items():
        dd = eq / eq.cummax() - 1.0
        ax.plot(dd.index, dd.values, label=name, linewidth=1.4, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    cfg = BacktestConfig(
        top_k=5,
        threshold=CFG["backtest"]["threshold"],
        cost_per_trade=0.001,
    )
    tickers = CFG["tickers"]

    pred_dl_path = RESULTS_DIR / "predictions_dl.parquet"
    pred_ml_path = RESULTS_DIR / "predictions_ml.parquet"
    if not pred_dl_path.exists() or not pred_ml_path.exists():
        raise FileNotFoundError(
            f"Predictions parquet missing. Run scripts.train_all first.\n"
            f"  expected: {pred_dl_path}\n  expected: {pred_ml_path}"
        )

    pred_dl = pd.read_parquet(pred_dl_path)
    pred_ml = pd.read_parquet(pred_ml_path)

    universe = sorted(set(pred_dl["ticker"]) | set(pred_ml["ticker"]))
    start = pd.Timestamp(min(pred_dl["anchor_date"].min(), pred_ml["date"].min()))
    end = pd.Timestamp(max(pred_dl["anchor_date"].max(), pred_ml["date"].max())) + pd.Timedelta(days=20)

    prices = load_close_panel(universe, start=start, end=end)
    spy_close = load_spy().set_index("date")["close"]
    spy_close.index = pd.to_datetime(spy_close.index)
    spy_close = spy_close.loc[start:end]

    results = {"DL_seq2seq": run_topk_daily(build_dl_signal(pred_dl), prices, cfg)}
    for m in ML_MODELS:
        try:
            sig = build_ml_signal(pred_ml, m)
        except ValueError:
            print(f"Skip {m}: no rows in predictions_ml")
            continue
        results[f"ML_{m}"] = run_topk_daily(sig, prices, cfg)

    bench = {
        "SPY": buy_and_hold(spy_close),
        "EW_BuyHold": buy_and_hold_equal_weight(prices, universe),
        "Cash": pd.Series(1.0, index=results["DL_seq2seq"].equity.index),
    }

    spy_returns = bench["SPY"].pct_change().dropna()

    rows = []
    for name, r in results.items():
        m = compute_metrics(r.daily_returns, r.equity, r.turnover, spy_returns)
        rows.append({"strategy": name, "n_days": len(r.daily_returns), **m})
    for name, eq in bench.items():
        ret = eq.pct_change().dropna()
        m = compute_metrics(ret, eq, benchmark_returns=spy_returns)
        rows.append({"strategy": name, "n_days": len(ret), **m})

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "backtest_summary.csv", index=False)

    eq_frames = []
    for name, r in results.items():
        eq_frames.append(r.equity.rename("equity").to_frame()
                         .assign(strategy=name).reset_index().rename(columns={"index": "date"}))
    for name, eq in bench.items():
        eq_frames.append(eq.rename("equity").to_frame()
                         .assign(strategy=name).reset_index().rename(columns={"index": "date"}))
    pd.concat(eq_frames, ignore_index=True).to_parquet(
        RESULTS_DIR / "backtest_equity.parquet", index=False)

    _plot_equity(results, bench, RESULTS_DIR / "backtest_equity.png")
    _plot_drawdown(results, bench, RESULTS_DIR / "backtest_drawdown.png")

    print("\n=== Backtest summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved: {RESULTS_DIR / 'backtest_summary.csv'}")
    print(f"Saved: {RESULTS_DIR / 'backtest_equity.parquet'}")
    print(f"Saved: {RESULTS_DIR / 'backtest_equity.png'}")
    print(f"Saved: {RESULTS_DIR / 'backtest_drawdown.png'}")


if __name__ == "__main__":
    main()
