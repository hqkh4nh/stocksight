"""End-to-end backtest: 6 strategies + 3 benchmarks + cost sensitivity + plots.

Reads results/predictions_dl.parquet + results/predictions_ml.parquet,
loads SPY + per-ticker close, runs Top-K daily rebalance for each model
signal, aligns all strategies+benchmarks to a common window for FAIR
comparison, runs cost-sensitivity loop, writes:

  results/backtest_summary.csv            (9 strategies, aligned n_days)
  results/backtest_cost_sensitivity.csv   (6 strategies x 4 cost levels)
  results/backtest_equity.parquet         (long format)
  results/backtest_equity.png             (9-line full)
  results/backtest_drawdown.png           (9-line full)
  results/backtest_main.png               (5-line equity + drawdown stacked)
  results/backtest_sharpe_scatter.png     (risk-return + iso-Sharpe lines)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.backtest import (BacktestConfig, align_to_common_window,
                          build_dl_signal, build_ml_signal, buy_and_hold,
                          buy_and_hold_equal_weight, compute_metrics,
                          load_close_panel, run_topk_daily)
from src.config import CFG, RESULTS_DIR
from src.data_loader import load_spy

ML_MODELS = ["ridge", "rf_reg", "xgb_reg", "naive_zero", "naive_persistence"]
COST_LEVELS_BP = [0, 5, 10, 20]
MAIN_STRATEGIES = ["DL_seq2seq", "ML_ridge", "SPY", "EW_BuyHold", "Cash"]
STRAT_COLORS = {
    "DL_seq2seq": "tab:blue", "ML_ridge": "tab:orange",
    "ML_rf_reg": "tab:purple", "ML_xgb_reg": "tab:brown",
    "ML_naive_zero": "tab:olive", "ML_naive_persistence": "tab:pink",
    "SPY": "tab:red", "EW_BuyHold": "tab:green", "Cash": "tab:gray",
}


def _plot_equity(strats: dict, bench: dict, out_path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, r in strats.items():
        ax.plot(r.equity.index, r.equity.values, label=name, linewidth=1.2,
                color=STRAT_COLORS.get(name))
    for name, eq in bench.items():
        ax.plot(eq.index, eq.values, label=name, linewidth=1.6, linestyle="--",
                color=STRAT_COLORS.get(name))
    ax.set_yscale("log")
    ax.set_title("Equity curves (log scale, start=1.0, common window)")
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
        ax.plot(dd.index, dd.values, label=name, linewidth=1.0,
                color=STRAT_COLORS.get(name))
    for name, eq in bench.items():
        dd = eq / eq.cummax() - 1.0
        ax.plot(dd.index, dd.values, label=name, linewidth=1.4, linestyle="--",
                color=STRAT_COLORS.get(name))
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_main_combined(strats: dict, bench: dict, out_path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    all_eq = {**{k: v.equity for k, v in strats.items()}, **bench}
    for name in MAIN_STRATEGIES:
        if name not in all_eq:
            continue
        eq = all_eq[name]
        is_bench = name in bench
        ax1.plot(eq.index, eq.values, label=name, linewidth=1.6,
                 color=STRAT_COLORS[name], linestyle="--" if is_bench else "-")
        dd = eq / eq.cummax() - 1.0
        ax2.plot(dd.index, dd.values, label=name, linewidth=1.3,
                 color=STRAT_COLORS[name], linestyle="--" if is_bench else "-")

    ax1.set_yscale("log")
    ax1.set_title("Equity curves (log scale, common window)")
    ax1.set_ylabel("Equity")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_title("Drawdown")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_sharpe_scatter(summary: pd.DataFrame, out_path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    df = summary[summary["ann_vol"] > 1e-6].copy()
    x_max = float(df["ann_vol"].max()) * 1.15
    y_min = min(0.0, float(df["ann_return"].min()) * 1.15)
    y_max = float(df["ann_return"].max()) * 1.15

    for s_iso in [0.5, 1.0, 1.5, 2.0]:
        ax.plot([0, x_max], [0, s_iso * x_max], "--",
                color="gray", alpha=0.4, linewidth=0.8)
        ax.annotate(f"Sharpe={s_iso}",
                    xy=(x_max, s_iso * x_max),
                    fontsize=8, color="gray",
                    xytext=(-5, 2), textcoords="offset points", ha="right")

    for _, row in df.iterrows():
        name = row["strategy"]
        c = STRAT_COLORS.get(name, "tab:gray")
        ax.scatter(row["ann_vol"], row["ann_return"], s=140,
                   color=c, edgecolor="black", zorder=5)
        ax.annotate(name, (row["ann_vol"], row["ann_return"]),
                    xytext=(8, 5), textcoords="offset points", fontsize=9)

    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.set_title("Risk-Return profile (iso-Sharpe lines)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, x_max)
    ax.set_ylim(y_min, y_max)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _run_cost_sensitivity(signals: dict, prices: pd.DataFrame,
                          base_cfg: BacktestConfig) -> pd.DataFrame:
    rows = []
    for name, sig in signals.items():
        for bp in COST_LEVELS_BP:
            cfg = BacktestConfig(top_k=base_cfg.top_k,
                                 threshold=base_cfg.threshold,
                                 cost_per_trade=bp / 10000.0)
            r = run_topk_daily(sig, prices, cfg)
            m = compute_metrics(r.daily_returns, r.equity, r.turnover)
            rows.append({"strategy": name, "cost_bp": bp,
                         "sharpe": m["sharpe"],
                         "ann_return": m["ann_return"],
                         "total_return": m["total_return"],
                         "max_dd": m["max_dd"]})
    return pd.DataFrame(rows)


def main():
    cfg = BacktestConfig(
        top_k=5,
        threshold=CFG["backtest"]["threshold"],
        cost_per_trade=0.001,
    )

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

    signals = {"DL_seq2seq": build_dl_signal(pred_dl)}
    for m in ML_MODELS:
        try:
            signals[f"ML_{m}"] = build_ml_signal(pred_ml, m)
        except ValueError:
            print(f"Skip {m}: no rows in predictions_ml")
            continue

    print("Running default backtest (10bp cost)...")
    results = {name: run_topk_daily(sig, prices, cfg) for name, sig in signals.items()}
    bench = {
        "SPY": buy_and_hold(spy_close),
        "EW_BuyHold": buy_and_hold_equal_weight(prices, universe),
        "Cash": pd.Series(1.0, index=results["DL_seq2seq"].equity.index),
    }

    results, bench, common = align_to_common_window(results, bench)
    print(f"Common window: {common.min().date()} -> {common.max().date()} "
          f"({len(common)} days)")

    print("Running cost sensitivity (4 levels x 6 strategies)...")
    cost_df = _run_cost_sensitivity(signals, prices, cfg)
    cost_df.to_csv(RESULTS_DIR / "backtest_cost_sensitivity.csv", index=False)

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
                         .assign(strategy=name).reset_index()
                         .rename(columns={"index": "date"}))
    for name, eq in bench.items():
        eq_frames.append(eq.rename("equity").to_frame()
                         .assign(strategy=name).reset_index()
                         .rename(columns={"index": "date"}))
    pd.concat(eq_frames, ignore_index=True).to_parquet(
        RESULTS_DIR / "backtest_equity.parquet", index=False)

    _plot_equity(results, bench, RESULTS_DIR / "backtest_equity.png")
    _plot_drawdown(results, bench, RESULTS_DIR / "backtest_drawdown.png")
    _plot_main_combined(results, bench, RESULTS_DIR / "backtest_main.png")
    _plot_sharpe_scatter(summary, RESULTS_DIR / "backtest_sharpe_scatter.png")

    print("\n=== Backtest summary (common window) ===")
    print(summary.to_string(index=False))
    print("\n=== Cost sensitivity (Sharpe by cost_bp) ===")
    print(cost_df.pivot(index="strategy", columns="cost_bp", values="sharpe")
          .to_string())

    print(f"\nSaved: {RESULTS_DIR / 'backtest_summary.csv'}")
    print(f"Saved: {RESULTS_DIR / 'backtest_cost_sensitivity.csv'}")
    print(f"Saved: {RESULTS_DIR / 'backtest_equity.parquet'}")
    print(f"Saved: {RESULTS_DIR / 'backtest_equity.png'}")
    print(f"Saved: {RESULTS_DIR / 'backtest_drawdown.png'}")
    print(f"Saved: {RESULTS_DIR / 'backtest_main.png'}")
    print(f"Saved: {RESULTS_DIR / 'backtest_sharpe_scatter.png'}")


if __name__ == "__main__":
    main()
