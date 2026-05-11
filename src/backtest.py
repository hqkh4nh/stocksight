"""Top-K daily rebalance backtest for DL/ML/baseline predictions.

Long-only. Position at date t is decided from signal at date t-1
(no look-ahead). Equal-weight 1/top_k per selected ticker; cash for the
remainder when fewer than top_k tickers pass the threshold.

Cost model: cost_per_trade applied to each unit of trade volume,
where total_volume_t = sum_i |w_{i,t} - w_{i,t-1}|.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.data_loader import load_stock


@dataclass
class BacktestConfig:
    top_k: int = 5
    threshold: float = 0.005
    cost_per_trade: float = 0.001
    annualization: int = 252


@dataclass
class BacktestResult:
    equity: pd.Series
    daily_returns: pd.Series
    positions: pd.DataFrame
    turnover: pd.Series
    metrics: dict = field(default_factory=dict)


# --- signal builders ---------------------------------------------------------

def build_dl_signal(predictions_dl: pd.DataFrame) -> pd.DataFrame:
    """Wide df: index=anchor_date, columns=ticker, values=cum_pred_14d.

    cum_pred_14d = y_pred_price[horizon_step=14] / close_anchor - 1
    """
    df = predictions_dl.loc[predictions_dl["horizon_step"] == 14].copy()
    df["cum_pred_14d"] = df["y_pred_price"] / df["close_anchor"] - 1.0
    df["anchor_date"] = pd.to_datetime(df["anchor_date"])
    wide = df.pivot(index="anchor_date", columns="ticker", values="cum_pred_14d")
    return wide.sort_index()


def build_ml_signal(predictions_ml: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Wide df: index=date, columns=ticker, values=y_pred_return."""
    df = predictions_ml.loc[predictions_ml["model"] == model_name].copy()
    if df.empty:
        raise ValueError(f"No predictions for model={model_name}")
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot(index="date", columns="ticker", values="y_pred_return")
    return wide.sort_index()


# --- price loader -----------------------------------------------------------

def load_close_panel(tickers: list[str],
                     start: pd.Timestamp,
                     end: pd.Timestamp) -> pd.DataFrame:
    """Wide df: index=date, columns=ticker, values=close (auto-adjusted)."""
    series = {}
    for t in tickers:
        df = load_stock(t)[["date", "close"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        series[t] = df.set_index("date")["close"]
    panel = pd.DataFrame(series).sort_index()
    return panel


# --- core engine ------------------------------------------------------------

def _build_weights(signal: pd.DataFrame, top_k: int, threshold: float) -> pd.DataFrame:
    """For each row: keep up to top_k tickers with score > threshold; weight = 1/top_k each.
    Remainder is implicit cash. Output same index/columns as signal, values in [0, 1/top_k]."""
    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    w_unit = 1.0 / top_k
    for date, row in signal.iterrows():
        eligible = row[row > threshold].dropna()
        if eligible.empty:
            continue
        chosen = eligible.nlargest(top_k).index
        weights.loc[date, chosen] = w_unit
    return weights


def run_topk_daily(signal: pd.DataFrame,
                   prices: pd.DataFrame,
                   cfg: BacktestConfig) -> BacktestResult:
    """
    1. Restrict to dates available in BOTH signal & prices; reindex signal to price dates,
       forward-fill (max 1 day) so every price day has a signal.
    2. Compute weights from signal, then SHIFT by 1 day (signal at t-1 drives position at t).
    3. Per-ticker daily return = price[t] / price[t-1] - 1.
    4. Gross portfolio return[t] = sum_i w[t] * r[t].
    5. Volume[t] = sum_i |w[t] - w[t-1]|. Cost[t] = cost_per_trade * volume[t].
    6. Net return = gross - cost. Equity = cumprod(1 + net).
    """
    common_tickers = [t for t in signal.columns if t in prices.columns]
    signal = signal[common_tickers]
    prices = prices[common_tickers]

    start = max(signal.index.min(), prices.index.min())
    end = min(signal.index.max(), prices.index.max())
    prices = prices.loc[start:end].dropna(how="all")

    signal_aligned = signal.reindex(prices.index).ffill(limit=1)
    weights_today = _build_weights(signal_aligned, cfg.top_k, cfg.threshold)
    positions = weights_today.shift(1).fillna(0.0)

    rets = prices.pct_change().fillna(0.0)
    gross_ret = (positions * rets).sum(axis=1)

    weight_diff = positions.diff().abs().fillna(positions.abs())
    volume = weight_diff.sum(axis=1)
    cost = cfg.cost_per_trade * volume
    net_ret = gross_ret - cost

    equity = (1.0 + net_ret).cumprod()
    turnover = volume / 2.0

    return BacktestResult(
        equity=equity, daily_returns=net_ret,
        positions=positions, turnover=turnover,
    )


# --- common-window alignment ------------------------------------------------

def align_to_common_window(
    strategy_results: dict,
    benchmark_equities: dict,
) -> tuple[dict, dict, pd.DatetimeIndex]:
    """Trim every strategy & benchmark to the common date intersection.

    Returns NEW BacktestResult objects with equity re-derived from sliced
    daily_returns (equity[0] = 1 + r[0]) and benchmark equities re-normalized
    via pct_change → cumprod on the same window.
    """
    common = None
    for r in strategy_results.values():
        idx = r.daily_returns.index
        common = idx if common is None else common.intersection(idx)
    for eq in benchmark_equities.values():
        ret_idx = eq.pct_change().dropna().index
        common = common.intersection(ret_idx)
    common = common.sort_values()

    new_results = {}
    for name, r in strategy_results.items():
        rets = r.daily_returns.loc[common]
        eq = (1.0 + rets).cumprod()
        positions = r.positions.loc[common]
        turnover = r.turnover.loc[common]
        new_results[name] = BacktestResult(
            equity=eq, daily_returns=rets,
            positions=positions, turnover=turnover,
        )

    new_bench = {}
    for name, eq in benchmark_equities.items():
        rets = eq.pct_change().loc[common].fillna(0.0)
        new_bench[name] = (1.0 + rets).cumprod()

    return new_results, new_bench, common


# --- benchmarks -------------------------------------------------------------

def buy_and_hold(prices: pd.Series) -> pd.Series:
    """Single-asset B&H, normalized to start=1."""
    p = prices.dropna()
    return p / p.iloc[0]


def buy_and_hold_equal_weight(prices: pd.DataFrame, tickers: list[str]) -> pd.Series:
    """Equal-weight B&H: each ticker normalized to start=1, then averaged."""
    cols = [t for t in tickers if t in prices.columns]
    p = prices[cols].dropna(how="all")
    p = p.dropna(axis=0)
    norm = p / p.iloc[0]
    return norm.mean(axis=1)


# --- metrics ----------------------------------------------------------------

def _max_drawdown(equity: pd.Series) -> tuple[float, int]:
    cummax = equity.cummax()
    dd = equity / cummax - 1.0
    max_dd = float(dd.min())
    end = dd.idxmin()
    start = equity.loc[:end].idxmax()
    duration_days = int((end - start).days) if pd.notna(start) and pd.notna(end) else 0
    return max_dd, duration_days


def _alpha_tstat(strategy_ret: pd.Series, bench_ret: pd.Series) -> tuple[Optional[float], Optional[float]]:
    df = pd.concat([strategy_ret, bench_ret], axis=1, keys=["s", "b"]).dropna()
    if len(df) < 30:
        return None, None
    x = np.column_stack([np.ones(len(df)), df["b"].values])
    y = df["s"].values
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    resid = y - x @ coef
    n, k = len(y), 2
    if n - k <= 0:
        return alpha, None
    sigma2 = float((resid @ resid) / (n - k))
    xtx_inv = np.linalg.inv(x.T @ x)
    se_alpha = float(np.sqrt(sigma2 * xtx_inv[0, 0]))
    tstat = alpha / se_alpha if se_alpha > 0 else None
    return alpha, tstat


def compute_metrics(daily_returns: pd.Series,
                    equity: pd.Series,
                    turnover: Optional[pd.Series] = None,
                    benchmark_returns: Optional[pd.Series] = None,
                    annualization: int = 252) -> dict:
    r = daily_returns.dropna()
    if len(r) == 0:
        return {"total_return": 0.0, "ann_return": 0.0, "ann_vol": 0.0,
                "sharpe": 0.0, "max_dd": 0.0, "max_dd_days": 0,
                "hit_rate": 0.0, "avg_turnover": 0.0,
                "alpha_vs_bench": None, "tstat_alpha": None}

    total_ret = float(equity.iloc[-1] - 1.0)
    ann_ret = float(r.mean() * annualization)
    ann_vol = float(r.std() * np.sqrt(annualization))
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
    max_dd, dd_days = _max_drawdown(equity)
    hit_rate = float((r > 0).mean())
    avg_turnover = float(turnover.mean()) if turnover is not None else 0.0

    alpha = tstat = None
    if benchmark_returns is not None:
        alpha, tstat = _alpha_tstat(r, benchmark_returns)
        if alpha is not None:
            alpha = alpha * annualization

    return {
        "total_return": total_ret,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "max_dd_days": dd_days,
        "hit_rate": hit_rate,
        "avg_turnover": avg_turnover,
        "alpha_vs_bench": alpha,
        "tstat_alpha": tstat,
    }
