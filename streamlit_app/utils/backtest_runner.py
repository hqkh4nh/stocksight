"""Single-ticker backtest wrapper using existing src.backtest engine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.backtest import (BacktestConfig, BacktestResult, buy_and_hold,
                          compute_metrics, load_close_panel)
from src.config import CFG
from streamlit_app.utils.data import (load_predictions_dl,
                                      load_predictions_ml)


def _build_single_dl_signal(ticker: str) -> pd.DataFrame:
    df = load_predictions_dl()
    df = df[(df["ticker"] == ticker) & (df["horizon_step"] == 14)].copy()
    df["cum_pred_14d"] = df["y_pred_price"] / df["close_anchor"] - 1.0
    return (df[["anchor_date", "cum_pred_14d"]]
            .rename(columns={"anchor_date": "date", "cum_pred_14d": ticker})
            .set_index("date")
            .sort_index())


def _build_single_ml_signal(ticker: str, model_name: str) -> pd.DataFrame:
    df = load_predictions_ml()
    df = df[(df["ticker"] == ticker) & (df["model"] == model_name)].copy()
    return (df[["date", "y_pred_return"]]
            .rename(columns={"y_pred_return": ticker})
            .set_index("date")
            .sort_index())


def _topk1_signal_run(signal: pd.DataFrame, prices: pd.DataFrame,
                      cfg: BacktestConfig) -> BacktestResult:
    """Single-asset variant: if signal > threshold, go full long; else cash."""
    ticker = signal.columns[0]
    common = signal.index.intersection(prices.index)
    signal = signal.loc[common]
    prices = prices.loc[common]

    weight = (signal[ticker] > cfg.threshold).astype(float)
    positions = weight.shift(1).fillna(0.0).to_frame(name=ticker)

    rets = prices.pct_change().fillna(0.0)
    gross_ret = (positions[ticker] * rets[ticker])
    volume = positions[ticker].diff().abs().fillna(positions[ticker].abs())
    cost = cfg.cost_per_trade * volume
    net_ret = gross_ret - cost
    equity = (1.0 + net_ret).cumprod()
    return BacktestResult(equity=equity, daily_returns=net_ret,
                          positions=positions, turnover=volume / 2.0)


@st.cache_data
def run_single(ticker: str, model_name: str, initial_capital: float = 10000.0,
               cost_per_trade: float = 0.001) -> dict:
    """Run backtest on a single ticker using its model's predictions.

    Returns dict with: equity, benchmark, drawdown, trades, metrics.
    """
    if model_name == "dl":
        signal = _build_single_dl_signal(ticker)
    else:
        signal = _build_single_ml_signal(ticker, model_name)

    if signal.empty:
        raise ValueError(f"No predictions for {ticker} / {model_name}")

    start = pd.Timestamp(signal.index.min())
    end = pd.Timestamp(signal.index.max()) + pd.Timedelta(days=30)
    prices = load_close_panel([ticker], start, end)

    cfg = BacktestConfig(top_k=1, threshold=CFG["backtest"]["threshold"],
                         cost_per_trade=cost_per_trade)
    result = _topk1_signal_run(signal, prices, cfg)

    bh = buy_and_hold(prices[ticker].loc[result.equity.index])
    bh_ret = bh.pct_change().fillna(0.0)

    metrics = compute_metrics(result.daily_returns, result.equity,
                              turnover=result.turnover,
                              benchmark_returns=bh_ret)
    bh_metrics = compute_metrics(bh_ret, bh)

    pos = result.positions[ticker]
    drawdown = result.equity / result.equity.cummax() - 1.0

    trade_rows = []
    prev_w = 0.0
    for d, w in pos.items():
        if abs(w - prev_w) > 1e-9:
            action = "ENTER" if w > prev_w else "EXIT"
            trade_rows.append({"date": d, "action": action,
                               "weight": w, "price": prices[ticker].loc[d]
                                                    if d in prices.index else np.nan})
        prev_w = w
    trades = pd.DataFrame(trade_rows)

    return {
        "equity": result.equity * initial_capital,
        "benchmark": bh * initial_capital,
        "drawdown": drawdown,
        "trades": trades,
        "metrics": metrics,
        "bh_metrics": bh_metrics,
    }
