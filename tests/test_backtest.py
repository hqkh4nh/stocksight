"""Unit tests for src.backtest with synthetic data."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import (BacktestConfig, _build_weights, _max_drawdown,
                          buy_and_hold, buy_and_hold_equal_weight,
                          compute_metrics, run_topk_daily)


@pytest.fixture
def synthetic_panel():
    """3 tickers, 100 business days, deterministic random walk."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=100)
    rets = rng.normal(0.0005, 0.01, size=(100, 3))
    prices = pd.DataFrame(100.0 * np.cumprod(1 + rets, axis=0),
                          index=dates, columns=["A", "B", "C"])
    return prices


def _signal_from_scores(scores: np.ndarray, dates, tickers):
    return pd.DataFrame(scores, index=dates, columns=tickers)


def test_topk_long_only_no_short_position(synthetic_panel):
    prices = synthetic_panel
    scores = -np.abs(np.random.default_rng(1).normal(0, 0.01, size=prices.shape))
    sig = _signal_from_scores(scores, prices.index, prices.columns)
    res = run_topk_daily(sig, prices, BacktestConfig(top_k=2, threshold=0.005))
    assert (res.positions >= 0).all().all()
    assert res.positions.sum(axis=1).max() <= 1.0 + 1e-9


def test_threshold_filters_low_scores(synthetic_panel):
    prices = synthetic_panel
    scores = np.full(prices.shape, 0.001)
    sig = _signal_from_scores(scores, prices.index, prices.columns)
    res = run_topk_daily(sig, prices, BacktestConfig(top_k=3, threshold=0.005))
    assert (res.positions == 0).all().all()


def test_zero_turnover_no_cost(synthetic_panel):
    prices = synthetic_panel
    scores = np.tile([0.10, 0.05, 0.01], (len(prices), 1))
    sig = _signal_from_scores(scores, prices.index, prices.columns)
    res_costly = run_topk_daily(sig, prices,
                                BacktestConfig(top_k=2, threshold=0.005, cost_per_trade=0.01))
    res_free = run_topk_daily(sig, prices,
                              BacktestConfig(top_k=2, threshold=0.005, cost_per_trade=0.0))
    diff = (res_free.equity - res_costly.equity).abs().max()
    assert diff < 0.02


def test_cost_reduces_return(synthetic_panel):
    prices = synthetic_panel
    rng = np.random.default_rng(7)
    scores = rng.normal(0.02, 0.05, size=prices.shape)
    sig = _signal_from_scores(scores, prices.index, prices.columns)
    res_free = run_topk_daily(sig, prices,
                              BacktestConfig(top_k=2, threshold=0.005, cost_per_trade=0.0))
    res_costly = run_topk_daily(sig, prices,
                                BacktestConfig(top_k=2, threshold=0.005, cost_per_trade=0.01))
    assert res_costly.equity.iloc[-1] < res_free.equity.iloc[-1]


def test_buy_and_hold_equal_weight_matches_individual_average(synthetic_panel):
    prices = synthetic_panel
    eq_ew = buy_and_hold_equal_weight(prices, list(prices.columns))
    individual = pd.DataFrame({t: buy_and_hold(prices[t]) for t in prices.columns})
    expected = individual.mean(axis=1)
    pd.testing.assert_series_equal(eq_ew, expected, check_names=False)


def test_metrics_sharpe_zero_for_zero_returns():
    dates = pd.bdate_range("2024-01-01", periods=60)
    r = pd.Series(0.0, index=dates)
    eq = (1 + r).cumprod()
    m = compute_metrics(r, eq)
    assert m["sharpe"] == 0.0
    assert m["max_dd"] == 0.0


def test_max_drawdown_recovery():
    dates = pd.bdate_range("2024-01-01", periods=4)
    eq = pd.Series([1.0, 1.0, 0.5, 0.7], index=dates)
    max_dd, _ = _max_drawdown(eq)
    assert max_dd == pytest.approx(-0.5)


def test_signal_lag_no_lookahead(synthetic_panel):
    """Day with strong signal should affect NEXT day's return, not same day."""
    prices = synthetic_panel
    scores = np.zeros(prices.shape)
    scores[10, 0] = 0.5
    sig = _signal_from_scores(scores, prices.index, prices.columns)
    res = run_topk_daily(sig, prices, BacktestConfig(top_k=1, threshold=0.005,
                                                    cost_per_trade=0.0))
    assert res.positions.iloc[10].sum() == 0.0
    assert res.positions.iloc[11, 0] > 0.0


def test_build_weights_top_k_cap(synthetic_panel):
    prices = synthetic_panel
    scores = np.tile([0.10, 0.05, 0.02], (len(prices), 1))
    sig = _signal_from_scores(scores, prices.index, prices.columns)
    w = _build_weights(sig, top_k=2, threshold=0.005)
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all()
    nonzero = (w > 0).sum(axis=1)
    assert nonzero.max() == 2
