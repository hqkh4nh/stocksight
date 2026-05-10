"""Smoke test for ML predictions parquet schema produced by train_one_ticker.

Does NOT spin up the DL model (avoids GPU dependency in unit test). Instead it
exercises the ML pred-rows builder directly with a tiny synthetic split.
"""
import pandas as pd

from scripts.train_all import _baseline_predictions, _build_ml_pred_rows


def test_baseline_predictions_naive_zero_shape(tiny_split):
    yv, yt = _baseline_predictions("naive_zero", tiny_split)
    assert yv.shape == tiny_split.y_return_val.shape
    assert yt.shape == tiny_split.y_return_test.shape
    assert (yv == 0).all() and (yt == 0).all()


def test_baseline_predictions_persistence_uses_prior(tiny_split):
    yv, yt = _baseline_predictions("naive_persistence", tiny_split)
    assert yv.shape == tiny_split.y_return_val.shape
    assert yt.shape == tiny_split.y_return_test.shape
    # First test prediction should equal last validation return.
    assert yt[0] == float(tiny_split.y_return_val[-1])


def test_ml_pred_rows_schema(tiny_split):
    yv, yt = _baseline_predictions("naive_zero", tiny_split)
    dates = pd.Series(pd.date_range("2024-01-01", periods=len(yt), freq="B"))
    rows = _build_ml_pred_rows("FOO", "naive_zero", dates, tiny_split.y_return_test, yt)
    df = pd.DataFrame(rows)
    assert {"date", "ticker", "model", "y_true_return", "y_pred_return",
            "y_pred_direction"} == set(df.columns)
    assert (df["ticker"] == "FOO").all()
    assert (df["model"] == "naive_zero").all()
    assert len(df) == len(yt)
