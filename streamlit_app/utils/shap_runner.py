"""SHAP wrappers for tree-based ML models (RF, XGB).

Uses prepare_ml_split (via prepare_dl_pipeline which composes it) to obtain
the same 2-D test features the model was trained on.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.config import MODELS_DIR
from src.preprocessing import prepare_dl_pipeline
from streamlit_app.utils.data import macro_df

FEATURE_NAME_EN = {
    "returns": "Daily return",
    "log_returns": "Log return",
    "close_pct_lag_1": "Close % 1d ago",
    "close_pct_lag_5": "Close % 5d ago",
    "ma_5": "MA 5",
    "ma_20": "MA 20",
    "ema_12": "EMA 12",
    "ema_26": "EMA 26",
    "rsi_14": "RSI 14",
    "macd": "MACD",
    "macd_signal": "MACD signal",
    "volatility_21": "Volatility 21d",
    "bb_width": "Bollinger width",
    "volume_ratio": "Volume / MA20",
    "vix_change": "VIX change",
    "tnx_change": "10Y rate change",
    "oil_change": "Oil change",
    "usd_change": "USD index change",
    "stock_oil_corr_21": "Oil corr 21d",
    "vol_x_oil": "Vol x |Oil|",
    "stock_rate_corr_21": "Rate corr 21d",
}


def pretty_name(feat: str) -> str:
    return FEATURE_NAME_EN.get(feat, feat)


@st.cache_resource
def _load_model(ticker: str, model_name: str):
    path = MODELS_DIR / f"{ticker}_{model_name}.pkl"
    return joblib.load(path)


@st.cache_resource
def _get_test_frame(ticker: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X_test_df, test_dates) using same pipeline as training."""
    _, split = prepare_dl_pipeline(ticker, macro_df())
    df = pd.DataFrame(split.X_test, columns=split.feature_names)
    return df, split.test_dates


@st.cache_resource
def compute_shap(ticker: str, model_name: str):
    """Compute SHAP values once per (ticker, model_name). Cached."""
    model = _load_model(ticker, model_name)
    X_test, dates = _get_test_frame(ticker)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    return shap_values, X_test, dates


def interpret_top_features(shap_row: np.ndarray, feature_names: list,
                           top_n: int = 3) -> str:
    abs_vals = np.abs(shap_row)
    order = np.argsort(abs_vals)[::-1][:top_n]
    parts = []
    for i in order:
        sign = "+" if shap_row[i] >= 0 else "-"
        parts.append(f"{pretty_name(feature_names[i])} ({sign}{abs(shap_row[i]):.4f})")
    total = float(np.sum(np.abs(shap_row[order])))
    return (f"Prediction is driven mainly by {', '.join(parts)}. "
            f"Sum of |SHAP| for these {top_n} factors = {total:.4f} return.")
