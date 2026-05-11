"""Cached data loaders for the Streamlit app."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import RESULTS_DIR
from src.data_loader import load_stock as _load_stock_raw
from src.features import compute_features
from src.preprocessing import build_macro_df


@st.cache_resource
def macro_df() -> pd.DataFrame:
    return build_macro_df()


@st.cache_data
def load_stock_cached(ticker: str) -> pd.DataFrame:
    df = _load_stock_raw(ticker).copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data
def load_features_cached(ticker: str) -> pd.DataFrame:
    stock = load_stock_cached(ticker)
    feats = compute_features(stock, macro_df(), ticker=ticker)
    return feats.dropna().reset_index(drop=True)


@st.cache_data
def load_ml_summary() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "ml_summary.csv")


@st.cache_data
def load_dl_summary() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "dl_summary.csv")


@st.cache_data
def load_run_metadata() -> dict:
    path = RESULTS_DIR / "run_metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@st.cache_data
def load_predictions_ml() -> pd.DataFrame:
    path = RESULTS_DIR / "predictions_ml.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_predictions_dl() -> pd.DataFrame:
    path = RESULTS_DIR / "predictions_dl.parquet"
    df = pd.read_parquet(path)
    df["anchor_date"] = pd.to_datetime(df["anchor_date"])
    df["target_date"] = pd.to_datetime(df["target_date"])
    return df


@st.cache_data
def load_backtest_summary() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "backtest_summary.csv")


@st.cache_data
def load_per_horizon_ml() -> pd.DataFrame | None:
    path = RESULTS_DIR / "per_horizon_ml.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def last_close(ticker: str) -> tuple[pd.Timestamp, float]:
    df = load_stock_cached(ticker)
    return df["date"].iloc[-1], float(df["close"].iloc[-1])


def sparkline_window(ticker: str, n_days: int = 90) -> pd.DataFrame:
    df = load_stock_cached(ticker)
    return df.tail(n_days)[["date", "close"]].reset_index(drop=True)
