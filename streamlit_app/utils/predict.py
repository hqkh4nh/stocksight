"""Live predict: ML recursive forecast + DL seq2seq inference.

ML scalers persisted by `save_ml_models` as `{ticker}_scaler_X.joblib`.
DL scalers persisted lazily on first load via `_ensure_dl_scaler`.
"""
from __future__ import annotations

from typing import Literal

import joblib
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

from src.config import MODELS_DIR
from src.features import feature_columns
from src.models.dl_model import bounded_return, directional_loss
from src.models.ml_model import recursive_forecast_14
from src.preprocessing import prepare_dl_pipeline
from streamlit_app.utils.data import load_features_cached, macro_df

ModelName = Literal["ridge", "rf_reg", "xgb_reg", "dl"]

ML_MODEL_FILES = {
    "ridge": "{ticker}_ridge.pkl",
    "rf_reg": "{ticker}_rf_reg.pkl",
    "xgb_reg": "{ticker}_xgb_reg.pkl",
}


@st.cache_resource
def _ensure_dl_scaler(ticker: str) -> MinMaxScaler:
    """Load DL scaler from disk, or rebuild it from the train split if missing."""
    path = MODELS_DIR / f"{ticker}_dl_scaler.joblib"
    if path.exists():
        return joblib.load(path)
    dl, _ = prepare_dl_pipeline(ticker, macro_df())
    joblib.dump(dl.scaler_X, path)
    return dl.scaler_X


@st.cache_resource
def _load_ml_model(ticker: str, name: str):
    fname = ML_MODEL_FILES[name].format(ticker=ticker)
    return joblib.load(MODELS_DIR / fname)


@st.cache_resource
def _load_ml_scaler(ticker: str) -> MinMaxScaler:
    return joblib.load(MODELS_DIR / f"{ticker}_scaler_X.joblib")


@st.cache_resource
def _load_dl_model(ticker: str):
    import tensorflow as tf
    return tf.keras.models.load_model(
        MODELS_DIR / f"{ticker}_dl.keras",
        custom_objects={
            "directional_loss": directional_loss,
            "bounded_return": bounded_return,
        },
        compile=False,
    )


def _predict_ml_returns(ticker: str, model_name: str, horizon: int) -> np.ndarray:
    feats = load_features_cached(ticker)
    feat_cols = feature_columns(ticker)
    last_row = feats[feat_cols].iloc[-1].values.astype(float)
    model = _load_ml_model(ticker, model_name)
    scaler = _load_ml_scaler(ticker)
    return recursive_forecast_14(model, last_row, feat_cols=feat_cols,
                                 scaler_X=scaler, horizon=horizon)


def _predict_dl_returns(ticker: str, horizon: int) -> np.ndarray:
    feats = load_features_cached(ticker)
    feat_cols = feature_columns(ticker)
    window = 60
    if len(feats) < window:
        raise ValueError(f"Not enough rows for {ticker}: have {len(feats)}, need {window}")
    last_window = feats[feat_cols].iloc[-window:].values.astype(np.float32)
    scaler = _ensure_dl_scaler(ticker)
    X = scaler.transform(last_window).reshape(1, window, -1)
    model = _load_dl_model(ticker)
    pred = model.predict(X, verbose=0)  # (1, 14, 1)
    returns = pred.reshape(-1)[:horizon]
    return returns


@st.cache_data
def predict_returns(ticker: str, model_name: ModelName, horizon: int,
                    last_data_date: str) -> np.ndarray:
    """Cache key includes last_data_date so fresh data invalidates entries."""
    if model_name == "dl":
        return _predict_dl_returns(ticker, horizon)
    return _predict_ml_returns(ticker, model_name, horizon)


def predict_path(ticker: str, model_name: ModelName, horizon: int = 14) -> np.ndarray:
    """Return predicted PRICE path (length=horizon) from latest close."""
    from streamlit_app.utils.data import last_close
    last_date, close = last_close(ticker)
    rets = predict_returns(ticker, model_name, horizon,
                           last_data_date=str(last_date.date()))
    path = close * np.cumprod(1.0 + rets)
    return path


def cum_return(returns: np.ndarray) -> float:
    return float(np.prod(1.0 + returns) - 1.0)


def recommendation(cum_ret: float, threshold: float) -> dict:
    """Map cumulative return → BUY / HOLD / SELL + confidence in [0, 1]."""
    if cum_ret > threshold:
        action = "BUY"
    elif cum_ret < -threshold:
        action = "SELL"
    else:
        action = "HOLD"
    confidence = float(min(1.0, abs(cum_ret) / (3.0 * threshold))) if threshold > 0 else 0.0
    return {"action": action, "cum_return": cum_ret, "confidence": confidence}
