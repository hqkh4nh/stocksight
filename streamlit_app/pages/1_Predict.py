"""Prediction & Recommendation page — live predict + recommendation card."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import CFG
from streamlit_app.utils.components import inject_css, page_header, recommendation_card
from streamlit_app.utils.data import last_close, load_stock_cached
from streamlit_app.utils.predict import (cum_return, predict_returns,
                                          recommendation)
from streamlit_app.utils.theme import (ACCENT, INK, MUTED, plotly_template)

st.set_page_config(page_title="Predict — StockSight", layout="wide",
                   initial_sidebar_state="expanded")
inject_css()

page_header("Prediction & recommendation",
            overline="Live forecast · 1-14 day horizon · 4 models")

MODEL_OPTIONS = {
    "CNN+BiLSTM seq2seq (DL)": "dl",
    "Ridge": "ridge",
    "Random Forest regressor": "rf_reg",
    "XGBoost regressor": "xgb_reg",
}

with st.sidebar:
    st.markdown("### Forecast inputs")
    default_t = st.session_state.get("selected_ticker", CFG["tickers"][0])
    if default_t not in CFG["tickers"]:
        default_t = CFG["tickers"][0]
    ticker = st.selectbox("Ticker", CFG["tickers"],
                          index=CFG["tickers"].index(default_t))
    model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
    model_name = MODEL_OPTIONS[model_label]
    horizon = st.slider("Horizon (days)", min_value=1, max_value=14, value=14)
    run = st.button("Run prediction", width="stretch")

st.session_state["selected_ticker"] = ticker

if not run:
    st.info("Choose ticker + model on the left, then press **Run prediction**.")
    st.stop()

with st.spinner(f"Predicting {ticker} · {model_label} · {horizon}d..."):
    last_date, close_now = last_close(ticker)
    rets = predict_returns(ticker, model_name, horizon,
                           last_data_date=str(last_date.date()))
    path = close_now * np.cumprod(1.0 + rets)
    cr = cum_return(rets)
    rec = recommendation(cr, threshold=CFG["backtest"]["threshold"])

recommendation_card(rec["action"], rec["cum_return"], horizon, rec["confidence"])

stock = load_stock_cached(ticker)
hist = stock.tail(90)[["date", "close"]].copy()

future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon)
proj_df = pd.DataFrame({"date": future_dates, "close": path})

resid_std = float(np.std(np.diff(np.log(path)))) * np.sqrt(np.arange(1, horizon + 1))
upper = path * (1 + resid_std)
lower = path * (1 - resid_std)

fig = go.Figure()
fig.add_trace(go.Scatter(x=hist["date"], y=hist["close"], mode="lines",
                          line=dict(color=INK, width=1.6),
                          name="Historical close"))
fig.add_trace(go.Scatter(x=proj_df["date"], y=upper, mode="lines",
                          line=dict(color="rgba(180,95,6,0)", width=0),
                          showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=proj_df["date"], y=lower, mode="lines",
                          line=dict(color="rgba(180,95,6,0)", width=0),
                          fill="tonexty", fillcolor="rgba(180,95,6,0.12)",
                          name="±1 σ band"))
fig.add_trace(go.Scatter(x=proj_df["date"], y=proj_df["close"], mode="lines+markers",
                          line=dict(color=ACCENT, width=1.6, dash="dash"),
                          marker=dict(size=5, color=ACCENT),
                          name=f"Forecast ({model_label})"))
fig.add_vline(x=last_date, line=dict(color=MUTED, width=0.8, dash="dot"))
fig.update_layout(template=plotly_template(), height=460,
                  legend=dict(orientation="h", y=1.05))
st.plotly_chart(fig, width="stretch", key="predict_chart")


# === Prediction table ===
st.markdown("### Daily projection")
table = pd.DataFrame({
    "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
    "Price (USD)": [f"${p:,.2f}" for p in path],
    "Direction": ["▲" if r >= 0 else "▼" for r in rets],
    "Return": [f"{r*100:+.2f}%" for r in rets],
    "Cum return": [f"{(np.prod(1 + rets[:i+1]) - 1)*100:+.2f}%"
                   for i in range(len(rets))],
})
st.dataframe(table, width="stretch", hide_index=True)


# === Actions ===
c1, c2 = st.columns([1, 4])
with c1:
    csv = table.to_csv(index=False).encode("utf-8")
    st.download_button("Export CSV", csv,
                       file_name=f"{ticker}_{model_name}_{horizon}d.csv",
                       mime="text/csv", width="stretch")
