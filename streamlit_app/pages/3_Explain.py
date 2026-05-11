"""SHAP explanations for tree-based ML predictions."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import streamlit as st

from src.config import CFG
from streamlit_app.utils.components import inject_css, page_header
from streamlit_app.utils.shap_runner import (compute_shap,
                                              interpret_top_features,
                                              pretty_name)
from streamlit_app.utils.theme import BG, INK

st.set_page_config(page_title="Explain — StockSight",
                   layout="wide", initial_sidebar_state="collapsed")
inject_css()

page_header("Explainability (SHAP)",
            overline="Tree-model attributions · beeswarm · waterfall · feature bar")

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "font.family": "serif",
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.grid": False,
})

c1, c2 = st.columns([1, 2])
with c1:
    model_label = st.radio("Model", ["Random Forest", "XGBoost"],
                            horizontal=True)
    model_key = {"Random Forest": "rf_reg", "XGBoost": "xgb_reg"}[model_label]
with c2:
    default_t = st.session_state.get("selected_ticker", CFG["tickers"][0])
    ticker = st.selectbox("Ticker", CFG["tickers"],
                           index=CFG["tickers"].index(default_t))

with st.spinner(f"Computing SHAP for {ticker} · {model_label}..."):
    try:
        shap_values, X_test, dates = compute_shap(ticker, model_key)
    except FileNotFoundError as e:
        st.error(f"Model file missing: {e}. Run training first.")
        st.stop()

# Date picker for waterfall
date_str = st.select_slider(
    "Test-set date for waterfall",
    options=[d.strftime("%Y-%m-%d") for d in dates],
    value=dates.iloc[-1].strftime("%Y-%m-%d"),
)
date_idx = int(np.where(dates.dt.strftime("%Y-%m-%d") == date_str)[0][0])

# Auto-interpretation
interp = interpret_top_features(shap_values.values[date_idx],
                                 list(X_test.columns))
st.markdown(f'<div class="ss-overline">Reading</div>'
             f'<div style="font-family: Inter, sans-serif; font-size: 0.95rem; '
             f'line-height: 1.5; margin-bottom: 1rem;">{interp}</div>',
             unsafe_allow_html=True)

# Beeswarm
st.markdown("### Beeswarm — feature importance distribution")
fig = plt.figure(figsize=(9, 5))
shap.plots.beeswarm(shap_values, max_display=15, show=False)
plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# Waterfall
st.markdown(f"### Waterfall — single prediction on {date_str}")
display = shap_values[date_idx]
display.feature_names = [pretty_name(f) for f in X_test.columns]
fig = plt.figure(figsize=(9, 5))
shap.plots.waterfall(display, max_display=15, show=False)
plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# Bar
st.markdown("### Mean |SHAP| — global importance")
fig = plt.figure(figsize=(9, 5))
shap.plots.bar(shap_values, max_display=15, show=False)
plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)
