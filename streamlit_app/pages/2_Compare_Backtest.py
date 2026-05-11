"""Compare models & run single-asset backtest."""
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
from streamlit_app.utils.backtest_runner import run_single
from streamlit_app.utils.components import inject_css, kpi_card, page_header
from streamlit_app.utils.data import (load_backtest_summary, load_dl_summary,
                                       load_ml_summary, load_per_horizon_ml)
from streamlit_app.utils.theme import (ACCENT, INK, MUTED, PALETTE, SEMANTIC,
                                        plotly_template)

st.set_page_config(page_title="Compare & Backtest — StockSight",
                   layout="wide", initial_sidebar_state="expanded")
inject_css()

page_header("Compare & Backtest",
            overline="Cross-model metrics · per-horizon error · single-asset backtest")

tab_cmp, tab_bt = st.tabs(["Model comparison", "Backtest"])

# === Tab 1: Model comparison =================================================
with tab_cmp:
    ml = load_ml_summary()
    dl = load_dl_summary()

    ml_reg = ml[ml["model"].isin(["ridge", "rf_reg", "xgb_reg"])].copy()
    agg_ml = (ml_reg.groupby("model")
              [["test_mae_return", "test_rmse_return",
                "r2_score", "directional_accuracy"]]
              .mean().reset_index())

    dl_mean = pd.DataFrame([{
        "model": "dl_seq2seq",
        "test_mae_return": float(dl["test_mae_price_d1"].mean()) /
                           float(dl["test_mae_price_d1"].mean() + 1.0),
        "test_rmse_return": np.nan,
        "r2_score": np.nan,
        "directional_accuracy": float(dl["test_dir_accuracy_perday"].mean()),
    }])

    # Bar chart — directional accuracy across 4 model groups
    st.markdown("### Cross-model performance")
    bar_df = pd.concat([agg_ml, dl_mean], ignore_index=True)
    fig = go.Figure()
    metrics_to_plot = [("test_mae_return", "MAE (return)"),
                       ("directional_accuracy", "Directional accuracy"),
                       ("r2_score", "R-squared")]
    for i, (col, label) in enumerate(metrics_to_plot):
        fig.add_trace(go.Bar(name=label, x=bar_df["model"],
                              y=bar_df[col].astype(float),
                              marker_color=PALETTE[i]))
    fig.update_layout(template=plotly_template(), barmode="group", height=380,
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch", key="cmp_bar")

    # Per-horizon error
    st.markdown("### Per-horizon forecast error")
    per_horizon_ml = load_per_horizon_ml()
    fig = go.Figure()
    horizons = [1, 7, 14]
    dl_mae = [float(dl[f"test_mae_price_d{h}"].mean()) for h in horizons]
    fig.add_trace(go.Scatter(x=horizons, y=dl_mae, mode="lines+markers",
                              line=dict(color=INK, width=2.0),
                              marker=dict(size=8), name="DL seq2seq"))
    if per_horizon_ml is not None:
        agg = (per_horizon_ml.groupby(["model", "horizon"])["mae_price"]
                             .mean().reset_index())
        for i, m in enumerate(agg["model"].unique()):
            sub = agg[agg["model"] == m].sort_values("horizon")
            fig.add_trace(go.Scatter(x=sub["horizon"], y=sub["mae_price"],
                                      mode="lines+markers",
                                      line=dict(color=PALETTE[i + 1], width=1.4,
                                                dash="dash"),
                                      marker=dict(size=6),
                                      name=m))
    else:
        st.caption("Run `python -m scripts.compute_per_horizon_ml` "
                   "to populate ML per-horizon curves.")
    fig.update_layout(template=plotly_template(), height=380,
                      xaxis_title="Horizon (days)",
                      yaxis_title="MAE (price units, USD)")
    st.plotly_chart(fig, width="stretch", key="per_h")

    # Radar
    st.markdown("### Per-axis profile")
    backtest = load_backtest_summary()
    bt_map = dict(zip(backtest["strategy"], backtest["sharpe"]))
    radar_rows = []
    for m_label, ml_key, bt_key in [("Ridge", "ridge", "ML_ridge"),
                                     ("RF", "rf_reg", "ML_rf_reg"),
                                     ("XGB", "xgb_reg", "ML_xgb_reg"),
                                     ("DL", None, "DL_seq2seq")]:
        if ml_key is None:
            dir_acc = float(dl["test_dir_accuracy_perday"].mean())
            mae_norm = 0.5
            r2 = float(dl.get("test_dir_accuracy_t14", pd.Series([0.5])).mean())
        else:
            sub = ml_reg[ml_reg["model"] == ml_key]
            dir_acc = float(sub["directional_accuracy"].mean())
            mae_norm = 1.0 - float(sub["test_mae_return"].mean()) / \
                       float(ml_reg["test_mae_return"].max())
            r2 = float(sub["r2_score"].mean())
        sharpe = bt_map.get(bt_key, 0.0)
        sharpe_norm = max(0.0, min(1.0, (sharpe + 1.0) / 3.0))
        radar_rows.append({
            "model": m_label,
            "Dir-acc": dir_acc,
            "1-MAE": mae_norm,
            "R-sq": max(0.0, r2),
            "Sharpe": sharpe_norm,
            "Hit-rate": dir_acc,
        })
    axes = ["Dir-acc", "1-MAE", "R-sq", "Sharpe", "Hit-rate"]
    fig = go.Figure()
    for i, row in enumerate(radar_rows):
        fig.add_trace(go.Scatterpolar(
            r=[row[a] for a in axes] + [row[axes[0]]],
            theta=axes + [axes[0]],
            fill="toself", name=row["model"],
            line=dict(color=PALETTE[i], width=1.4),
        ))
    fig.update_layout(template=plotly_template(), height=460,
                      polar=dict(radialaxis=dict(range=[0, 1], visible=True)))
    st.plotly_chart(fig, width="stretch", key="radar")

    # Summary table
    st.markdown("### Summary table")
    summary_view = bar_df.set_index("model")[
        ["test_mae_return", "test_rmse_return", "r2_score", "directional_accuracy"]
    ].round(4)
    st.dataframe(
        summary_view.style.highlight_max(axis=0, props="background-color:#F2EEE6"),
        width="stretch",
    )


# === Tab 2: Backtest =========================================================
with tab_bt:
    st.markdown("### Single-asset backtest")
    c1, c2, c3, c4 = st.columns([1.2, 1.5, 1, 1])
    with c1:
        ticker = st.selectbox("Ticker", CFG["tickers"],
                              index=CFG["tickers"].index(
                                  st.session_state.get("selected_ticker",
                                                       CFG["tickers"][0])))
    with c2:
        model_label = st.selectbox(
            "Model", ["CNN+BiLSTM (DL)", "Ridge", "Random Forest", "XGBoost"])
        model_key = {"CNN+BiLSTM (DL)": "dl", "Ridge": "ridge",
                     "Random Forest": "rf_reg", "XGBoost": "xgb_reg"}[model_label]
    with c3:
        initial = st.number_input("Initial capital", value=10000,
                                   step=1000, min_value=1000)
    with c4:
        cost_bp = st.number_input("Cost (bp)", value=10, step=5, min_value=0)

    if st.button("Run backtest", width="content"):
        with st.spinner(f"Backtesting {ticker} · {model_label}..."):
            try:
                res = run_single(ticker, model_key,
                                  initial_capital=float(initial),
                                  cost_per_trade=cost_bp / 10000)
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                st.stop()

        m = res["metrics"]
        bh = res["bh_metrics"]
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi_card("Total return", f"{m['total_return']*100:+.1f}%",
                     sub=f"B&H {bh['total_return']*100:+.1f}%")
        with k2:
            kpi_card("Sharpe", f"{m['sharpe']:.2f}",
                     sub=f"B&H {bh['sharpe']:.2f}")
        with k3:
            kpi_card("Max drawdown", f"{m['max_dd']*100:.1f}%",
                     sub=f"{m['max_dd_days']}d")
        with k4:
            kpi_card("Hit rate", f"{m['hit_rate']*100:.1f}%",
                     sub=f"turnover {m['avg_turnover']:.2%}/day")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res["equity"].index, y=res["equity"].values,
                                  mode="lines", line=dict(color=INK, width=1.8),
                                  name="Strategy"))
        fig.add_trace(go.Scatter(x=res["benchmark"].index, y=res["benchmark"].values,
                                  mode="lines",
                                  line=dict(color=ACCENT, width=1.4, dash="dash"),
                                  name="Buy & Hold"))
        fig.update_layout(template=plotly_template(), height=380,
                          yaxis_title="Equity (USD)",
                          legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, width="stretch", key="bt_eq")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res["drawdown"].index,
                                  y=res["drawdown"].values * 100,
                                  mode="lines", line=dict(color=SEMANTIC["sell"], width=1),
                                  fill="tozeroy",
                                  fillcolor="rgba(155,44,44,0.25)",
                                  name="Drawdown"))
        fig.update_layout(template=plotly_template(), height=240,
                          yaxis_title="Drawdown (%)",
                          showlegend=False)
        st.plotly_chart(fig, width="stretch", key="bt_dd")

        if not res["trades"].empty:
            st.markdown("#### Trades")
            tt = res["trades"].copy()
            tt["date"] = pd.to_datetime(tt["date"]).dt.strftime("%Y-%m-%d")
            tt["weight"] = tt["weight"].map(lambda v: f"{v:.0%}")
            tt["price"] = tt["price"].map(lambda v: f"${v:,.2f}"
                                           if pd.notna(v) else "—")
            st.dataframe(tt.tail(50), width="stretch", hide_index=True)
    else:
        st.info("Press **Run backtest** to execute on the selected configuration.")
