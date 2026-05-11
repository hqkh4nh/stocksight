"""StockSight — Overview page.

KPI strip · 21-ticker sparkline grid grouped by sector · technical tabs for selected ticker.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make project root importable when streamlit launches from anywhere
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import CFG
from src.features import FEATURE_COLUMNS_BASE, MACRO_COLUMNS
from src.sector_config import TICKER_SECTOR
from streamlit_app.utils.components import inject_css, kpi_card, page_header
from streamlit_app.utils.data import (load_dl_summary, load_features_cached,
                                      load_ml_summary, load_run_metadata,
                                      load_stock_cached, sparkline_window)
from streamlit_app.utils.theme import (ACCENT, BG, HAIRLINE, INK, MUTED,
                                        SEMANTIC, plotly_template)

st.set_page_config(page_title="StockSight", layout="wide",
                   initial_sidebar_state="collapsed")
inject_css()

SECTOR_ORDER = ["energy", "industrial", "consumer_disc", "consumer_staples",
                "financial", "utility", "healthcare", "tech_media"]
SECTOR_LABEL = {
    "energy": "Energy", "industrial": "Industrials",
    "consumer_disc": "Consumer Discretionary",
    "consumer_staples": "Consumer Staples",
    "financial": "Financials", "utility": "Utilities",
    "healthcare": "Healthcare", "tech_media": "Tech & Media",
}

if "selected_ticker" not in st.session_state:
    st.session_state["selected_ticker"] = CFG["tickers"][0]


# === Header ===
meta = load_run_metadata()
dl_df = load_dl_summary()
avg_dir = float(dl_df["test_dir_accuracy_perday"].mean()) if "test_dir_accuracy_perday" in dl_df.columns else 0.0
ended_at = meta.get("ended_at", "—")
n_tickers = len(CFG["tickers"])

overline = (f"As of {ended_at}  ·  {n_tickers} symbols  ·  "
            f"{avg_dir*100:.1f}% mean directional accuracy")
page_header("StockSight", overline=overline)


# === KPI strip ===
ml_df = load_ml_summary()
best_dl_row = dl_df.loc[dl_df["test_dir_accuracy_perday"].idxmax()] \
    if "test_dir_accuracy_perday" in dl_df.columns and len(dl_df) else None

cols = st.columns(4)
with cols[0]:
    kpi_card("Universe", str(n_tickers), sub="tickers covered")
with cols[1]:
    if best_dl_row is not None:
        kpi_card("Best DL ticker", str(best_dl_row["ticker"]),
                 sub=f"{best_dl_row['test_dir_accuracy_perday']*100:.1f}% dir-acc")
    else:
        kpi_card("Best DL ticker", "—")
with cols[2]:
    kpi_card("Mean dir-acc", f"{avg_dir*100:.1f}%",
             sub="DL seq2seq, per-day")
with cols[3]:
    kpi_card("Last update", ended_at.split("T")[0] if "T" in ended_at else ended_at)


# === Sparkline grid ===
st.markdown("## Universe")

dir_acc_by_ticker = dict(zip(dl_df["ticker"], dl_df["test_dir_accuracy_perday"])) \
    if "test_dir_accuracy_perday" in dl_df.columns else {}


def _spark_fig(prices: pd.Series, last_close: float, first_close: float) -> go.Figure:
    color = SEMANTIC["buy"] if last_close >= first_close else SEMANTIC["sell"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(prices))), y=prices.values,
                              mode="lines", line=dict(color=color, width=1.3),
                              fill="tozeroy",
                              fillcolor=color.replace(")", ",0.06)").replace("rgb", "rgba")
                                       if color.startswith("rgb") else color + "10",
                              hoverinfo="skip"))
    fig.update_layout(template=plotly_template(),
                      margin=dict(l=0, r=0, t=0, b=0),
                      height=44, showlegend=False,
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


for sector in SECTOR_ORDER:
    tickers_in_sector = [t for t, s in TICKER_SECTOR.items() if s == sector]
    if not tickers_in_sector:
        continue
    st.markdown(f'<div class="ss-overline" style="margin-top:0.8rem">'
                f'{SECTOR_LABEL[sector]}</div>', unsafe_allow_html=True)
    cols = st.columns(min(len(tickers_in_sector), 6))
    for idx, t in enumerate(tickers_in_sector):
        with cols[idx % len(cols)]:
            spark = sparkline_window(t, n_days=90)
            first_p = float(spark["close"].iloc[0])
            last_p = float(spark["close"].iloc[-1])
            delta = (last_p / first_p - 1.0) if first_p > 0 else 0.0
            delta_cls = "delta-pos" if delta >= 0 else "delta-neg"
            delta_sym = "+" if delta >= 0 else "−"
            dir_pct = dir_acc_by_ticker.get(t)
            dir_str = f"{dir_pct*100:.0f}%" if dir_pct is not None else "—"
            st.markdown(
                f'<div class="ss-spark">'
                f'<div class="tkr">{t}</div>'
                f'<div class="px">${last_p:,.2f}</div>'
                f'<div class="{delta_cls}">{delta_sym}{abs(delta)*100:.1f}% · 90d · '
                f'dir {dir_str}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(_spark_fig(spark["close"], last_p, first_p),
                            use_container_width=True,
                            config={"displayModeBar": False},
                            key=f"spark_{t}")
            if st.button("Select", key=f"sel_{t}",
                         use_container_width=True):
                st.session_state["selected_ticker"] = t
                st.rerun()


# === Per-ticker analysis ===
sel = st.session_state["selected_ticker"]
st.markdown(f"## {sel} · technical view")

stock = load_stock_cached(sel)
feats = load_features_cached(sel)

tab_c, tab_m, tab_b, tab_h = st.tabs(["Candlestick + Volume", "Momentum (RSI / MACD)",
                                      "Bollinger", "Correlation"])

with tab_c:
    df1y = stock.tail(252).copy()
    df1y["ma_5"] = df1y["close"].rolling(5).mean()
    df1y["ma_20"] = df1y["close"].rolling(20).mean()
    df1y["ma_50"] = df1y["close"].rolling(50).mean()

    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df1y["date"], open=df1y["open"], high=df1y["high"],
                                  low=df1y["low"], close=df1y["close"],
                                  increasing_line_color=SEMANTIC["buy"],
                                  decreasing_line_color=SEMANTIC["sell"],
                                  increasing_fillcolor=SEMANTIC["buy"],
                                  decreasing_fillcolor=SEMANTIC["sell"],
                                  name="OHLC", showlegend=False),
                  row=1, col=1)
    for col, color, name in [("ma_5", "#B45F06", "MA 5"),
                             ("ma_20", "#3D6A91", "MA 20"),
                             ("ma_50", "#7A6BBA", "MA 50")]:
        fig.add_trace(go.Scatter(x=df1y["date"], y=df1y[col], mode="lines",
                                  line=dict(color=color, width=1.2),
                                  name=name), row=1, col=1)
    fig.add_trace(go.Bar(x=df1y["date"], y=df1y["volume"],
                          marker_color="#A8A39A", showlegend=False,
                          name="Volume"), row=2, col=1)
    fig.update_layout(template=plotly_template(), height=520,
                      xaxis_rangeslider_visible=False,
                      margin=dict(l=40, r=20, t=20, b=30))
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True, key="candle")

with tab_m:
    df1y = feats.tail(252).copy()
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=df1y["date"], y=df1y["rsi_14"], mode="lines",
                              line=dict(color=INK, width=1.4),
                              name="RSI 14"), row=1, col=1)
    fig.add_hrect(y0=30, y1=70, line_width=0,
                  fillcolor="#F2EEE6", opacity=0.5, row=1, col=1)
    fig.add_hline(y=70, line=dict(color=MUTED, width=0.8, dash="dot"), row=1, col=1)
    fig.add_hline(y=30, line=dict(color=MUTED, width=0.8, dash="dot"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df1y["date"], y=df1y["macd"], mode="lines",
                              line=dict(color=INK, width=1.4), name="MACD"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=df1y["date"], y=df1y["macd_signal"], mode="lines",
                              line=dict(color=ACCENT, width=1.2, dash="dash"),
                              name="Signal"), row=2, col=1)
    fig.update_layout(template=plotly_template(), height=520)
    fig.update_yaxes(title_text="RSI", row=1, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True, key="momentum")

with tab_b:
    df1y = feats.tail(252).copy()
    df1y["bb_mid"] = df1y["close"].rolling(20).mean()
    df1y["bb_std"] = df1y["close"].rolling(20).std()
    df1y["bb_upper"] = df1y["bb_mid"] + 2 * df1y["bb_std"]
    df1y["bb_lower"] = df1y["bb_mid"] - 2 * df1y["bb_std"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df1y["date"], y=df1y["bb_upper"], mode="lines",
                              line=dict(color=MUTED, width=0.8), name="Upper",
                              showlegend=False))
    fig.add_trace(go.Scatter(x=df1y["date"], y=df1y["bb_lower"], mode="lines",
                              line=dict(color=MUTED, width=0.8), name="Lower",
                              fill="tonexty", fillcolor="rgba(180,95,6,0.08)",
                              showlegend=False))
    fig.add_trace(go.Scatter(x=df1y["date"], y=df1y["bb_mid"], mode="lines",
                              line=dict(color=ACCENT, width=1.0, dash="dash"),
                              name="Mid (MA20)"))
    fig.add_trace(go.Scatter(x=df1y["date"], y=df1y["close"], mode="lines",
                              line=dict(color=INK, width=1.5), name="Close"))
    fig.update_layout(template=plotly_template(), height=460)
    st.plotly_chart(fig, use_container_width=True, key="bb")

with tab_h:
    base = [c for c in FEATURE_COLUMNS_BASE if c in feats.columns]
    macro = [c for c in MACRO_COLUMNS if c in feats.columns]
    cols = base + macro
    corr = feats[cols].corr().round(2)

    annotations = []
    for i, r in enumerate(corr.index):
        for j, c in enumerate(corr.columns):
            v = corr.iloc[i, j]
            if abs(v) > 0.5 and i != j:
                annotations.append(dict(x=c, y=r, text=f"{v:.2f}",
                                         showarrow=False,
                                         font=dict(size=9, color=INK,
                                                   family="JetBrains Mono")))
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
                                     colorscale=[[0.0, "#9B2C2C"], [0.5, BG],
                                                  [1.0, "#13294B"]],
                                     zmid=0, zmin=-1, zmax=1,
                                     showscale=True))
    fig.update_layout(template=plotly_template(), height=540,
                      annotations=annotations,
                      xaxis=dict(tickangle=-45))
    st.plotly_chart(fig, use_container_width=True, key="corr")
