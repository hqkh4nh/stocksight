"""Reusable HTML/Streamlit components matching the handcrafted theme."""
from __future__ import annotations

from pathlib import Path

import streamlit as st


_ASSETS = Path(__file__).resolve().parent.parent / "assets"


def inject_css() -> None:
    css = (_ASSETS / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def page_header(title: str, overline: str | None = None) -> None:
    if overline:
        st.markdown(f'<div class="ss-overline">{overline}</div>',
                    unsafe_allow_html=True)
    st.markdown(f"# {title}")


def kpi_card(label: str, value: str, sub: str | None = None) -> None:
    html = (f'<div class="ss-kpi"><div class="label">{label}</div>'
            f'<div class="value">{value}</div>')
    if sub:
        html += f'<div class="sub">{sub}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def recommendation_card(action: str, cum_return: float, horizon: int,
                        confidence: float) -> None:
    cls = action.lower()
    pct = f"{cum_return*100:+.2f}%"
    conf = f"{confidence*100:.0f}%"
    html = (
        f'<div class="ss-rec">'
        f'<div class="badge {cls}">{action}</div>'
        f'<div class="meta">Predicted <span class="num">{pct}</span> over '
        f'<span class="num">{horizon}</span> days · '
        f'<span class="num">{conf}</span> confidence</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)
