"""Plotly template, palette, and number formatters for the handcrafted theme."""
from __future__ import annotations

BG = "#FBFAF7"
PAPER = "#F2EEE6"
INK = "#13294B"
ACCENT = "#B45F06"
HAIRLINE = "#E5E0D9"
MUTED = "#6B7280"

PALETTE = [
    "#13294B", "#B45F06", "#2F6F4E", "#7A6BBA",
    "#9B2C2C", "#A2845E", "#3D6A91", "#7E8C5D",
]

SEMANTIC = {
    "buy": "#2F6F4E",
    "hold": "#A2845E",
    "sell": "#9B2C2C",
    "neutral": MUTED,
}


def plotly_template() -> dict:
    return {
        "layout": {
            "paper_bgcolor": BG,
            "plot_bgcolor": BG,
            "font": {"family": "Inter, sans-serif", "color": INK, "size": 12},
            "title": {"font": {"family": "Source Serif 4, serif",
                               "size": 16, "color": INK}, "x": 0.0, "xanchor": "left"},
            "colorway": PALETTE,
            "xaxis": {
                "gridcolor": HAIRLINE,
                "zeroline": False,
                "linecolor": HAIRLINE,
                "tickfont": {"family": "JetBrains Mono, monospace", "size": 11},
            },
            "yaxis": {
                "gridcolor": HAIRLINE,
                "zeroline": False,
                "linecolor": HAIRLINE,
                "tickfont": {"family": "JetBrains Mono, monospace", "size": 11},
            },
            "legend": {
                "bgcolor": "rgba(0,0,0,0)",
                "font": {"family": "Inter, sans-serif", "size": 11},
                "borderwidth": 0,
            },
            "margin": {"l": 50, "r": 30, "t": 40, "b": 40},
            "hoverlabel": {
                "bgcolor": BG,
                "bordercolor": INK,
                "font": {"family": "JetBrains Mono, monospace", "color": INK},
            },
        }
    }


def fmt_pct(x: float, digits: int = 2) -> str:
    if x is None:
        return "—"
    return f"{x*100:.{digits}f}%"


def fmt_signed_pct(x: float, digits: int = 2) -> str:
    if x is None:
        return "—"
    sign = "+" if x >= 0 else "−"
    return f"{sign}{abs(x)*100:.{digits}f}%"


def fmt_money(x: float, digits: int = 2) -> str:
    if x is None:
        return "—"
    return f"${x:,.{digits}f}"


def fmt_num(x: float, digits: int = 2) -> str:
    if x is None:
        return "—"
    return f"{x:,.{digits}f}"
