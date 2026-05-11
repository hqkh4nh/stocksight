"""Plotly template, palette, and number formatters for the redesigned theme."""
from __future__ import annotations

BG = "#FAFAF8"
PAPER = "#F4F2EE"
INK = "#1F2937"
ACCENT = "#C2410C"
HAIRLINE = "#E5E7EB"
MUTED = "#6B7280"

PALETTE = [
    "#1F2937", "#C2410C", "#2F6F4E", "#7A6BBA",
    "#9B2C2C", "#A2845E", "#3D6A91", "#7E8C5D",
]

SEMANTIC = {
    "buy": "#2F6F4E",
    "hold": "#A2845E",
    "sell": "#9B2C2C",
    "neutral": MUTED,
}

_FONT = "Be Vietnam Pro, system-ui, sans-serif"


def plotly_template() -> dict:
    return {
        "layout": {
            "paper_bgcolor": BG,
            "plot_bgcolor": BG,
            "font": {"family": _FONT, "color": INK, "size": 12},
            "title": {"font": {"family": _FONT, "size": 15, "color": INK},
                       "x": 0.0, "xanchor": "left"},
            "colorway": PALETTE,
            "xaxis": {
                "gridcolor": HAIRLINE,
                "zeroline": False,
                "linecolor": HAIRLINE,
                "tickfont": {"family": _FONT, "size": 11},
            },
            "yaxis": {
                "gridcolor": HAIRLINE,
                "zeroline": False,
                "linecolor": HAIRLINE,
                "tickfont": {"family": _FONT, "size": 11},
            },
            "legend": {
                "bgcolor": "rgba(0,0,0,0)",
                "font": {"family": _FONT, "size": 11},
                "borderwidth": 0,
            },
            "margin": {"l": 50, "r": 30, "t": 40, "b": 40},
            "hoverlabel": {
                "bgcolor": BG,
                "bordercolor": INK,
                "font": {"family": _FONT, "color": INK},
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
