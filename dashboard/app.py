"""
AusClimate Dashboard -- dashboard/app.py
Real-Time Australian Climate Analytics Platform
University of Technology Sydney -- Capstone 2026
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from config import (
    PROCESSED_DATA_PATH,
    MERGED_DATA_PATH,
    STATION_SUMMARY_PATH,
    BARRA2_CACHE_PATH,
    AEMO_REGION_LABELS,
)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AusClimate",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME SYSTEM
# Two fully defined themes. All legacy variable names (PLOT_BG, COLOUR_SCALE)
# are derived from the theme so existing chart code needs zero changes.
# ─────────────────────────────────────────────────────────────────────────────

THEMES = {
    "dark": {
        "bg_primary":     "#0D1117",
        "bg_card":        "#161B22",
        "bg_sidebar":     "#0D1117",
        "bg_input":       "#1C2128",
        "text_primary":   "#E6EDF3",
        "text_secondary": "#7D8590",
        "text_muted":     "#3D444D",
        "border":         "#21262D",
        "border_light":   "#30363D",
        "accent":         "#1F6FEB",
        "accent_light":   "#388BFD",
        "success":        "#3FB950",
        "warning":        "#D29922",
        "critical":       "#F85149",
        "plot_bg":        "rgba(0,0,0,0)",
        "paper_bg":       "rgba(0,0,0,0)",
        "plotly_tmpl":    "plotly_dark",
        "grid_color":     "rgba(255,255,255,0.05)",
        "axis_color":     "rgba(255,255,255,0.08)",
        "mapbox_style":   "carto-darkmatter",
    },
    "light": {
        "bg_primary":     "#F6F8FA",
        "bg_card":        "#FFFFFF",
        "bg_sidebar":     "#F6F8FA",
        "bg_input":       "#F0F3F8",
        "text_primary":   "#1F2328",
        "text_secondary": "#636C76",
        "text_muted":     "#A0AAB4",
        "border":         "#D1D9E0",
        "border_light":   "#E6EBF0",
        "accent":         "#0969DA",
        "accent_light":   "#1B79E5",
        "success":        "#1A7F37",
        "warning":        "#9A6700",
        "critical":       "#CF222E",
        "plot_bg":        "rgba(0,0,0,0)",
        "paper_bg":       "rgba(0,0,0,0)",
        "plotly_tmpl":    "plotly_white",
        "grid_color":     "rgba(0,0,0,0.04)",
        "axis_color":     "rgba(0,0,0,0.06)",
        "mapbox_style":   "carto-positron",
    },
}

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

T = THEMES[st.session_state.theme]

# Legacy variable names -- keep all existing chart code working unchanged
PLOT_BG           = T["plot_bg"]
PAPER_BG          = T["paper_bg"]
COLOUR_SCALE      = "RdYlBu_r"
COLOUR_SCALE_TEMP = "RdYlBu_r"
COLOUR_SCALE_DIV  = "RdBu_r"
COLOUR_SCALE_SEQ  = "YlOrRd"
COLOUR_SCALE_BLUE = "Blues"


# ── CSS ───────────────────────────────────────────────────────────────────────

def build_css(T: dict) -> str:
    is_dark = T["bg_primary"].startswith("#0") or T["bg_primary"].startswith("#1")
    shadow  = "0 1px 3px rgba(0,0,0,0.4), 0 4px 12px rgba(0,0,0,0.3)" \
        if is_dark else "0 1px 3px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.06)"
    shadow_hover = "0 2px 8px rgba(0,0,0,0.5), 0 8px 24px rgba(0,0,0,0.35)" \
        if is_dark else "0 2px 8px rgba(0,0,0,0.12), 0 8px 24px rgba(0,0,0,0.08)"
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset ──────────────────────────────────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}
#MainMenu, footer, header {{ visibility: hidden; }}

/* ── Top accent bar ─────────────────────────────────────────── */
.stApp::before {{
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg,
        {T["accent"]} 0%,
        {T["accent_light"]} 50%,
        {T["success"]} 100%);
    z-index: 9999;
}}

/* ── App background ─────────────────────────────────────────── */
.stApp {{
    background-color: {T["bg_primary"]} !important;
}}
div[data-testid="stMainBlockContainer"] {{
    padding-top: 24px !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}}

/* ── Sidebar ────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background-color: {T["bg_sidebar"]} !important;
    border-right: 1px solid {T["border"]} !important;
    box-shadow: 2px 0 12px rgba(0,0,0,0.15) !important;
}}
section[data-testid="stSidebar"] > div {{
    padding-top: 16px !important;
}}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown p {{
    color: {T["text_secondary"]} !important;
    font-size: 12px !important;
}}
section[data-testid="stSidebar"] .stMultiSelect span,
section[data-testid="stSidebar"] .stSelectbox span {{
    color: {T["text_primary"]} !important;
    font-size: 12px !important;
}}

/* ── Metric cards ───────────────────────────────────────────── */
div[data-testid="metric-container"] {{
    background-color: {T["bg_card"]} !important;
    border: 1px solid {T["border"]} !important;
    border-left: 3px solid {T["accent"]} !important;
    border-radius: 8px !important;
    padding: 16px 20px 14px !important;
    box-shadow: {shadow} !important;
    transition: all 0.2s ease !important;
    position: relative;
    overflow: hidden;
}}
div[data-testid="metric-container"]:hover {{
    border-left-color: {T["accent_light"]} !important;
    box-shadow: {shadow_hover} !important;
    transform: translateY(-1px);
}}
div[data-testid="metric-container"] label {{
    color: {T["text_secondary"]} !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    margin-bottom: 6px !important;
}}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {T["text_primary"]} !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    line-height: 1.1 !important;
}}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {{
    font-size: 11px !important;
    font-weight: 500 !important;
    color: {T["text_secondary"]} !important;
    margin-top: 4px !important;
}}

/* ── Tabs ───────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background-color: {T["bg_card"]} !important;
    border: 1px solid {T["border"]} !important;
    border-bottom: none !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 6px 6px 0 !important;
    gap: 2px !important;
    box-shadow: {shadow} !important;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    border-radius: 6px 6px 0 0 !important;
    color: {T["text_secondary"]} !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 9px 16px !important;
    border: none !important;
    transition: all 0.18s ease !important;
    letter-spacing: 0.01em !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {T["text_primary"]} !important;
    background-color: {T["border"]}60 !important;
}}
.stTabs [aria-selected="true"] {{
    color: {T["accent_light"]} !important;
    font-weight: 700 !important;
    background-color: {T["bg_primary"]} !important;
    border-bottom: 3px solid {T["accent"]} !important;
}}
.stTabs [data-baseweb="tab-panel"] {{
    background-color: {T["bg_primary"]} !important;
    border: 1px solid {T["border"]} !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
    padding: 24px 20px !important;
}}

/* ── Buttons ────────────────────────────────────────────────── */
.stButton > button {{
    background-color: {T["bg_card"]} !important;
    border: 1px solid {T["border"]} !important;
    color: {T["text_secondary"]} !important;
    border-radius: 6px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 7px 16px !important;
    transition: all 0.18s ease !important;
    letter-spacing: 0.01em !important;
    box-shadow: {shadow} !important;
}}
.stButton > button:hover {{
    background-color: {T["accent"]} !important;
    border-color: {T["accent"]} !important;
    color: #ffffff !important;
    box-shadow: 0 4px 12px {T["accent"]}40 !important;
    transform: translateY(-1px) !important;
}}
.stButton > button:active {{
    transform: translateY(0) !important;
}}

/* ── Inputs ─────────────────────────────────────────────────── */
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div {{
    background-color: {T["bg_input"]} !important;
    border: 1px solid {T["border"]} !important;
    border-radius: 6px !important;
    color: {T["text_primary"]} !important;
    font-size: 12px !important;
    transition: border-color 0.15s !important;
}}
.stSelectbox [data-baseweb="select"] > div:focus-within,
.stMultiSelect [data-baseweb="select"] > div:focus-within {{
    border-color: {T["accent"]} !important;
    box-shadow: 0 0 0 2px {T["accent"]}25 !important;
}}
.stRadio [data-baseweb="radio"] {{
    gap: 6px !important;
}}
.stRadio label,
.stRadio [data-testid="stWidgetLabel"] {{
    color: {T["text_secondary"]} !important;
    font-size: 11px !important;
}}

/* ── DataFrames ─────────────────────────────────────────────── */
.stDataFrame {{
    border: 1px solid {T["border"]} !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    box-shadow: {shadow} !important;
}}
.stDataFrame th {{
    background-color: {T["bg_input"]} !important;
    color: {T["text_secondary"]} !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    padding: 10px 12px !important;
    border-bottom: 1px solid {T["border"]} !important;
}}
.stDataFrame td {{
    font-size: 12px !important;
    color: {T["text_primary"]} !important;
    padding: 8px 12px !important;
    border-bottom: 1px solid {T["border"]}50 !important;
}}

/* ── Alerts ─────────────────────────────────────────────────── */
div[data-testid="stNotification"],
.stAlert {{
    border-radius: 7px !important;
    border-left-width: 4px !important;
    font-size: 13px !important;
}}
.stSuccess {{
    background-color: {T["success"]}12 !important;
    border-left: 4px solid {T["success"]} !important;
    color: {T["success"]} !important;
}}
.stInfo {{
    background-color: {T["accent"]}12 !important;
    border-left: 4px solid {T["accent"]} !important;
    color: {T["accent_light"]} !important;
}}
.stWarning {{
    background-color: {T["warning"]}12 !important;
    border-left: 4px solid {T["warning"]} !important;
    color: {T["warning"]} !important;
}}
.stError {{
    background-color: {T["critical"]}12 !important;
    border-left: 4px solid {T["critical"]} !important;
    color: {T["critical"]} !important;
}}

/* ── Dividers ───────────────────────────────────────────────── */
hr {{
    border: none !important;
    border-top: 1px solid {T["border"]} !important;
    margin: 20px 0 !important;
}}

/* ── Download button ────────────────────────────────────────── */
.stDownloadButton > button {{
    background-color: {T["accent"]} !important;
    border: 1px solid {T["accent"]} !important;
    color: #fff !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 2px 8px {T["accent"]}40 !important;
}}
.stDownloadButton > button:hover {{
    background-color: {T["accent_light"]} !important;
    border-color: {T["accent_light"]} !important;
    transform: translateY(-1px) !important;
}}

/* ── Spinner ────────────────────────────────────────────────── */
.stSpinner > div {{ border-top-color: {T["accent"]} !important; }}

/* ── Chat ───────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {{
    background-color: {T["bg_card"]} !important;
    border: 1px solid {T["border"]} !important;
    border-radius: 8px !important;
    padding: 14px 18px !important;
    margin-bottom: 10px !important;
    box-shadow: {shadow} !important;
}}

/* ── Scrollbar ──────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{
    background: {T["border"]};
    border-radius: 4px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: {T["accent"]};
}}

/* ── Expander ───────────────────────────────────────────────── */
.streamlit-expanderHeader {{
    background-color: {T["bg_card"]} !important;
    border: 1px solid {T["border"]} !important;
    border-radius: 6px !important;
    color: {T["text_secondary"]} !important;
    font-size: 12px !important;
    font-weight: 500 !important;
}}

/* ── Text visibility (critical for light mode) ──────────── */
h1, h2, h3, h4, h5, h6 {{
    color: {T["text_primary"]} !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}}
p, span, label, div {{
    color: {T["text_primary"]};
}}
.stMarkdown, .stMarkdown p, .stMarkdown span {{
    color: {T["text_primary"]} !important;
}}
.stSubheader, [data-testid="stHeading"] {{
    color: {T["text_primary"]} !important;
}}
.stCaption, [data-testid="stCaptionContainer"] p {{
    color: {T["text_muted"]} !important;
    font-size: 11px !important;
}}
/* Tab panel text */
.stTabs [data-baseweb="tab-panel"] p,
.stTabs [data-baseweb="tab-panel"] span,
.stTabs [data-baseweb="tab-panel"] label {{
    color: {T["text_primary"]} !important;
}}
/* Radio and checkbox labels */
.stRadio label span,
.stCheckbox label span {{
    color: {T["text_primary"]} !important;
}}
/* Selectbox text */
.stSelectbox [data-baseweb="select"] span,
.stMultiSelect [data-baseweb="select"] span {{
    color: {T["text_primary"]} !important;
}}
/* Success/info/warning text */
.stSuccess p, .stSuccess span {{ color: {T["success"]} !important; }}
.stInfo p, .stInfo span {{ color: {T["accent_light"]} !important; }}
.stWarning p, .stWarning span {{ color: {T["warning"]} !important; }}
.stError p, .stError span {{ color: {T["critical"]} !important; }}

/* ──────────────────────────────────────────────────────────── */
/* CUSTOM COMPONENTS                                           */
/* ──────────────────────────────────────────────────────────── */

/* Operational header */
.aus-header {{
    background: linear-gradient(135deg,
        {T["bg_card"]} 0%,
        {T["bg_input"]} 100%);
    border: 1px solid {T["border"]};
    border-radius: 10px;
    padding: 18px 24px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 14px;
    box-shadow: {shadow};
    position: relative;
    overflow: hidden;
}}
.aus-header::after {{
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg,
        {T["accent"]}, {T["success"]}, {T["accent_light"]});
    opacity: 0.6;
}}
.aus-header-left {{
    display: flex;
    align-items: center;
    gap: 16px;
}}
.aus-header-icon {{
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, {T["accent"]}, {T["accent_light"]});
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
    box-shadow: 0 4px 12px {T["accent"]}40;
}}
.aus-header-title {{
    font-size: 20px;
    font-weight: 800;
    color: {T["text_primary"]};
    letter-spacing: -0.03em;
    line-height: 1;
}}
.aus-header-sub {{
    font-size: 11px;
    color: {T["text_secondary"]};
    margin-top: 3px;
    letter-spacing: 0.02em;
}}
.aus-header-right {{
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}}
.aus-timestamp {{
    font-size: 10px;
    color: {T["text_muted"]};
    letter-spacing: 0.05em;
    font-variant-numeric: tabular-nums;
    margin-left: 4px;
}}

/* Badges */
.aus-badge {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    border-radius: 99px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    white-space: nowrap;
    transition: all 0.15s;
}}
.aus-badge-live {{
    background: {T["success"]}20;
    color: {T["success"]};
    border: 1px solid {T["success"]}50;
}}
.aus-badge-live::before {{
    content: '';
    width: 6px; height: 6px;
    background: {T["success"]};
    border-radius: 50%;
    display: inline-block;
    animation: pulse-green 2s infinite;
}}
.aus-badge-pending {{
    background: {T["warning"]}20;
    color: {T["warning"]};
    border: 1px solid {T["warning"]}50;
}}
.aus-badge-pending::before {{
    content: '';
    width: 6px; height: 6px;
    background: {T["warning"]};
    border-radius: 50%;
    display: inline-block;
}}
.aus-badge-offline {{
    background: {T["text_muted"]}15;
    color: {T["text_muted"]};
    border: 1px solid {T["text_muted"]}30;
}}
.aus-badge-offline::before {{
    content: '';
    width: 6px; height: 6px;
    background: {T["text_muted"]};
    border-radius: 50%;
    display: inline-block;
}}

@keyframes pulse-green {{
    0%, 100% {{ opacity: 1; box-shadow: 0 0 0 0 {T["success"]}60; }}
    50% {{ opacity: 0.8; box-shadow: 0 0 0 4px {T["success"]}00; }}
}}

/* Sidebar brand */
.aus-sidebar-brand {{
    padding: 8px 4px 16px;
    border-bottom: 1px solid {T["border"]};
    margin-bottom: 16px;
}}
.aus-sidebar-brand-title {{
    font-size: 16px;
    font-weight: 800;
    color: {T["text_primary"]};
    letter-spacing: -0.03em;
    line-height: 1;
}}
.aus-sidebar-brand-dot {{
    display: inline-block;
    width: 7px; height: 7px;
    background: {T["accent"]};
    border-radius: 50%;
    margin-left: 2px;
    vertical-align: super;
    font-size: 6px;
}}
.aus-sidebar-brand-sub {{
    font-size: 9px;
    color: {T["text_muted"]};
    letter-spacing: 0.1em;
    margin-top: 3px;
    text-transform: uppercase;
    font-weight: 500;
}}

/* Section labels */
.aus-section-label {{
    font-size: 9px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: {T["text_muted"]};
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid {T["border"]};
}}

/* Source rows in sidebar */
.aus-src-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid {T["border"]}50;
}}
.aus-src-row:last-child {{ border-bottom: none; }}
.aus-src-label {{
    font-size: 12px;
    color: {T["text_secondary"]};
    font-weight: 500;
}}
.aus-src-wrap {{
    margin-bottom: 12px;
    background: {T["bg_input"]};
    border: 1px solid {T["border"]};
    border-radius: 8px;
    padding: 6px 12px;
}}

/* Info cards for tab content */
.aus-info-card {{
    background: {T["bg_card"]};
    border: 1px solid {T["border"]};
    border-radius: 8px;
    padding: 20px 24px;
    margin: 12px 0;
    box-shadow: {shadow};
}}
</style>
"""

st.markdown(build_css(T), unsafe_allow_html=True)


# ── Chart layout helper ───────────────────────────────────────────────────────

def chart_layout(fig,
                 title: str = "",
                 height: int = 400,
                 xangle: int = 0,
                 legend: bool = False,
                 margin: dict = None) -> object:
    """Apply consistent professional layout to any plotly figure."""
    if margin is None:
        margin = dict(t=44, b=36, l=44, r=16)
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=13, color=T["text_primary"],
                      family="Inter, -apple-system, sans-serif"),
            x=0, xanchor="left",
            pad=dict(l=0, t=0),
        ),
        plot_bgcolor=T["plot_bg"],
        paper_bgcolor=T["paper_bg"],
        font=dict(
            family="Inter, -apple-system, sans-serif",
            size=11,
            color=T["text_primary"],      # changed from text_secondary
        ),
        height=height,
        margin=margin,
        showlegend=legend,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=T["border"],
            borderwidth=1,
            font=dict(size=11, color=T["text_primary"]),
        ),
        xaxis=dict(
            gridcolor=T["grid_color"],
            linecolor=T["axis_color"],
            tickfont=dict(size=10, color=T["text_primary"]),
            tickangle=xangle,
            title_font=dict(size=11, color=T["text_secondary"]),
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor=T["grid_color"],
            linecolor=T["axis_color"],
            tickfont=dict(size=10, color=T["text_primary"]),
            title_font=dict(size=11, color=T["text_secondary"]),
            zeroline=False,
        ),
        coloraxis_showscale=False,
        hoverlabel=dict(
            bgcolor=T["bg_card"],
            bordercolor=T["border"],
            font=dict(size=11, color=T["text_primary"]),
        ),
        template=T["plotly_tmpl"],
    )
    return fig


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800)
def load_all_data():
    def safe_read(path):
        if os.path.exists(path):
            try:
                return pd.read_parquet(path)
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()
    merged  = safe_read(MERGED_DATA_PATH)
    if merged.empty:
        merged = safe_read(PROCESSED_DATA_PATH)
    summary = safe_read(STATION_SUMMARY_PATH)
    return merged, summary


@st.cache_data(ttl=300)
def load_aemo_data():
    def safe_read(path):
        if os.path.exists(path):
            try:
                return pd.read_parquet(path)
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()
    return (
        safe_read("data/processed/aemo_latest.parquet"),
        safe_read("data/processed/aemo_bom_merged.parquet"),
        safe_read("data/processed/aemo_history.parquet"),
    )


@st.cache_data(ttl=1800)
def load_shap_data():
    def safe_read(path):
        if os.path.exists(path):
            try:
                return pd.read_parquet(path)
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()
    return (
        safe_read("data/processed/shap_results.parquet"),
        safe_read("data/processed/shap_global_importance.parquet"),
    )


# ── Load all data ─────────────────────────────────────────────────────────────

df, summary                        = load_all_data()
df_aemo, df_aemo_bom, df_aemo_hist = load_aemo_data()
df_shap, df_shap_global            = load_shap_data()

# ── Pre-compute availability flags ────────────────────────────────────────────

barra2_available = (
    "barra2_temp_c" in df.columns
    and df["barra2_temp_c"].notna().any()
) if not df.empty else False

anomaly_available = (
    "temp_anomaly_c" in df.columns
    and df["temp_anomaly_c"].notna().any()
) if not df.empty else False

shap_available = (
    "shap_computed" in df.columns
    and df["shap_computed"].any()
) if not df.empty else False

ml_available = (
    "ml_anomaly_flag" in df.columns
    and df["ml_anomaly_flag"].notna().any()
) if not df.empty else False

aemo_available = (
    not df_aemo.empty
    and "total_demand_mw" in df_aemo.columns
    and df_aemo["total_demand_mw"].notna().any()
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:

    st.markdown(f"""
    <div class="aus-sidebar-brand">
        <div class="aus-sidebar-brand-title">
            AusClimate<span class="aus-sidebar-brand-dot"></span>
        </div>
        <div class="aus-sidebar-brand-sub">Real-Time Climate Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    # Theme toggle
    st.markdown('<p class="aus-section-label">Display</p>',
                unsafe_allow_html=True)
    btn_label = "Switch to Light Mode" \
        if st.session_state.theme == "dark" else "Switch to Dark Mode"
    if st.button(btn_label, use_container_width=True, key="theme_toggle"):
        st.session_state.theme = (
            "light" if st.session_state.theme == "dark" else "dark"
        )
        st.rerun()

    st.divider()

    # Data source status
    st.markdown('<p class="aus-section-label">Data Sources</p>',
                unsafe_allow_html=True)

    n_stations  = df["station_name"].nunique() \
        if not df.empty and "station_name" in df.columns else 0
    bom_cls     = "aus-badge-live" if n_stations > 0 else "aus-badge-offline"
    bom_label   = f"LIVE {n_stations:,}" if n_stations > 0 else "OFFLINE"
    b2_cls      = "aus-badge-live" if barra2_available else "aus-badge-pending"
    b2_label    = "ACTIVE" if barra2_available else "PENDING"
    aemo_cls    = "aus-badge-live" if aemo_available else "aus-badge-pending"
    aemo_label  = "LIVE" if aemo_available else "PENDING"

    st.markdown(f"""
    <div class="aus-src-wrap">
        <div class="aus-src-row">
            <span class="aus-src-label">Bureau of Meteorology</span>
            <span class="aus-badge {bom_cls}">{bom_label}</span>
        </div>
        <div class="aus-src-row">
            <span class="aus-src-label">BARRA2 Reanalysis</span>
            <span class="aus-badge {b2_cls}">{b2_label}</span>
        </div>
        <div class="aus-src-row">
            <span class="aus-src-label">AEMO Dispatch</span>
            <span class="aus-badge {aemo_cls}">{aemo_label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Refresh BOM Data", use_container_width=True,
                 key="refresh_btn"):
        with st.spinner("Refreshing..."):
            try:
                from ingestion.bom import fetch_all_stations, get_station_coords
                from pipeline.cleaner import clean_bom_data
                from pipeline.feature_eng import compute_station_summary
                from pipeline.storage import save_parquet
                from pipeline.spatial_join import merge_bom_barra2

                df_raw   = fetch_all_stations()
                df_clean = clean_bom_data(df_raw)
                save_parquet(df_clean, PROCESSED_DATA_PATH)

                b2_cached = pd.DataFrame()
                if os.path.exists(BARRA2_CACHE_PATH):
                    try:
                        b2_cached = pd.read_parquet(BARRA2_CACHE_PATH)
                    except Exception:
                        pass

                merged = merge_bom_barra2(df_clean, b2_cached)
                save_parquet(merged, MERGED_DATA_PATH)
                feat = compute_station_summary(merged)
                save_parquet(feat, STATION_SUMMARY_PATH)
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Refresh failed: {e}")

    st.divider()

    # Filters
    if df.empty:
        st.warning("No data. Run `python main.py` first.")
        st.stop()

    st.markdown('<p class="aus-section-label">Filters</p>',
                unsafe_allow_html=True)

    all_states = sorted(df["state"].dropna().unique().tolist()) \
        if "state" in df.columns else []

    selected_states = st.multiselect(
        "States", options=all_states, default=all_states,
    )

    df_state = (
        df[df["state"].isin(selected_states)]
        if selected_states and "state" in df.columns else df
    )

    all_stations_list = sorted(
        df_state["station_name"].dropna().unique().tolist()
    ) if "station_name" in df_state.columns else []

    selected_stations = st.multiselect(
        "Stations",
        options=all_stations_list,
        default=all_stations_list[:12]
        if len(all_stations_list) > 12 else all_stations_list,
    )

    FEATURE_OPTIONS = {}
    for col, label in {
        "temperature_c":  "Temperature (C)",
        "feels_like_c":   "Feels Like (C)",
        "humidity_pct":   "Humidity (%)",
        "wind_speed_kmh": "Wind Speed (km/h)",
        "gust_kmh":       "Max Gust (km/h)",
        "pressure_hpa":   "Pressure (hPa)",
        "dewpoint_c":     "Dewpoint (C)",
        "rainfall_mm":    "Rainfall (mm)",
    }.items():
        if col in df.columns:
            FEATURE_OPTIONS[label] = col

    selected_label = st.selectbox("Variable", list(FEATURE_OPTIONS.keys()))
    selected_col   = FEATURE_OPTIONS[selected_label]

    chart_type = st.radio(
        "Chart Type",
        ["Bar", "Horizontal Bar", "Line", "Scatter",
         "Box", "Violin", "Radar", "Heatmap", "Bubble"],
    )


# ── Filter data ───────────────────────────────────────────────────────────────

if selected_stations:
    df_filtered = df[df["station_name"].isin(selected_stations)].copy()
else:
    df_filtered = df_state.copy()

df_filtered[selected_col] = pd.to_numeric(
    df_filtered[selected_col], errors="coerce"
)

station_mean = (
    df_filtered.groupby("station_name")[selected_col]
    .mean().dropna().reset_index()
    .sort_values(selected_col, ascending=False)
)
station_mean.columns = ["Station", selected_label]


# ── Operational header ────────────────────────────────────────────────────────

now_str = datetime.now().strftime("%d %b %Y  %H:%M AEST")

st.markdown(f"""
<div class="aus-header">
    <div class="aus-header-left">
        <div class="aus-header-icon">🌡️</div>
        <div>
            <div class="aus-header-title">AusClimate</div>
            <div class="aus-header-sub">
                Real-Time Australian Climate Analytics Platform
                &nbsp;&middot;&nbsp; University of Technology Sydney
            </div>
        </div>
    </div>
    <div class="aus-header-right">
        <span class="aus-badge {bom_cls}">BOM {n_stations:,} stn</span>
        <span class="aus-badge {b2_cls}">BARRA2</span>
        <span class="aus-badge {aemo_cls}">AEMO</span>
        <span class="aus-timestamp">{now_str}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview",
    "Station Map",
    "Anomalies",
    "BARRA2 Comparison",
    "Data Table",
    "Energy",
    "Climate Assistant",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 -- OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Live Australian Climate Overview")

    vals = pd.to_numeric(df_filtered[selected_col], errors="coerce").dropna()
    avg_val = vals.mean() if not vals.empty else 0
    max_val = vals.max()  if not vals.empty else 0
    min_val = vals.min()  if not vals.empty else 0

    max_station = ""
    if not vals.empty and "station_name" in df_filtered.columns:
        idx = pd.to_numeric(
            df_filtered[selected_col], errors="coerce").idxmax()
        max_station = df_filtered.loc[idx, "station_name"]

    n_states   = df["state"].nunique() if "state" in df.columns else 0
    n_records  = len(df)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Stations",        f"{n_stations:,}")
    c2.metric("States",          n_states)
    c3.metric("Records",         f"{n_records:,}")
    c4.metric(f"Avg {selected_label}", f"{avg_val:.1f}")
    c5.metric("Highest",         f"{max_val:.1f}", max_station)
    c6.metric("Lowest",          f"{min_val:.1f}")

    st.divider()

    # Main chart
    if station_mean.empty:
        st.info("Select stations from the sidebar.")
    else:
        if chart_type == "Bar":
            fig = px.bar(
                station_mean, x="Station", y=selected_label,
                color=selected_label,
                color_continuous_scale=COLOUR_SCALE,
                text_auto=".1f", template=T["plotly_tmpl"],
            )
            fig.update_traces(textposition="outside")
            chart_layout(fig, height=440, xangle=-40)

        elif chart_type == "Horizontal Bar":
            fig = px.bar(
                station_mean.sort_values(selected_label),
                x=selected_label, y="Station",
                orientation="h",
                color=selected_label,
                color_continuous_scale=COLOUR_SCALE,
                text_auto=".1f", template=T["plotly_tmpl"],
                height=max(440, len(station_mean) * 24),
            )
            chart_layout(
                fig,
                height=max(440, len(station_mean) * 24)
            )

        elif chart_type == "Line":
            if "timestamp" in df_filtered.columns:
                fig = px.line(
                    df_filtered.dropna(
                        subset=["timestamp", selected_col]
                    ).sort_values("timestamp"),
                    x="timestamp", y=selected_col,
                    color="station_name",
                    template=T["plotly_tmpl"],
                    labels={selected_col: selected_label,
                            "timestamp": "Time",
                            "station_name": "Station"},
                )
            else:
                fig = px.line(
                    station_mean, x="Station", y=selected_label,
                    markers=True, template=T["plotly_tmpl"],
                )
            chart_layout(fig, height=440, legend=True)

        elif chart_type == "Scatter":
            fig = px.scatter(
                df_filtered.dropna(subset=[selected_col]),
                x="station_name", y=selected_col,
                color="state" if "state" in df_filtered.columns else None,
                opacity=0.5, template=T["plotly_tmpl"],
                labels={selected_col: selected_label,
                        "station_name": "Station"},
            )
            chart_layout(fig, height=440, xangle=-40)

        elif chart_type == "Box":
            fig = px.box(
                df_filtered.dropna(subset=[selected_col]),
                x="state" if "state" in df_filtered.columns else "station_name",
                y=selected_col,
                color="state" if "state" in df_filtered.columns else None,
                points="outliers", template=T["plotly_tmpl"],
                labels={selected_col: selected_label},
            )
            chart_layout(fig, height=440)

        elif chart_type == "Violin":
            fig = px.violin(
                df_filtered.dropna(subset=[selected_col]),
                x="state" if "state" in df_filtered.columns else "station_name",
                y=selected_col,
                color="state" if "state" in df_filtered.columns else None,
                box=True, template=T["plotly_tmpl"],
                labels={selected_col: selected_label},
            )
            chart_layout(fig, height=440)

        elif chart_type == "Bubble":
            need = ["temperature_c", "humidity_pct", "wind_speed_kmh"]
            if all(c in df_filtered.columns for c in need):
                bub = df_filtered.groupby("station_name").agg({
                    "temperature_c":  "mean",
                    "humidity_pct":   "mean",
                    "wind_speed_kmh": "mean",
                    "state":          "first",
                }).reset_index().dropna()
                for c in need:
                    bub[c] = pd.to_numeric(bub[c], errors="coerce").round(1)
                fig = px.scatter(
                    bub, x="temperature_c", y="humidity_pct",
                    size="wind_speed_kmh",
                    color="state" if "state" in bub.columns else "station_name",
                    hover_name="station_name",
                    size_max=30, template=T["plotly_tmpl"],
                    labels={
                        "temperature_c":  "Temperature (C)",
                        "humidity_pct":   "Humidity (%)",
                        "wind_speed_kmh": "Wind Speed (km/h)",
                    },
                )
                chart_layout(fig, height=480, legend=True)
            else:
                st.info("Bubble chart needs temperature, humidity, and wind data.")
                fig = None

        elif chart_type == "Radar":
            radar_vars = {
                "temperature_c":  "Temp",
                "humidity_pct":   "Humidity",
                "wind_speed_kmh": "Wind",
                "pressure_hpa":   "Pressure",
                "feels_like_c":   "Feels Like",
            }
            avail_r = {k: v for k, v in radar_vars.items()
                       if k in df_filtered.columns}
            gcol    = "state" if "state" in df_filtered.columns else "station_name"
            fig     = go.Figure()
            for gv in df_filtered[gcol].dropna().unique():
                grp   = df_filtered[df_filtered[gcol] == gv]
                means = [
                    pd.to_numeric(grp[c], errors="coerce").mean()
                    for c in avail_r
                ]
                fig.add_trace(go.Scatterpolar(
                    r=means, theta=list(avail_r.values()),
                    fill="toself", name=str(gv), opacity=0.6,
                ))
            fig.update_layout(
                polar=dict(
                    bgcolor=T["bg_card"],
                    radialaxis=dict(
                        visible=True,
                        gridcolor=T["grid_color"],
                        linecolor=T["axis_color"],
                        tickfont=dict(size=9, color=T["text_secondary"]),
                    ),
                    angularaxis=dict(
                        gridcolor=T["grid_color"],
                        linecolor=T["axis_color"],
                        tickfont=dict(size=10, color=T["text_secondary"]),
                    ),
                ),
                paper_bgcolor=T["paper_bg"],
                font=dict(color=T["text_secondary"], size=11),
                height=480, showlegend=True,
                margin=dict(t=44, b=20, l=40, r=40),
                legend=dict(
                    bgcolor="rgba(0,0,0,0)",
                    bordercolor=T["border"], borderwidth=1,
                    font=dict(size=11, color=T["text_secondary"]),
                ),
            )

        elif chart_type == "Heatmap":
            hmap_vars = {
                "temperature_c":  "Temp (C)",
                "humidity_pct":   "Humidity (%)",
                "wind_speed_kmh": "Wind (km/h)",
                "pressure_hpa":   "Pressure (hPa)",
                "feels_like_c":   "Feels Like (C)",
            }
            avail_h = {k: v for k, v in hmap_vars.items()
                       if k in df_filtered.columns}
            hdf     = df_filtered.groupby("station_name")[
                list(avail_h.keys())
            ].mean().reset_index().set_index("station_name")
            for c in hdf.columns:
                hdf[c] = pd.to_numeric(hdf[c], errors="coerce")
            hn = hdf.copy()
            for c in hn.columns:
                r = hn[c].max() - hn[c].min()
                if r > 0:
                    hn[c] = (hn[c] - hn[c].min()) / r
            hn.columns = list(avail_h.values())
            fig = px.imshow(
                hn.head(40).T,
                color_continuous_scale=COLOUR_SCALE,
                aspect="auto",
                template=T["plotly_tmpl"],
                labels=dict(x="Station", y="Variable", color="Value"),
            )
            chart_layout(fig, height=max(280, len(avail_h) * 60),
                         xangle=-45)

        else:
            fig = None

        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

    # Pinnable tile dashboard
    st.divider()
    st.subheader("My Dashboard")

    if "tiles" not in st.session_state:
        st.session_state.tiles = []

    b1, b2, _ = st.columns([1, 1, 4])
    with b1:
        if st.button("Pin Chart", use_container_width=True):
            if selected_stations:
                st.session_state.tiles.append({
                    "stations": selected_stations[:],
                    "col":      selected_col,
                    "label":    selected_label,
                    "chart":    chart_type,
                })
                st.toast("Chart pinned.")
    with b2:
        if st.button("Clear All", use_container_width=True):
            st.session_state.tiles = []

    if st.session_state.tiles:
        tile_cols    = st.columns(2)
        to_remove    = []
        for i, tile in enumerate(st.session_state.tiles):
            with tile_cols[i % 2]:
                with st.container(border=True):
                    td = df[df["station_name"].isin(
                        tile["stations"])].copy()
                    td[tile["col"]] = pd.to_numeric(
                        td[tile["col"]], errors="coerce")
                    ts = (
                        td.groupby("station_name")[tile["col"]]
                        .mean().dropna().reset_index()
                        .sort_values(tile["col"], ascending=False)
                    )
                    ts.columns = ["Station", tile["label"]]

                    if tile["chart"] == "Horizontal Bar":
                        tf = px.bar(
                            ts.sort_values(tile["label"]),
                            x=tile["label"], y="Station",
                            orientation="h",
                            color=tile["label"],
                            color_continuous_scale=COLOUR_SCALE,
                            template=T["plotly_tmpl"],
                        )
                    elif (tile["chart"] == "Line"
                          and "timestamp" in td.columns):
                        tf = px.line(
                            td.sort_values("timestamp"),
                            x="timestamp", y=tile["col"],
                            color="station_name",
                            template=T["plotly_tmpl"],
                        )
                    elif tile["chart"] == "Scatter":
                        tf = px.scatter(
                            td.dropna(subset=[tile["col"]]),
                            x="station_name", y=tile["col"],
                            color="state" if "state" in td.columns else None,
                            opacity=0.5, template=T["plotly_tmpl"],
                        )
                    elif tile["chart"] == "Box":
                        tf = px.box(
                            td.dropna(subset=[tile["col"]]),
                            x="state" if "state" in td.columns else "station_name",
                            y=tile["col"],
                            template=T["plotly_tmpl"],
                        )
                    elif tile["chart"] == "Violin":
                        tf = px.violin(
                            td.dropna(subset=[tile["col"]]),
                            x="state" if "state" in td.columns else "station_name",
                            y=tile["col"],
                            box=True, template=T["plotly_tmpl"],
                        )
                    else:
                        tf = px.bar(
                            ts, x="Station", y=tile["label"],
                            color=tile["label"],
                            color_continuous_scale=COLOUR_SCALE,
                            template=T["plotly_tmpl"],
                        )

                    chart_layout(
                        tf, title=tile["label"], height=260,
                        margin=dict(t=40, b=20, l=20, r=10)
                    )
                    st.plotly_chart(tf, use_container_width=True)
                    if st.button("Remove", key=f"rm_{i}"):
                        to_remove.append(i)

        for idx in reversed(to_remove):
            st.session_state.tiles.pop(idx)
        if to_remove:
            st.rerun()
    else:
        st.caption("Select variable and chart type then click Pin Chart.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 -- STATION MAP
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("National Station Coverage")

    if "latitude" not in df.columns or "longitude" not in df.columns:
        st.info("No coordinate data available.")
    else:
        map_df = (
            df.groupby("station_name").agg({
                "latitude":      "first",
                "longitude":     "first",
                "state":         "first",
                "temperature_c": "mean",
                "humidity_pct":  "mean",
            }).reset_index()
        )
        for c in ["latitude", "longitude", "temperature_c", "humidity_pct"]:
            map_df[c] = pd.to_numeric(map_df[c], errors="coerce")
        map_df = map_df.dropna(subset=["latitude", "longitude"])

        map_col_opts = {}
        for col, lbl in {
            "temperature_c": "Temperature (C)",
            "humidity_pct":  "Humidity (%)",
        }.items():
            if col in map_df.columns:
                map_col_opts[lbl] = col

        map_colour_label = st.selectbox(
            "Colour by", list(map_col_opts.keys()), key="map_col"
        )
        map_colour_col = map_col_opts[map_colour_label]

        fig_map = px.scatter_mapbox(
            map_df,
            lat="latitude", lon="longitude",
            color=map_colour_col,
            color_continuous_scale=COLOUR_SCALE,
            size_max=9,
            hover_name="station_name",
            hover_data={
                "state":         True,
                "temperature_c": ":.1f",
                "humidity_pct":  ":.1f",
                "latitude":      False,
                "longitude":     False,
            },
            mapbox_style=T["mapbox_style"],
            zoom=3.5,
            center={"lat": -26, "lon": 134},
            height=560,
            labels={map_colour_col: map_colour_label},
        )
        chart_layout(fig_map, margin=dict(t=20, b=0, l=0, r=0))
        st.plotly_chart(fig_map, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Stations per State")
            sc = (df.groupby("state")["station_name"]
                    .nunique().reset_index())
            sc.columns = ["State", "Stations"]
            fig_sc = px.bar(
                sc.sort_values("Stations", ascending=False),
                x="State", y="Stations",
                color="Stations",
                color_continuous_scale=COLOUR_SCALE_BLUE,
                text_auto=True, template=T["plotly_tmpl"],
            )
            chart_layout(fig_sc, height=300)
            st.plotly_chart(fig_sc, use_container_width=True)

        with c2:
            st.subheader("Temperature Distribution")
            temp_vals = pd.to_numeric(
                df["temperature_c"], errors="coerce"
            ).dropna()
            fig_hist = px.histogram(
                x=temp_vals, nbins=40,
                template=T["plotly_tmpl"],
                color_discrete_sequence=[T["accent"]],
                labels={"x": "Temperature (C)", "count": "Observations"},
            )
            chart_layout(fig_hist, height=300)
            st.plotly_chart(fig_hist, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 -- ANOMALIES
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Climate Anomaly Detection")

    if ml_available:
        st.success(
            "Isolation Forest active - "
            "anomalies detected across 5 climate variables."
        )
        ml_df      = df[df["ml_anomaly_flag"] == True].copy()
        n_total    = len(df)
        n_anomaly  = int(df["ml_anomaly_flag"].sum())
        n_severe   = int((df["ml_severity"] == "Severe").sum()) \
            if "ml_severity" in df.columns else 0
        n_moderate = int((df["ml_severity"] == "Moderate").sum()) \
            if "ml_severity" in df.columns else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Scored",    f"{n_total:,}")
        c2.metric("Flagged",   f"{n_anomaly:,}")
        c3.metric("Severe",    n_severe)
        c4.metric("Moderate",  n_moderate)

        st.divider()

        col_sev, col_drv = st.columns(2)
        with col_sev:
            if "ml_severity" in df.columns:
                sev = df["ml_severity"].value_counts().reset_index()
                sev.columns = ["Severity", "Count"]
                sev_colours = {
                    "Severe":   T["critical"],
                    "Moderate": T["warning"],
                    "Mild":     "#A0A020",
                    "Normal":   T["success"],
                    "Unknown":  T["text_muted"],
                }
                fig_sev = px.bar(
                    sev, x="Severity", y="Count",
                    color="Severity",
                    color_discrete_map=sev_colours,
                    template=T["plotly_tmpl"],
                    text_auto=True,
                )
                chart_layout(fig_sev, title="Severity Distribution",
                             height=300)
                fig_sev.update_layout(showlegend=False)
                st.plotly_chart(fig_sev, use_container_width=True)

        with col_drv:
            drv_col = "shap_top_driver" \
                if shap_available else "ml_primary_driver"
            if drv_col in df.columns:
                drvs = (
                    ml_df[drv_col].replace("", np.nan).dropna()
                    .value_counts().reset_index()
                )
                drvs.columns = ["Driver", "Count"]
                if not drvs.empty:
                    fig_drv = px.pie(
                        drvs, names="Driver", values="Count",
                        template=T["plotly_tmpl"],
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    chart_layout(fig_drv,
                                 title="Primary Driver" +
                                 " (SHAP)" if shap_available else " (Phase 2)",
                                 height=300,
                                 margin=dict(t=44, b=20, l=20, r=20))
                    st.plotly_chart(fig_drv, use_container_width=True)

        # Risk map
        st.subheader("Station Risk Map")
        try:
            from anomaly.detector import compute_station_anomaly_summary
            station_risk = compute_station_anomaly_summary(df)

            # Merge coordinates if missing from station_risk
            if not station_risk.empty:
                has_coords = (
                    "latitude" in station_risk.columns
                    and station_risk["latitude"].notna().any()
                )
                if not has_coords:
                    stn_list_path = "data/processed/station_list.parquet"
                    if os.path.exists(stn_list_path):
                        stn_coords = pd.read_parquet(stn_list_path)
                        if "lat" in stn_coords.columns:
                            stn_coords = stn_coords.rename(
                                columns={"lat": "latitude",
                                         "lon": "longitude"}
                            )
                        station_risk = station_risk.merge(
                            stn_coords[["station_name",
                                        "latitude", "longitude"]],
                            on="station_name", how="left"
                        )
                    elif "latitude" in df.columns:
                        coords = (
                            df.groupby("station_name")[
                                ["latitude", "longitude"]
                            ].first().reset_index()
                        )
                        station_risk = station_risk.merge(
                            coords, on="station_name", how="left"
                        )

            if not station_risk.empty:
                risk_colours = {
                    "High":   T["critical"],
                    "Medium": T["warning"],
                    "Low":    "#A0A020",
                    "Normal": T["success"],
                }
                map_risk = station_risk.dropna(
                    subset=["latitude", "longitude"]
                )
                if not map_risk.empty:
                    fig_risk = px.scatter_mapbox(
                        map_risk,
                        lat="latitude", lon="longitude",
                        color="station_risk",
                        color_discrete_map=risk_colours,
                        size="if_anomaly_count",
                        size_max=18,
                        hover_name="station_name",
                        hover_data={
                        "state":              True,
                        "if_anomaly_count":   True,
                        "if_anomaly_rate_pct":True,
                        "station_risk":       True,
                        "latitude":           False,
                        "longitude":          False,
                        },
                        mapbox_style=T["mapbox_style"],
                        zoom=3.5,
                        center={"lat": -26, "lon": 134},
                        height=480,
                    )
                    chart_layout(
                        fig_risk,
                        title="Station Risk -- dot size = anomaly count",
                        margin=dict(t=44, b=0, l=0, r=0),
                        legend=True
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)
                else:
                    st.info("No coordinate data available for risk map.")

                risk_rename = {
                    "station_name":      "Station",
                    "state":             "State",
                    "total_observations":"Total Obs",
                    "if_anomaly_count":  "IF Anomalies",
                    "if_anomaly_rate_pct":"IF Rate (%)",
                    "lof_anomaly_count":  "LOF Anomalies",
                    "ensemble_count":     "Consensus",
                    "severe_count":      "Severe",
                    "moderate_count":    "Moderate",
                    "avg_anomaly_score": "Avg Score",
                    "primary_driver":    "Driver",
                    "station_risk":      "Risk",
                }
                avail_r = {k: v for k, v in risk_rename.items()
                           if k in station_risk.columns}
                disp_r  = station_risk[list(avail_r.keys())].copy()
                disp_r.rename(columns=avail_r, inplace=True)
                st.dataframe(
                    disp_r.sort_values("Anomalies", ascending=False)
                    if "Anomalies" in disp_r.columns else disp_r,
                    use_container_width=True, hide_index=True,
                )
        except Exception as e:
            st.error(f"Risk map error: {e}")
        
    # SHAP section
        if shap_available:
            st.divider()
            st.subheader("SHAP Explainability")
            st.caption(
                "Shapley values show how much each variable contributed "
                "to each anomaly score. More negative = stronger anomaly driver."
            )
    
            feat_labels = {
                "temperature_c":  "Temperature",
                "feels_like_c":   "Apparent Temp",
                "humidity_pct":   "Humidity",
                "wind_speed_kmh": "Wind Speed",
                "pressure_hpa":   "Pressure",
            }
            feat_units = {
                "temperature_c":  "C",
                "feels_like_c":   "C",
                "humidity_pct":   "%",
                "wind_speed_kmh": "km/h",
                "pressure_hpa":   "hPa",
            }
            shap_feat_cols = [
                c for c in df.columns
                if c.startswith("shap_")
                and c not in ["shap_top_driver", "shap_top_driver_val",
                              "shap_explanation", "shap_computed"]
            ]
    
            if not df_shap_global.empty:
                st.markdown("**Global Feature Importance**")
                fig_gi = px.bar(
                    df_shap_global,
                    x="mean_abs_shap", y="label",
                    orientation="h",
                    color="mean_abs_shap",
                    color_continuous_scale=COLOUR_SCALE_SEQ,
                    text=df_shap_global["mean_abs_shap"].round(4).astype(str),
                    template=T["plotly_tmpl"],
                    labels={"mean_abs_shap": "Mean |SHAP|", "label": "Variable"},
                )
                fig_gi.update_traces(textposition="outside")
                chart_layout(fig_gi, height=280,
                             margin=dict(t=30, b=20, l=120, r=60))
                st.plotly_chart(fig_gi, use_container_width=True)
    
            # Waterfall
            st.markdown("**Station Waterfall Chart**")
            anom_stns = df[
                (df["ml_anomaly_flag"] == True) &
                (df.get("shap_computed", pd.Series(False, index=df.index)))
            ]["station_name"].dropna().unique().tolist() \
                if "shap_computed" in df.columns else []
    
            if anom_stns:
                sel_stn  = st.selectbox("Select station",
                                        options=sorted(anom_stns),
                                        key="shap_stn")
                stn_rows = df[
                    (df["station_name"] == sel_stn) &
                    (df["ml_anomaly_flag"] == True)
                ].sort_values("ml_anomaly_score")
    
                if not stn_rows.empty:
                    top_row = stn_rows.iloc[0]

                    # Pull SHAP values from df_shap if missing in df
                    if not df_shap.empty:
                        shap_match = pd.DataFrame()
                        if "station_name" in df_shap.columns:
                            shap_match = df_shap[
                                df_shap["station_name"] == sel_stn
                            ]
                        if shap_match.empty:
                            shap_match = df_shap[
                                df_shap.index.isin(stn_rows.index)
                            ]
                        if not shap_match.empty:
                            top_row = top_row.copy()
                            for sc in shap_feat_cols:
                                if sc in shap_match.columns:
                                    val = shap_match[sc].dropna()
                                    if not val.empty:
                                        top_row[sc] = val.iloc[0]

                    wf_l, wf_v = [], []
                    for sc in shap_feat_cols:
                        feat  = sc.replace("shap_", "")
                        label = feat_labels.get(feat, feat)
                        unit  = feat_units.get(feat, "")
                        if sc not in top_row.index:
                            continue
                        sv = top_row[sc]
                        # Skip NaN values
                        try:
                            sv = float(sv)
                            if np.isnan(sv):
                                continue
                        except (TypeError, ValueError):
                            continue
                        av = float(top_row[feat]) \
                            if feat in top_row.index else np.nan
                        try:
                            avs = f"{av:.1f}{unit}" \
                                if not np.isnan(av) else "N/A"
                        except (TypeError, ValueError):
                            avs = "N/A"
                        wf_l.append(f"{label} ({avs})")
                        wf_v.append(sv)

                    if wf_v:
                        pairs  = sorted(zip(wf_l, wf_v), key=lambda x: x[1])
                        wl     = [p[0] for p in pairs]
                        wv     = [p[1] for p in pairs]
                        colors = [T["critical"] if v < 0 else T["success"]
                                  for v in wv]
                        score  = float(top_row.get("ml_anomaly_score", 0))
                        fig_wf = go.Figure(go.Bar(
                            x=wl, y=wv,
                            marker_color=colors,
                            text=[f"{v:+.4f}" for v in wv],
                            textposition="outside",
                            textfont=dict(size=10, color=T["text_secondary"]),
                        ))
                        fig_wf.add_hline(y=0, line_color=T["border"],
                                         opacity=0.5)
                        chart_layout(
                            fig_wf,
                            title=f"SHAP Waterfall -- {sel_stn} "
                                  f"(score: {score:.4f})",
                            height=380,
                        )
                        st.plotly_chart(fig_wf, use_container_width=True)

                        expl = top_row.get("shap_explanation", "")
                        if expl and str(expl) not in ("", "nan"):
                            st.info(f"Explanation: {expl}")
                    else:
                        st.info(
                            "SHAP values not available for this station. "
                            "Run `python main.py --retrain` to recompute."
                        )
    
            # SHAP table
            st.markdown("**Top 20 Anomalies with SHAP Explanations**")
            expl_cols = [c for c in [
                "station_name", "state", "temperature_c", "humidity_pct",
                "ml_anomaly_score", "ml_severity",
                "shap_top_driver", "shap_explanation",
            ] if c in df.columns]
            if "shap_computed" in df.columns:
                expl_df = (
                    df[df["shap_computed"] == True][expl_cols]
                    .sort_values("ml_anomaly_score").head(20)
                    .rename(columns={
                        "station_name":     "Station",
                        "state":            "State",
                        "temperature_c":    "Temp (C)",
                        "humidity_pct":     "Humidity (%)",
                        "ml_anomaly_score": "Score",
                        "ml_severity":      "Severity",
                        "shap_top_driver":  "Top Driver",
                        "shap_explanation": "SHAP Explanation",
                    })
                )
                st.dataframe(expl_df, use_container_width=True, hide_index=True)
    
            st.caption(
                "Score: more negative = more anomalous. "
                "Red bars push toward anomalous, green bars pull toward normal."
            )
        
                # LOF and ensemble section
        lof_available = (
            "lof_anomaly_flag" in df.columns
            and df["lof_anomaly_flag"].notna().any()
        )
        ensemble_available = (
            "ensemble_anomaly" in df.columns
            and df["ensemble_anomaly"].any()
        )

        if lof_available:
            st.divider()
            st.subheader("LOF and Ensemble Voting")

            n_if_flags  = int(df["ml_anomaly_flag"].sum())
            n_lof_flags = int(df["lof_anomaly_flag"].sum())
            n_ensemble  = int(df["ensemble_anomaly"].sum()) \
                if ensemble_available else 0
            n_if_only   = int(df.get("if_only", pd.Series(False)).sum())
            n_lof_only  = int(df.get("lof_only", pd.Series(False)).sum())

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("IF Anomalies",        f"{n_if_flags:,}")
            c2.metric("LOF Anomalies",        f"{n_lof_flags:,}")
            c3.metric("Consensus (Both)",     f"{n_ensemble:,}",
                      "High confidence")
            c4.metric("Agreement Rate",
                      f"{n_ensemble/max(n_if_flags,1)*100:.0f}%")

            st.divider()

            # Venn-style breakdown chart
            comparison_data = pd.DataFrame({
                "Category": [
                    "IF only (not LOF)",
                    "LOF only (not IF)",
                    "Both agree (Ensemble)"
                ],
                "Count": [n_if_only, n_lof_only, n_ensemble],
                "Color": [
                    T["warning"],
                    T["accent"],
                    T["critical"]
                ]
            })

            col_venn, col_detail = st.columns([1, 1])
            with col_venn:
                fig_venn = px.bar(
                    comparison_data,
                    x="Category", y="Count",
                    color="Category",
                    color_discrete_sequence=[
                        T["warning"], T["accent"], T["critical"]
                    ],
                    text_auto=True,
                    template=T["plotly_tmpl"],
                    title="IF vs LOF Agreement",
                )
                chart_layout(fig_venn, height=320, xangle=-10)
                fig_venn.update_layout(showlegend=False)
                st.plotly_chart(fig_venn, use_container_width=True)

            with col_detail:
                st.markdown(f"""
                <div style="padding:16px; background:{T['bg_card']};
                            border:1px solid {T['border']};
                            border-radius:8px; margin-top:8px;">
                    <p style="color:{T['text_secondary']};
                              font-size:11px; font-weight:700;
                              text-transform:uppercase;
                              letter-spacing:0.1em; margin-bottom:12px;">
                        How to Interpret
                    </p>
                    <p style="color:{T['text_primary']}; font-size:13px;
                              line-height:1.6; margin-bottom:10px;">
                        <span style="color:{T['critical']}; font-weight:600;">
                            Ensemble ({n_ensemble:,} observations)
                        </span><br>
                        Flagged by BOTH models independently.
                        Highest confidence. Very low false positive rate.
                    </p>
                    <p style="color:{T['text_primary']}; font-size:13px;
                              line-height:1.6; margin-bottom:10px;">
                        <span style="color:{T['warning']}; font-weight:600;">
                            IF only ({n_if_only:,} observations)
                        </span><br>
                        Globally unusual but locally similar to neighbours.
                        May be mild extremes worth monitoring.
                    </p>
                    <p style="color:{T['text_primary']}; font-size:13px;
                              line-height:1.6;">
                        <span style="color:{T['accent_light']}; font-weight:600;">
                            LOF only ({n_lof_only:,} observations)
                        </span><br>
                        Locally isolated but not globally extreme.
                        Often microclimatic anomalies.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Ensemble top anomalies map
            if ensemble_available:
                st.markdown("**High-Confidence Ensemble Anomalies by Station**")
                ens_df = df[df["ensemble_anomaly"] == True].copy()

                if not ens_df.empty and "latitude" in ens_df.columns:
                    ens_map = (
                        ens_df.groupby("station_name").agg({
                            "latitude":           "first",
                            "longitude":          "first",
                            "state":              "first",
                            "ensemble_anomaly":   "sum",
                            "ml_anomaly_score":   "mean",
                            "ml_primary_driver":  lambda x:
                                x.mode().iloc[0]
                                if not x.mode().empty else "",
                        }).reset_index()
                        .dropna(subset=["latitude", "longitude"])
                    )
                    if not ens_map.empty:
                        fig_ens_map = px.scatter_mapbox(
                            ens_map,
                            lat="latitude", lon="longitude",
                            color="ml_anomaly_score",
                            color_continuous_scale=COLOUR_SCALE_DIV,
                            size="ensemble_anomaly",
                            size_max=20,
                            hover_name="station_name",
                            hover_data={
                                "state":             True,
                                "ensemble_anomaly":  True,
                                "ml_anomaly_score":  ":.4f",
                                "ml_primary_driver": True,
                                "latitude":          False,
                                "longitude":         False,
                            },
                            mapbox_style=T["mapbox_style"],
                            zoom=3.5,
                            center={"lat": -26, "lon": 134},
                            height=440,
                        )
                        chart_layout(
                            fig_ens_map,
                            title="Ensemble Anomalies -- "
                                  "dot size = count, colour = IF score",
                            margin=dict(t=44, b=0, l=0, r=0),
                        )
                        st.plotly_chart(
                            fig_ens_map, use_container_width=True
                        )

                # Ensemble detail table
                ens_cols = [c for c in [
                    "station_name", "state",
                    "temperature_c", "humidity_pct",
                    "ml_anomaly_score", "lof_anomaly_score",
                    "ensemble_severity", "ml_primary_driver",
                ] if c in ens_df.columns]

                ens_display = (
                    ens_df[ens_cols]
                    .sort_values("ml_anomaly_score")
                    .head(20)
                    .rename(columns={
                        "station_name":      "Station",
                        "state":             "State",
                        "temperature_c":     "Temp (C)",
                        "humidity_pct":      "Humidity (%)",
                        "ml_anomaly_score":  "IF Score",
                        "lof_anomaly_score": "LOF Score",
                        "ensemble_severity": "Severity",
                        "ml_primary_driver": "Driver",
                    })
                )
                st.dataframe(
                    ens_display,
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "Ensemble anomalies are flagged by both "
                    "Isolation Forest and Local Outlier Factor independently. "
                    "IF Score: more negative = more isolated. "
                    "LOF Score: more negative = lower local density."
                )

        
        # Top anomalies
        st.subheader("Most Anomalous Observations")
        detail_cols = [c for c in [
            "station_name", "state", "temperature_c",
            "feels_like_c", "humidity_pct", "wind_speed_kmh",
            "ml_anomaly_score", "ml_severity", "ml_primary_driver"
        ] if c in df.columns]
        top_anom = (
            ml_df[detail_cols].sort_values("ml_anomaly_score").head(20)
            .rename(columns={
                "station_name":      "Station",
                "state":             "State",
                "temperature_c":     "Temp (C)",
                "feels_like_c":      "Feels Like (C)",
                "humidity_pct":      "Humidity (%)",
                "wind_speed_kmh":    "Wind (km/h)",
                "ml_anomaly_score":  "Score",
                "ml_severity":       "Severity",
                "ml_primary_driver": "Driver",
            })
        )
        st.dataframe(top_anom, use_container_width=True, hide_index=True)

    elif anomaly_available:
        st.info("Showing BARRA2-based anomalies.")
        anom_df = (
            df.groupby("station_name")["temp_anomaly_c"]
            .mean().dropna().reset_index()
            .sort_values("temp_anomaly_c", ascending=False)
        )
        fig_a = px.bar(
            anom_df.head(40),
            x="station_name", y="temp_anomaly_c",
            color="temp_anomaly_c",
            color_continuous_scale=COLOUR_SCALE_DIV,
            color_continuous_midpoint=0,
            template=T["plotly_tmpl"],
            labels={"temp_anomaly_c": "Anomaly (C)",
                    "station_name": "Station"},
        )
        fig_a.add_hline(y=0, line_color=T["border"], opacity=0.5)
        chart_layout(
            fig_a,
            title="Temperature Anomaly vs BARRA2 Historical Baseline",
            height=420, xangle=-45
        )
        st.plotly_chart(fig_a, use_container_width=True)

    else:
        st.info(
            "Showing statistical outliers from live BOM data. "
            "Run `python main.py` to enable ML detection."
        )
        if "temperature_c" in df.columns:
            temps  = pd.to_numeric(df["temperature_c"], errors="coerce")
            t_mean = temps.mean()
            t_std  = temps.std()
            odf    = (
                df.groupby("station_name")
                .agg(temperature_c=("temperature_c", "mean"),
                     state=("state", "first"))
                .reset_index()
            )
            odf["temperature_c"] = pd.to_numeric(
                odf["temperature_c"], errors="coerce"
            )
            odf["z_score"] = ((odf["temperature_c"] - t_mean) / t_std).round(2)

            c1, c2, c3 = st.columns(3)
            c1.metric("Hot outliers",  int((odf["z_score"] >  2).sum()))
            c2.metric("Normal range",  int((odf["z_score"].abs() <= 2).sum()))
            c3.metric("Cold outliers", int((odf["z_score"] < -2).sum()))

            fig_z = px.bar(
                odf.sort_values("z_score", ascending=False).head(50),
                x="station_name", y="z_score",
                color="z_score",
                color_continuous_scale=COLOUR_SCALE_DIV,
                color_continuous_midpoint=0,
                text_auto=".1f", template=T["plotly_tmpl"],
                labels={"z_score": "Z-Score", "station_name": "Station"},
            )
            fig_z.add_hline(y=2,  line_dash="dash",
                            line_color=T["critical"], opacity=0.5)
            fig_z.add_hline(y=-2, line_dash="dash",
                            line_color=T["accent"],   opacity=0.5)
            chart_layout(
                fig_z,
                title="Temperature Z-Scores (top 50 stations)",
                height=440, xangle=-45
            )
            st.plotly_chart(fig_z, use_container_width=True)

    


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 -- BARRA2 COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("BOM Live vs BARRA2 Historical Baseline")

    if not barra2_available:
        st.info(
            "BARRA2 historical baseline is not yet loaded. "
            "Run `python main.py` to fetch it."
        )
        st.markdown("""
        **When BARRA2 loads this tab shows:**
        - BOM current vs BARRA2 historical mean per variable
        - Stations above and below historical norms
        - Difference charts per region
        """)
        if not summary.empty:
            disp_map = {
                "station_name":       "Station",
                "state":              "State",
                "mean_temperature_c": "Avg Temp (C)",
                "max_temperature_c":  "Max Temp (C)",
                "mean_humidity_pct":  "Avg Humidity (%)",
                "mean_wind_speed_kmh":"Avg Wind (km/h)",
                "record_count":       "Records",
            }
            avail = {k: v for k, v in disp_map.items()
                     if k in summary.columns}
            disp  = summary[list(avail.keys())].copy()
            disp.rename(columns=avail, inplace=True)
            for col in disp.select_dtypes("float").columns:
                disp[col] = disp[col].round(1)
            st.dataframe(
                disp.sort_values("Avg Temp (C)", ascending=False)
                if "Avg Temp (C)" in disp.columns else disp,
                use_container_width=True, hide_index=True,
            )
    else:
        barra2_cols = [c for c in df.columns if c.startswith("barra2_")]

        pair_opts = {}
        if "temperature_c" in df.columns and "barra2_temp_c" in df.columns:
            pair_opts["Temperature (C)"] = ("temperature_c", "barra2_temp_c")
        if "humidity_pct" in df.columns and "barra2_humidity_pct" in df.columns:
            pair_opts["Humidity (%)"] = ("humidity_pct", "barra2_humidity_pct")
        if "wind_speed_kmh" in df.columns and "barra2_wind_kmh" in df.columns:
            pair_opts["Wind (km/h)"] = ("wind_speed_kmh", "barra2_wind_kmh")

        if pair_opts:
            sel_pair = st.selectbox("Compare variable",
                                    list(pair_opts.keys()))
            bom_c, b2_c = pair_opts[sel_pair]

            agg = {bom_c: "mean", b2_c: "first"}
            if "state" in df.columns:
                agg["state"] = "first"

            comp_df = (
                df.groupby("station_name").agg(agg)
                .reset_index()
                .dropna()
                .sort_values(bom_c, ascending=False)
                .head(40)
            )
            for c in [bom_c, b2_c]:
                comp_df[c] = pd.to_numeric(
                    comp_df[c], errors="coerce"
                ).round(1)

            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                name="BOM Current",
                x=comp_df["station_name"],
                y=comp_df[bom_c],
                marker_color=T["critical"],
                opacity=0.85,
            ))
            fig_comp.add_trace(go.Bar(
                name="BARRA2 Historical",
                x=comp_df["station_name"],
                y=comp_df[b2_c],
                marker_color=T["accent"],
                opacity=0.85,
            ))
            fig_comp.update_layout(barmode="group")
            chart_layout(
                fig_comp,
                title=f"BOM Current vs BARRA2 Historical -- {sel_pair}",
                height=460, xangle=-45, legend=True,
                margin=dict(t=50, b=60, l=44, r=16),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            comp_df["difference"] = (
                comp_df[bom_c] - comp_df[b2_c]
            ).round(2)
            fig_diff = px.bar(
                comp_df.sort_values("difference", ascending=False),
                x="station_name", y="difference",
                color="difference",
                color_continuous_scale=COLOUR_SCALE_DIV,
                color_continuous_midpoint=0,
                template=T["plotly_tmpl"],
                labels={"station_name": "Station",
                        "difference": f"Difference ({sel_pair})"},
            )
            fig_diff.add_hline(y=0, line_color=T["border"], opacity=0.4)
            chart_layout(
                fig_diff,
                title=f"Difference: BOM minus BARRA2 Historical -- {sel_pair}",
                height=360, xangle=-45,
            )
            st.plotly_chart(fig_diff, use_container_width=True)

            if "barra2_baseline_year" in df.columns:
                yr = df["barra2_baseline_year"].dropna().iloc[0] \
                    if not df["barra2_baseline_year"].dropna().empty else "?"
                mo = df["barra2_baseline_month"].dropna().iloc[0] \
                    if "barra2_baseline_month" in df.columns else "?"
                st.caption(
                    f"BARRA2 baseline: {yr}-{int(mo):02d} | "
                    f"12km resolution AUS-11 grid | "
                    f"Source: NCI THREDDS ob53 CC-BY 4.0"
                )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 -- DATA TABLE
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Station Summary Table")

    if summary.empty:
        st.info("Run `python main.py` to generate station summaries.")
    else:
        disp_map = {
            "station_name":        "Station",
            "state":               "State",
            "mean_temperature_c":  "Avg Temp (C)",
            "max_temperature_c":   "Max Temp (C)",
            "min_temperature_c":   "Min Temp (C)",
            "temp_range_c":        "Range (C)",
            "mean_feels_like_c":   "Feels Like (C)",
            "mean_humidity_pct":   "Humidity (%)",
            "mean_wind_speed_kmh": "Wind (km/h)",
            "max_gust_kmh":        "Gust (km/h)",
            "mean_pressure_hpa":   "Pressure (hPa)",
            "heat_stress_score":   "Heat Stress",
            "vulnerability_index": "Vulnerability",
            "barra2_temp_c":       "BARRA2 Temp (C)",
            "temp_anomaly_c":      "Temp Anomaly (C)",
            "record_count":        "Records",
            "latitude":            "Lat",
            "longitude":           "Lon",
        }
        avail = {k: v for k, v in disp_map.items() if k in summary.columns}
        disp  = summary[list(avail.keys())].copy()
        disp.rename(columns=avail, inplace=True)
        for col in disp.select_dtypes("float").columns:
            disp[col] = disp[col].round(1)

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            if "State" in disp.columns:
                flt = st.multiselect(
                    "Filter by State",
                    options=sorted(disp["State"].dropna().unique().tolist()),
                    key="tbl_state",
                )
                if flt:
                    disp = disp[disp["State"].isin(flt)]
        with fc2:
            sort_opts = [c for c in disp.columns
                         if c not in ["Station", "State", "Lat", "Lon"]]
            sort_col  = st.selectbox("Sort by", sort_opts, key="tbl_sort")
        with fc3:
            sort_asc = st.radio("Order",
                                ["Descending", "Ascending"],
                                horizontal=True, key="tbl_order")

        if sort_col in disp.columns:
            disp = disp.sort_values(
                sort_col, ascending=(sort_asc == "Ascending")
            )

        st.dataframe(disp, use_container_width=True,
                     hide_index=True, height=560)

        dl1, dl2, _ = st.columns([1, 1, 4])
        with dl1:
            st.download_button(
                "Download CSV",
                data=disp.to_csv(index=False),
                file_name="ausclimate_station_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "Download JSON",
                data=disp.to_json(orient="records", indent=2),
                file_name="ausclimate_station_summary.json",
                mime="application/json",
                use_container_width=True,
            )

    with st.expander("Raw merged data (first 500 rows)"):
        if not df.empty:
            st.dataframe(df.head(500), use_container_width=True,
                         hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 -- ENERGY (AEMO)
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.subheader("Energy Demand vs Climate Conditions")
    st.caption(
        "AEMO 5-minute dispatch data for NEM regions overlaid with "
        "live BOM temperature readings. WA and NT are not on the NEM."
    )

    if st.button("Refresh AEMO Data", key="aemo_ref"):
        with st.spinner("Fetching AEMO dispatch..."):
            try:
                from ingestion.aemo import (
                    fetch_aemo_dispatch, merge_aemo_with_bom
                )
                from pipeline.storage import save_parquet as sp
                new_aemo   = fetch_aemo_dispatch(force_refresh=True)
                sp(new_aemo, "data/processed/aemo_latest.parquet")
                new_merged = merge_aemo_with_bom(df, new_aemo)
                if not new_merged.empty:
                    sp(new_merged, "data/processed/aemo_bom_merged.parquet")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"AEMO refresh failed: {e}")

    if not aemo_available:
        st.info(
            "AEMO data is not yet loaded or shows placeholder values. "
            "Click Refresh above or run `python main.py`."
        )
    else:
        total_dem     = df_aemo["total_demand_mw"].sum()
        has_price     = ("rrp_per_mwh" in df_aemo.columns
                         and df_aemo["rrp_per_mwh"].notna().any())
        max_price     = df_aemo["rrp_per_mwh"].max() if has_price else 0
        max_price_rgn = ""
        if has_price:
            idx = df_aemo["rrp_per_mwh"].idxmax()
            max_price_rgn = df_aemo.loc[idx, "region_name"] \
                if "region_name" in df_aemo.columns else ""

        settle_str = ""
        if "settlement_date" in df_aemo.columns:
            t = df_aemo["settlement_date"].dropna()
            if not t.empty:
                settle_str = str(t.iloc[0])[:16]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total NEM Demand", f"{total_dem:,.0f} MW")
        c2.metric("Highest Spot Price", f"${max_price:,.0f}/MWh",
                  max_price_rgn)
        c3.metric("NEM Regions", "5")
        c4.metric("Last Updated", settle_str or "Just now")

        st.divider()

        cl, cr = st.columns(2)
        with cl:
            dem_df = df_aemo.dropna(subset=["total_demand_mw"]).copy()
            dem_df = dem_df.sort_values("total_demand_mw", ascending=False)
            xc     = "region_name" \
                if "region_name" in dem_df.columns else "region"
            fig_dem = px.bar(
                dem_df, x=xc, y="total_demand_mw",
                color="total_demand_mw",
                color_continuous_scale=COLOUR_SCALE_SEQ,
                text_auto=".0f", template=T["plotly_tmpl"],
                labels={"total_demand_mw": "Demand (MW)", xc: "Region"},
            )
            fig_dem.update_traces(textposition="outside")
            chart_layout(fig_dem, title="Current Demand by Region",
                         height=340)
            st.plotly_chart(fig_dem, use_container_width=True)

        with cr:
            if has_price:
                pr_df = df_aemo.dropna(subset=["rrp_per_mwh"]).copy()
                pr_df = pr_df.sort_values("rrp_per_mwh", ascending=False)
                xc2   = "region_name" \
                    if "region_name" in pr_df.columns else "region"
                fig_pr = px.bar(
                    pr_df, x=xc2, y="rrp_per_mwh",
                    color="rrp_per_mwh",
                    color_continuous_scale="RdYlGn_r",
                    text_auto=".0f", template=T["plotly_tmpl"],
                    labels={"rrp_per_mwh": "$/MWh", xc2: "Region"},
                )
                fig_pr.update_traces(textposition="outside")
                chart_layout(fig_pr, title="Spot Price by Region",
                             height=340)
                st.plotly_chart(fig_pr, use_container_width=True)
            else:
                st.info("Spot price not available in this fetch.")

        st.divider()

        # Temperature vs demand
        st.subheader("Temperature vs Electricity Demand")
        if (not df_aemo_bom.empty
                and "avg_temp_c" in df_aemo_bom.columns
                and "total_demand_mw" in df_aemo_bom.columns):
            corr_df = df_aemo_bom.dropna(
                subset=["avg_temp_c", "total_demand_mw"]
            )
            if not corr_df.empty:
                xl   = "region_name" \
                    if "region_name" in corr_df.columns else "region"
                hover = {"region": True, "avg_temp_c": ":.1f",
                         "total_demand_mw": ":,.0f", "states": True}
                if has_price and "rrp_per_mwh" in corr_df.columns:
                    hover["rrp_per_mwh"] = ":,.0f"
                fig_corr = px.scatter(
                    corr_df, x="avg_temp_c", y="total_demand_mw",
                    text=xl, size=[40] * len(corr_df),
                    color="total_demand_mw",
                    color_continuous_scale=COLOUR_SCALE_SEQ,
                    hover_data=hover,
                    template=T["plotly_tmpl"],
                    labels={
                        "avg_temp_c":      "Average Temperature (C)",
                        "total_demand_mw": "Electricity Demand (MW)",
                    },
                )
                fig_corr.update_traces(
                    textposition="top center",
                    marker=dict(opacity=0.85,
                                line=dict(width=1, color=T["bg_card"]))
                )
                chart_layout(fig_corr, height=400)
                st.plotly_chart(fig_corr, use_container_width=True)

        # History
        if (not df_aemo_hist.empty
                and "settlement_date" in df_aemo_hist.columns):
            hist = df_aemo_hist.copy()
            hist["settlement_date"] = pd.to_datetime(
                hist["settlement_date"], errors="coerce"
            )
            hist["total_demand_mw"] = pd.to_numeric(
                hist["total_demand_mw"], errors="coerce"
            )
            hist = hist.dropna(
                subset=["settlement_date", "total_demand_mw"]
            )
            if not hist.empty:
                st.divider()
                st.subheader("Demand History")
                clr_col = "region_name" \
                    if "region_name" in hist.columns else "region"
                opts    = hist[clr_col].dropna().unique().tolist()
                sel_r   = st.multiselect("Regions", opts,
                                         default=opts[:3], key="aemo_h")
                hf      = hist[hist[clr_col].isin(sel_r)] \
                    if sel_r else hist
                fig_h   = px.line(
                    hf.sort_values("settlement_date"),
                    x="settlement_date", y="total_demand_mw",
                    color=clr_col, template=T["plotly_tmpl"],
                    labels={"settlement_date": "Time",
                            "total_demand_mw": "Demand (MW)",
                            clr_col: "Region"},
                )
                chart_layout(fig_h, title="NEM Regional Demand Over Time",
                             height=340, legend=True)
                st.plotly_chart(fig_h, use_container_width=True)

        # Region table
        st.divider()
        disp_src = df_aemo_bom if not df_aemo_bom.empty else df_aemo
        col_map  = {
            "region":          "Region ID",
            "region_name":     "Region",
            "states":          "States",
            "total_demand_mw": "Demand (MW)",
            "rrp_per_mwh":     "Spot Price ($/MWh)",
            "avg_temp_c":      "Avg Temp (C)",
            "max_temp_c":      "Max Temp (C)",
            "station_count":   "BOM Stations",
        }
        avail_c = {k: v for k, v in col_map.items()
                   if k in disp_src.columns}
        if avail_c:
            d2 = disp_src[list(avail_c.keys())].copy()
            d2.rename(columns=avail_c, inplace=True)
            for col in ["Demand (MW)", "Spot Price ($/MWh)",
                        "Avg Temp (C)", "Max Temp (C)"]:
                if col in d2.columns:
                    d2[col] = pd.to_numeric(
                        d2[col], errors="coerce"
                    ).round(1)
            st.dataframe(d2, use_container_width=True, hide_index=True)

        st.caption(
            "Source: AEMO NEMWeb public dispatch data | "
            "nemweb.com.au/Reports/Current/DispatchIS_Reports/ | "
            "CC-BY 4.0"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 -- CLIMATE ASSISTANT
# ─────────────────────────────────────────────────────────────────────────────
with tab7:
    st.subheader("Climate Assistant")
    st.caption(
        "Ask questions about current Australian climate conditions. "
        "Answers are grounded in live pipeline data."
    )

    def build_climate_context(df: pd.DataFrame,
                               summary: pd.DataFrame) -> str:
        lines = ["=== CURRENT AUSCLIMATE DATA ==="]
        n_stn = df["station_name"].nunique() \
            if "station_name" in df.columns else 0
        lines.append(f"Active BOM stations: {n_stn}")
        if "state" in df.columns:
            lines.append(
                f"States: {', '.join(sorted(df['state'].dropna().unique()))}"
            )
        if "temperature_c" in df.columns:
            temps = pd.to_numeric(df["temperature_c"], errors="coerce").dropna()
            lines += [
                f"National avg temp: {temps.mean():.1f}C",
                f"Highest temp: {temps.max():.1f}C",
                f"Lowest temp:  {temps.min():.1f}C",
            ]
        if not summary.empty and "mean_temperature_c" in summary.columns:
            s2 = summary.copy()
            s2["mean_temperature_c"] = pd.to_numeric(
                s2["mean_temperature_c"], errors="coerce"
            )
            top5 = s2.nlargest(5, "mean_temperature_c")
            lines.append("Hottest stations:")
            for _, r in top5.iterrows():
                lines.append(
                    f"  {r['station_name']} ({r.get('state','?')}): "
                    f"{r['mean_temperature_c']:.1f}C"
                )
        if "ml_anomaly_flag" in df.columns:
            n_f = int(df["ml_anomaly_flag"].sum())
            lines.append(f"ML anomalies flagged: {n_f}")
        if "state" in df.columns and "temperature_c" in df.columns:
            state_avg = (
                df.groupby("state")["temperature_c"]
                .apply(lambda x: pd.to_numeric(x, errors="coerce").mean())
                .dropna().sort_values(ascending=False)
            )
            lines.append("Avg temp by state:")
            for s, t in state_avg.items():
                lines.append(f"  {s}: {t:.1f}C")
        lines.append("=== END ===")
        return "\n".join(lines)

    GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
    ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

    if not GEMINI_KEY and not ANTHROPIC_KEY:
        st.warning(
            "Set `GEMINI_API_KEY` or `ANTHROPIC_API_KEY` as an "
            "environment variable then restart the dashboard."
        )
        st.code(
            "# PowerShell\n"
            "$env:GEMINI_API_KEY='your-key-here'\n"
            "python -m streamlit run dashboard/app.py",
            language="bash"
        )
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        st.markdown("**Example questions:**")
        examples = [
            "Which stations are anomalous?",
            "What is the hottest state right now?",
            "Are there compound heat events?",
            "Compare Darwin and Hobart temperatures",
            "What does BARRA2 comparison show?",
            "Which variable drives the most anomalies?",
        ]
        eg_cols = st.columns(3)
        for i, ex in enumerate(examples):
            with eg_cols[i % 3]:
                if st.button(ex, key=f"eg_{i}",
                             use_container_width=True):
                    st.session_state.chat_history.append(
                        {"role": "user", "content": ex}
                    )

        st.divider()

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input(
            "Ask about current Australian climate..."
        )

        if user_input:
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.markdown(user_input)

            context = build_climate_context(df, summary)
            system  = (
                "You are AusClimate Assistant, an AI embedded in a "
                "professional climate analytics platform used by "
                "Australian councils, government agencies, and emergency "
                "services. Answer concisely and professionally using the "
                "live data below.\n\n" + context
            )

            with st.chat_message("assistant"):
                with st.spinner("Analysing..."):
                    reply = ""
                    try:
                        if GEMINI_KEY:
                            import google.generativeai as genai
                            genai.configure(api_key=GEMINI_KEY)
                            model = genai.GenerativeModel(
                                "gemini-2.0-flash",
                                system_instruction=system
                            )
                            msgs = [
                                {"role": m["role"],
                                 "parts": [m["content"]]}
                                for m in st.session_state.chat_history
                            ]
                            resp  = model.generate_content(msgs)
                            reply = resp.text
                        elif ANTHROPIC_KEY:
                            import anthropic
                            client = anthropic.Anthropic(
                                api_key=ANTHROPIC_KEY
                            )
                            msgs   = [
                                {"role": m["role"],
                                 "content": m["content"]}
                                for m in st.session_state.chat_history
                            ]
                            resp   = client.messages.create(
                                model="claude-sonnet-4-6",
                                max_tokens=800,
                                system=system,
                                messages=msgs,
                            )
                            reply = resp.content[0].text
                    except Exception as e:
                        reply = f"Error: {e}"

                    st.markdown(reply)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": reply}
                    )

        if st.session_state.chat_history:
            if st.button("Clear conversation", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
cl, cr = st.columns([3, 1])
with cl:
    st.caption(
        "Data: Bureau of Meteorology Live JSON API | "
        "BARRA2 NCI THREDDS (BOM) | "
        "AEMO NEMWeb | "
        "AusClimate -- University of Technology Sydney 2026"
    )
with cr:
    st.caption(f"Loaded: {datetime.now().strftime('%H:%M:%S')}")