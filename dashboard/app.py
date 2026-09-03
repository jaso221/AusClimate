"""
AusClimate Dashboard  dashboard/app.py
Real-time Australian Climate Analytics Platform
UTS Capstone Project 2026
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── Theme system ──────────────────────────────────────────────────────────────

THEMES = {
    "dark": {
        "bg_primary":    "#0A0F1E",
        "bg_card":       "#0F1729",
        "bg_sidebar":    "#080D1A",
        "bg_input":      "#131D33",
        "text_primary":  "#E8EDF5",
        "text_secondary":"#7B8FAD",
        "text_muted":    "#4A5C78",
        "border":        "#1E2D4A",
        "border_hover":  "#2E4270",
        "accent":        "#1B6CF7",
        "accent_hover":  "#3B82F6",
        "success":       "#0EA472",
        "warning":       "#E6A817",
        "critical":      "#E53935",
        "plot_bg":       "rgba(0,0,0,0)",
        "paper_bg":      "rgba(0,0,0,0)",
        "plotly_tmpl":   "plotly_dark",
        "grid_color":    "rgba(255,255,255,0.04)",
        "axis_color":    "rgba(255,255,255,0.08)",
    },
    "light": {
        "bg_primary":    "#F0F4FA",
        "bg_card":       "#FFFFFF",
        "bg_sidebar":    "#E8EEF8",
        "bg_input":      "#F5F8FF",
        "text_primary":  "#0D1B35",
        "text_secondary":"#4A5C78",
        "text_muted":    "#8A9BB8",
        "border":        "#C5D2E8",
        "border_hover":  "#94AAD4",
        "accent":        "#1550C8",
        "accent_hover":  "#1B6CF7",
        "success":       "#0A7A57",
        "warning":       "#B5690A",
        "critical":      "#C41E1E",
        "plot_bg":       "rgba(0,0,0,0)",
        "paper_bg":      "rgba(0,0,0,0)",
        "plotly_tmpl":   "plotly_white",
        "grid_color":    "rgba(0,0,0,0.04)",
        "axis_color":    "rgba(0,0,0,0.08)",
    },
}

# Colour scales that work on both themes
COLOUR_SCALE_TEMP     = "RdYlBu_r"
COLOUR_SCALE_DEMAND   = "YlOrRd"
COLOUR_SCALE_DIVERGE  = "RdBu_r"
COLOUR_SCALE_BLUE     = "Blues"

# Initialise theme in session state
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

T = THEMES[st.session_state.theme]   # shorthand used everywhere below


def build_css(T: dict) -> str:
    """
    Generate complete CSS for the current theme.
    All colors are hardcoded from T so there is no dependency on
    CSS variables or Streamlit's built-in theming.
    """
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset and base ─────────────────────────────────────────── */
html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont,
                 'Segoe UI', system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
}}
#MainMenu, footer, header {{ visibility: hidden; }}

/* ── App background ──────────────────────────────────────────── */
.stApp {{
    background-color: {T["bg_primary"]};
}}

/* ── Sidebar ─────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background-color: {T["bg_sidebar"]};
    border-right: 1px solid {T["border"]};
}}
section[data-testid="stSidebar"] .stMarkdown p {{
    color: {T["text_secondary"]};
    font-size: 12px;
    line-height: 1.6;
}}
section[data-testid="stSidebar"] label {{
    color: {T["text_secondary"]} !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}}

/* ── Metric cards ────────────────────────────────────────────── */
div[data-testid="metric-container"] {{
    background-color: {T["bg_card"]};
    border: 1px solid {T["border"]};
    border-radius: 6px;
    padding: 16px 20px 14px;
    transition: border-color 0.15s ease;
}}
div[data-testid="metric-container"]:hover {{
    border-color: {T["border_hover"]};
}}
div[data-testid="metric-container"] label {{
    color: {T["text_secondary"]} !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {T["text_primary"]} !important;
    font-size: 22px !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {{
    font-size: 11px !important;
    color: {T["text_secondary"]} !important;
}}

/* ── Tabs ────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background-color: {T["bg_card"]};
    border-radius: 6px 6px 0 0;
    padding: 4px 4px 0;
    border: 1px solid {T["border"]};
    border-bottom: none;
    gap: 2px;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: transparent;
    border-radius: 4px 4px 0 0;
    color: {T["text_secondary"]};
    font-size: 12px;
    font-weight: 500;
    padding: 8px 14px;
    border: none;
    letter-spacing: 0.01em;
    transition: all 0.15s ease;
}}
.stTabs [aria-selected="true"] {{
    background-color: {T["bg_primary"]} !important;
    color: {T["text_primary"]} !important;
    font-weight: 600 !important;
    border-bottom: 2px solid {T["accent"]} !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {T["text_primary"]};
    background-color: {T["bg_primary"]}88;
}}
.stTabs [data-baseweb="tab-panel"] {{
    background-color: {T["bg_primary"]};
    border: 1px solid {T["border"]};
    border-top: none;
    border-radius: 0 0 6px 6px;
    padding: 20px;
}}

/* ── Buttons ─────────────────────────────────────────────────── */
.stButton > button {{
    background-color: {T["bg_card"]};
    border: 1px solid {T["border"]};
    color: {T["text_secondary"]};
    border-radius: 5px;
    font-size: 12px;
    font-weight: 500;
    padding: 6px 14px;
    transition: all 0.15s ease;
    letter-spacing: 0.01em;
}}
.stButton > button:hover {{
    background-color: {T["bg_input"]};
    border-color: {T["accent"]};
    color: {T["text_primary"]};
}}

/* ── Select and multiselect ──────────────────────────────────── */
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div {{
    background-color: {T["bg_input"]};
    border: 1px solid {T["border"]};
    border-radius: 5px;
    color: {T["text_primary"]};
    font-size: 13px;
}}

/* ── Radio buttons ───────────────────────────────────────────── */
.stRadio label {{
    color: {T["text_secondary"]} !important;
    font-size: 12px !important;
}}

/* ── DataFrames ──────────────────────────────────────────────── */
.stDataFrame {{
    border: 1px solid {T["border"]};
    border-radius: 6px;
    overflow: hidden;
}}
.stDataFrame [data-testid="stDataFrameResizable"] {{
    background-color: {T["bg_card"]};
}}

/* ── Alert boxes ─────────────────────────────────────────────── */
.stSuccess {{
    background-color: {T["success"]}18;
    border: 1px solid {T["success"]}50;
    border-radius: 5px;
    color: {T["success"]};
}}
.stInfo {{
    background-color: {T["accent"]}15;
    border: 1px solid {T["accent"]}40;
    border-radius: 5px;
    color: {T["accent"]};
}}
.stWarning {{
    background-color: {T["warning"]}18;
    border: 1px solid {T["warning"]}50;
    border-radius: 5px;
    color: {T["warning"]};
}}
.stError {{
    background-color: {T["critical"]}15;
    border: 1px solid {T["critical"]}40;
    border-radius: 5px;
    color: {T["critical"]};
}}

/* ── Chat messages ───────────────────────────────────────────── */
[data-testid="stChatMessage"] {{
    background-color: {T["bg_card"]};
    border: 1px solid {T["border"]};
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 8px;
}}
[data-testid="stChatInputContainer"] textarea {{
    background-color: {T["bg_input"]};
    border: 1px solid {T["border"]};
    color: {T["text_primary"]};
    font-size: 13px;
}}

/* ── Divider ─────────────────────────────────────────────────── */
hr {{
    border-color: {T["border"]};
    margin: 18px 0;
}}

/* ── Expander ────────────────────────────────────────────────── */
.streamlit-expanderHeader {{
    background-color: {T["bg_card"]};
    border: 1px solid {T["border"]};
    border-radius: 5px;
    color: {T["text_secondary"]};
    font-size: 12px;
    font-weight: 500;
}}

/* ── Download button ─────────────────────────────────────────── */
.stDownloadButton > button {{
    background-color: {T["accent"]};
    border: 1px solid {T["accent"]};
    color: #ffffff;
    font-size: 12px;
    font-weight: 500;
    border-radius: 5px;
}}
.stDownloadButton > button:hover {{
    background-color: {T["accent_hover"]};
    border-color: {T["accent_hover"]};
}}

/* ── Scrollbar ───────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {T["bg_primary"]}; }}
::-webkit-scrollbar-thumb {{
    background: {T["border"]};
    border-radius: 3px;
}}
::-webkit-scrollbar-thumb:hover {{ background: {T["border_hover"]}; }}

/* ── Status header ───────────────────────────────────────────── */
.aus-header {{
    background-color: {T["bg_card"]};
    border: 1px solid {T["border"]};
    border-radius: 6px;
    padding: 14px 20px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
}}
.aus-header-title {{
    font-size: 17px;
    font-weight: 600;
    color: {T["text_primary"]};
    letter-spacing: -0.02em;
}}
.aus-header-sub {{
    font-size: 11px;
    color: {T["text_secondary"]};
    margin-top: 2px;
    letter-spacing: 0.01em;
}}
.aus-badge {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 99px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}
.aus-badge-live {{
    background: {T["success"]}18;
    color: {T["success"]};
    border: 1px solid {T["success"]}40;
}}
.aus-badge-pending {{
    background: {T["warning"]}18;
    color: {T["warning"]};
    border: 1px solid {T["warning"]}40;
}}
.aus-badge-offline {{
    background: {T["text_muted"]}18;
    color: {T["text_muted"]};
    border: 1px solid {T["text_muted"]}40;
}}

/* ── Section label ───────────────────────────────────────────── */
.aus-section-label {{
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {T["text_muted"]};
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid {T["border"]};
}}

/* ── Sidebar logo area ───────────────────────────────────────── */
.aus-sidebar-brand {{
    padding: 4px 0 16px;
    border-bottom: 1px solid {T["border"]};
    margin-bottom: 16px;
}}
.aus-sidebar-brand-title {{
    font-size: 16px;
    font-weight: 700;
    color: {T["text_primary"]};
    letter-spacing: -0.02em;
}}
.aus-sidebar-brand-sub {{
    font-size: 10px;
    color: {T["text_muted"]};
    letter-spacing: 0.04em;
    margin-top: 2px;
}}

/* ── Spinner ─────────────────────────────────────────────────── */
.stSpinner > div {{
    border-top-color: {T["accent"]};
}}
</style>
"""


# Inject CSS immediately after function definition
st.markdown(build_css(T), unsafe_allow_html=True)


def chart_layout(fig,
                 title: str = "",
                 height: int = 400,
                 xangle: int = 0,
                 legend: bool = False,
                 margin: dict = None) -> object:
    """
    Apply consistent professional layout to any plotly figure.
    Respects current theme automatically.
    """
    if margin is None:
        margin = dict(t=44, b=36, l=44, r=16)

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                size=13,
                color=T["text_primary"],
                family="Inter, -apple-system, sans-serif"
            ),
            x=0,
            xanchor="left",
            pad=dict(l=0, t=0),
        ),
        plot_bgcolor=T["plot_bg"],
        paper_bgcolor=T["paper_bg"],
        font=dict(
            family="Inter, -apple-system, sans-serif",
            size=11,
            color=T["text_secondary"],
        ),
        height=height,
        margin=margin,
        showlegend=legend,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=T["border"],
            borderwidth=1,
            font=dict(size=11, color=T["text_secondary"]),
        ),
        xaxis=dict(
            gridcolor=T["grid_color"],
            linecolor=T["axis_color"],
            tickfont=dict(size=10, color=T["text_secondary"]),
            tickangle=xangle,
            title_font=dict(size=11, color=T["text_secondary"]),
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor=T["grid_color"],
            linecolor=T["axis_color"],
            tickfont=dict(size=10, color=T["text_secondary"]),
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


from config import (
    PROCESSED_DATA_PATH,
    MERGED_DATA_PATH,
    STATION_SUMMARY_PATH,
    BARRA2_CACHE_PATH,
    AEMO_REGION_LABELS,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AusClimate",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-label {
        font-size: 12px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 600;
        color: #fff;
        line-height: 1;
    }
    .metric-sub {
        font-size: 11px;
        color: #aaa;
        margin-top: 4px;
    }
    .status-ok   { color: #4ade80; font-weight: 600; }
    .status-warn { color: #facc15; font-weight: 600; }
    .status-err  { color: #f87171; font-weight: 600; }
    .section-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #666;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

COLOUR_SCALE = "RdYlBu_r"
PLOT_BG      = "rgba(0,0,0,0)"


# ── Data loading ──────────────────────────────────────────────────────────────


@st.cache_data(ttl=1800)
def load_all_data():
    """Load all pipeline output files. Returns empty DataFrames on missing."""
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
    """Load AEMO pipeline output files. 5-minute cache matches AEMO update frequency."""
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
    """Load saved SHAP values and global importance."""
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

df_shap, df_shap_global = load_shap_data()    
df, summary = load_all_data()
df_aemo, df_aemo_bom, df_aemo_history = load_aemo_data()



def check_barra2_has_data(dataframe: pd.DataFrame) -> bool:
    """True only if barra2_temp_c column exists AND has non-NaN values."""
    if "barra2_temp_c" not in dataframe.columns:
        return False
    return bool(dataframe["barra2_temp_c"].notna().any())


def check_anomaly_has_data(dataframe: pd.DataFrame) -> bool:
    """True only if temp_anomaly_c exists AND has non-NaN values."""
    if "temp_anomaly_c" not in dataframe.columns:
        return False
    return bool(dataframe["temp_anomaly_c"].notna().any())


barra2_available = check_barra2_has_data(df)
anomaly_available = check_anomaly_has_data(df)


# ── Filter data (computed before sidebar so counts are available) ──────────────

if df.empty:
    st.warning("No data found. Run `python main.py` first.")
    st.stop()

all_states = sorted(df["state"].dropna().unique().tolist()) \
    if "state" in df.columns else []

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:

    # Brand block
    st.markdown(f"""
    <div class="aus-sidebar-brand">
        <div class="aus-sidebar-brand-title">AusClimate</div>
        <div class="aus-sidebar-brand-sub">REAL-TIME CLIMATE ANALYTICS</div>
    </div>
    """, unsafe_allow_html=True)

    # Theme toggle
    st.markdown(
        '<p class="aus-section-label">Display</p>',
        unsafe_allow_html=True
    )
    theme_btn_label = "Switch to Light Mode" \
        if st.session_state.theme == "dark" \
        else "Switch to Dark Mode"
    if st.button(theme_btn_label,
                 use_container_width=True,
                 key="theme_toggle"):
        st.session_state.theme = (
            "light"
            if st.session_state.theme == "dark"
            else "dark"
        )
        st.rerun()

    st.divider()

    # Data source status
    st.markdown(
        '<p class="aus-section-label">Data Sources</p>',
        unsafe_allow_html=True
    )

    n_stations     = df["station_name"].nunique() \
        if "station_name" in df.columns else 0
    bom_cls        = "aus-badge-live" \
        if n_stations > 0 else "aus-badge-offline"
    bom_label      = f"LIVE {n_stations:,} STN" \
        if n_stations > 0 else "OFFLINE"
    barra2_cls     = "aus-badge-live" \
        if barra2_available else "aus-badge-pending"
    barra2_label   = "ACTIVE" if barra2_available else "PENDING"
    era5_cls   = "aus-badge-offline"
    era5_label = "NO KEY"
    aemo_available = (
        not df_aemo.empty
        and "total_demand_mw" in df_aemo.columns
        and df_aemo["total_demand_mw"].notna().any()
    )
    aemo_cls       = "aus-badge-live" \
        if aemo_available else "aus-badge-pending"
    aemo_label     = "LIVE" if aemo_available else "PENDING"

    st.markdown(f"""
    <div style="display:flex; flex-direction:column; gap:7px;
                margin-bottom:14px;">
        <div style="display:flex; justify-content:space-between;
                    align-items:center;">
            <span style="font-size:11px;
                         color:{T['text_secondary']};">BOM</span>
            <span class="aus-badge {bom_cls}">{bom_label}</span>
        </div>
        <div style="display:flex; justify-content:space-between;
                    align-items:center;">
            <span style="font-size:11px;
                         color:{T['text_secondary']};">BARRA2</span>
            <span class="aus-badge {barra2_cls}">{barra2_label}</span>
        </div>
        <div style="display:flex; justify-content:space-between;
                    align-items:center;">
            <span style="font-size:11px;
                         color:{T['text_secondary']};">AEMO</span>
            <span class="aus-badge {aemo_cls}">{aemo_label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Refresh button
    if st.button("Refresh BOM Data",
                 use_container_width=True,
                 key="refresh_btn"):
        with st.spinner("Refreshing..."):
            try:
                from ingestion.bom import (
                    fetch_all_stations, get_station_coords
                )
                from pipeline.cleaner import clean_bom_data
                from pipeline.feature_eng import compute_station_summary
                from pipeline.storage import save_parquet
                from pipeline.spatial_join import (
                    merge_bom_barra2, merge_era5
                )

                df_raw   = fetch_all_stations()
                df_clean = clean_bom_data(df_raw)
                save_parquet(df_clean, PROCESSED_DATA_PATH)

                barra2_cached = pd.DataFrame()
                if os.path.exists(BARRA2_CACHE_PATH):
                    try:
                        barra2_cached = pd.read_parquet(BARRA2_CACHE_PATH)
                    except Exception:
                        pass

                merged = merge_bom_barra2(df_clean, barra2_cached)
                save_parquet(merged, MERGED_DATA_PATH)
                feat = compute_station_summary(merged)
                save_parquet(feat, STATION_SUMMARY_PATH)
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Refresh failed: {e}")

    st.divider()

    # Filters
    st.markdown(
        '<p class="aus-section-label">Filters</p>',
        unsafe_allow_html=True
    )

    selected_states = st.multiselect(
        "States",
        options=all_states,
        default=all_states,
    )

    df_state = (
        df[df["state"].isin(selected_states)]
        if selected_states and "state" in df.columns
        else df
    )

    all_stations_list = sorted(
        df_state["station_name"].dropna().unique().tolist()
    ) if "station_name" in df_state.columns else []

    selected_stations = st.multiselect(
        "Stations",
        options=all_stations_list,
        default=all_stations_list[:12]
        if len(all_stations_list) > 12
        else all_stations_list,
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

    selected_label = st.selectbox(
        "Variable", list(FEATURE_OPTIONS.keys())
    )
    selected_col = FEATURE_OPTIONS[selected_label]

    chart_type = st.radio(
        "Chart Type",
        ["Bar", "Horizontal Bar", "Line",
         "Scatter", "Box", "Violin",
         "Radar", "Heatmap", "Bubble"],
    )


# ── Filter data based on sidebar selections ───────────────────────────────────

if selected_stations:
    df_filtered = df[
        df["station_name"].isin(selected_stations)
    ].copy()
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

from datetime import datetime as _dt
now_str = _dt.now().strftime("%d %b %Y  %H:%M AEST")

st.markdown(f"""
<div class="aus-header">
    <div>
        <div class="aus-header-title">AusClimate</div>
        <div class="aus-header-sub">
            Real-Time Australian Climate Analytics Platform
            &nbsp;|&nbsp; University of Technology Sydney
        </div>
    </div>
    <div style="display:flex; align-items:center;
                gap:8px; flex-wrap:wrap;">
        <span class="aus-badge {bom_cls}">
            BOM {n_stations:,} Stations
        </span>
        <span class="aus-badge {barra2_cls}">
            BARRA2 {barra2_label}
        </span>
        <span class="aus-badge {aemo_cls}">
            AEMO {aemo_label}
        </span>
        <span style="font-size:10px;
                     color:{T['text_muted']};
                     margin-left:8px;
                     letter-spacing:0.04em;">
            {now_str}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Tab content follows below here...



# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview",
    "🗺️ Station Map",
    "⚠️ Anomalies",
    "📈 BARRA2 Comparison",
    "📋 Data Table",
    "⚡ Energy",
    "🤖 Climate Assistant",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Live Australian Climate Overview")

    # ── Top metrics ───────────────────────────────────────────────
    total_stations = df["station_name"].nunique() \
        if "station_name" in df.columns else 0
    total_states   = df["state"].nunique() \
        if "state" in df.columns else 0
    total_records  = len(df)

    vals = pd.to_numeric(df_filtered[selected_col], errors="coerce").dropna()

    if not vals.empty:
        max_val = vals.max()
        min_val = vals.min()
        avg_val = vals.mean()
        rng_val = max_val - min_val

        max_idx = pd.to_numeric(
            df_filtered[selected_col], errors="coerce"
        ).idxmax()
        min_idx = pd.to_numeric(
            df_filtered[selected_col], errors="coerce"
        ).idxmin()
        max_station = df_filtered.loc[max_idx, "station_name"] \
            if "station_name" in df_filtered.columns else ""
        min_station = df_filtered.loc[min_idx, "station_name"] \
            if "station_name" in df_filtered.columns else ""
    else:
        max_val = min_val = avg_val = rng_val = 0
        max_station = min_station = ""

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Stations",   f"{total_stations:,}")
    c2.metric("States",           total_states)
    c3.metric("Total Records",    f"{total_records:,}")
    c4.metric(f"Avg {selected_label}", f"{avg_val:.1f}")
    c5.metric(f"Highest",         f"{max_val:.1f}", max_station)
    c6.metric(f"Lowest",          f"{min_val:.1f}", min_station)

    st.divider()

    # ── Main chart ────────────────────────────────────────────────
    if station_mean.empty:
        st.info("Select stations from the sidebar to see the chart.")
    else:
        if chart_type == "Bar":
            fig = px.bar(
                station_mean,
                x="Station", y=selected_label,
                color=selected_label,
                color_continuous_scale=COLOUR_SCALE,
                text_auto=".1f",
                height=440,
                title=f"Average {selected_label} by Station"
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                xaxis_tickangle=-40,
                coloraxis_showscale=False
            )

        elif chart_type == "Horizontal Bar":
            fig = px.bar(
                station_mean.sort_values(selected_label),
                x=selected_label, y="Station",
                orientation="h",
                color=selected_label,
                color_continuous_scale=COLOUR_SCALE,
                text_auto=".1f",
                height=max(440, len(station_mean) * 22),
                title=f"Average {selected_label} by Station"
            )
            fig.update_layout(coloraxis_showscale=False)

        elif chart_type == "Line":
            if "timestamp" in df_filtered.columns:
                line_df = df_filtered.dropna(
                    subset=["timestamp", selected_col]
                ).sort_values("timestamp")
                fig = px.line(
                    line_df,
                    x="timestamp", y=selected_col,
                    color="station_name",
                    height=440,
                    title=f"{selected_label} Over Time",
                    labels={selected_col: selected_label,
                            "timestamp": "Time",
                            "station_name": "Station"}
                )
            else:
                fig = px.line(
                    station_mean,
                    x="Station", y=selected_label,
                    markers=True, height=440,
                    title=f"{selected_label} by Station"
                )

        elif chart_type == "Scatter":
            fig = px.scatter(
                df_filtered.dropna(subset=[selected_col]),
                x="station_name", y=selected_col,
                color="state" if "state" in df_filtered.columns else None,
                opacity=0.5, height=440,
                title=f"{selected_label} - All Observations",
                labels={selected_col: selected_label,
                        "station_name": "Station",
                        "state": "State"}
            )
        
        elif chart_type == "Box":
            fig = px.box(
                df_filtered.dropna(subset=[selected_col]),
                x="state" if "state" in df_filtered.columns
                  else "station_name",
                y=selected_col,
                color="state" if "state" in df_filtered.columns
                      else None,
                title=f"{selected_label} Distribution by State",
                height=440,
                labels={
                    selected_col: selected_label,
                    "state": "State",
                },
                points="outliers",
            )

        elif chart_type == "Violin":
            fig = px.violin(
                df_filtered.dropna(subset=[selected_col]),
                x="state" if "state" in df_filtered.columns
                  else "station_name",
                y=selected_col,
                color="state" if "state" in df_filtered.columns
                      else None,
                box=True,
                title=f"{selected_label} Distribution by State "
                      f"(Violin + Box)",
                height=440,
                labels={
                    selected_col: selected_label,
                    "state": "State",
                },
            )

        elif chart_type == "Radar":
            # Radar chart: one trace per selected state
            # Shows the relative profile of all variables
            radar_vars = {
                "temperature_c":  "Temp",
                "humidity_pct":   "Humidity",
                "wind_speed_kmh": "Wind",
                "pressure_hpa":   "Pressure",
                "feels_like_c":   "Feels Like",
            }
            radar_available = {
                k: v for k, v in radar_vars.items()
                if k in df_filtered.columns
            }

            fig = go.Figure()

            group_col = "state" if "state" in df_filtered.columns \
                        else "station_name"
            for group_val in df_filtered[group_col].dropna().unique():
                grp = df_filtered[
                    df_filtered[group_col] == group_val
                ]
                means = []
                for col in radar_available:
                    v = pd.to_numeric(
                        grp[col], errors="coerce"
                    ).mean()
                    means.append(v if not np.isnan(v) else 0)

                fig.add_trace(go.Scatterpolar(
                    r=means,
                    theta=list(radar_available.values()),
                    fill="toself",
                    name=str(group_val),
                    opacity=0.6,
                ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True)
                ),
                title="Climate Variable Radar Profile by State",
                height=480,
                showlegend=True,
            )

        elif chart_type == "Heatmap":
            # Heatmap: stations (rows) vs variables (columns)
            # Colour = normalised value
            hmap_vars = {
                "temperature_c":  "Temp (°C)",
                "humidity_pct":   "Humidity (%)",
                "wind_speed_kmh": "Wind (km/h)",
                "pressure_hpa":   "Pressure (hPa)",
                "feels_like_c":   "Feels Like (°C)",
            }
            hmap_available = {
                k: v for k, v in hmap_vars.items()
                if k in df_filtered.columns
            }

            hmap_df = df_filtered.groupby("station_name")[
                list(hmap_available.keys())
            ].mean().reset_index()
            hmap_df = hmap_df.set_index("station_name")

            for col in hmap_df.columns:
                hmap_df[col] = pd.to_numeric(
                    hmap_df[col], errors="coerce"
                )

            # Normalise each column to 0-1 for comparable colour scale
            hmap_norm = hmap_df.copy()
            for col in hmap_norm.columns:
                col_min = hmap_norm[col].min()
                col_max = hmap_norm[col].max()
                rng     = col_max - col_min
                if rng > 0:
                    hmap_norm[col] = (
                        (hmap_norm[col] - col_min) / rng
                    )

            hmap_norm.columns = list(hmap_available.values())

            fig = px.imshow(
                hmap_norm.head(40).T,
                color_continuous_scale="RdYlBu_r",
                title="Climate Variable Heatmap: "
                      "Top 40 Stations (normalised 0-1)",
                height=max(300, len(hmap_available) * 60),
                aspect="auto",
                labels=dict(
                    x="Station",
                    y="Variable",
                    color="Relative Value"
                ),
            )
            fig.update_xaxes(tickangle=-45)

        elif chart_type == "Bubble":
            # Bubble chart: temperature vs humidity
            # Bubble size = wind speed
            # Colour = state
            bubble_df = df_filtered.groupby("station_name").agg({
                "temperature_c":  "mean",
                "humidity_pct":   "mean",
                "wind_speed_kmh": "mean",
                "state":          "first",
            }).reset_index().dropna()

            for col in ["temperature_c", "humidity_pct",
                        "wind_speed_kmh"]:
                bubble_df[col] = pd.to_numeric(
                    bubble_df[col], errors="coerce"
                ).round(1)

            fig = px.scatter(
                bubble_df,
                x="temperature_c",
                y="humidity_pct",
                size="wind_speed_kmh",
                color="state" if "state" in bubble_df.columns
                      else "station_name",
                hover_name="station_name",
                size_max=30,
                title="Temperature vs Humidity: "
                      "Bubble size = Wind Speed",
                height=480,
                labels={
                    "temperature_c":  "Temperature (°C)",
                    "humidity_pct":   "Humidity (%)",
                    "wind_speed_kmh": "Wind Speed (km/h)",
                    "state":          "State",
                },
            )
            fig.update_layout(
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=PLOT_BG,
            )

        fig.update_layout(
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=PLOT_BG,
            margin=dict(t=50, b=40, l=40, r=20),
        )
        st.plotly_chart(fig, width='stretch')

    # ── Dashboard tiles ───────────────────────────────────────────
    st.divider()
    st.subheader("📌 My Dashboard")

    if "tiles" not in st.session_state:
        st.session_state.tiles = []

    btn1, btn2, _ = st.columns([1, 1, 4])
    with btn1:
        if st.button("➕ Pin Chart", width='stretch'):
            if selected_stations:
                st.session_state.tiles.append({
                    "stations": selected_stations[:],
                    "col":      selected_col,
                    "label":    selected_label,
                    "chart":    chart_type,
                })
                st.toast("Chart pinned to dashboard.")
    with btn2:
        if st.button("🗑️ Clear All", width='stretch'):
            st.session_state.tiles = []

    if st.session_state.tiles:
        tile_cols = st.columns(2)
        tiles_to_remove = []
        for i, tile in enumerate(st.session_state.tiles):
            with tile_cols[i % 2]:
                with st.container(border=True):
                    td = df[df["station_name"].isin(tile["stations"])].copy()
                    td[tile["col"]] = pd.to_numeric(
                        td[tile["col"]], errors="coerce"
                    )
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
                            title=tile["label"], height=260
                        )
                    else:
                        tf = px.bar(
                            ts, x="Station", y=tile["label"],
                            color=tile["label"],
                            color_continuous_scale=COLOUR_SCALE,
                            title=tile["label"], height=260
                        )

                    tf.update_layout(
                        coloraxis_showscale=False,
                        margin=dict(t=40, b=20, l=10, r=10),
                        plot_bgcolor=PLOT_BG,
                        paper_bgcolor=PLOT_BG,
                        xaxis_tickangle=-30,
                    )
                    st.plotly_chart(tf, width='stretch')

                    if st.button("Remove", key=f"rm_{i}"):
                        tiles_to_remove.append(i)

        for idx in reversed(tiles_to_remove):
            st.session_state.tiles.pop(idx)
        if tiles_to_remove:
            st.rerun()
    else:
        st.caption("No charts pinned yet. Select a variable and "
                   "click ➕ Pin Chart.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: STATION MAP
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("🗺️ National Station Coverage")

    if "latitude" not in df.columns or "longitude" not in df.columns:
        st.info("No coordinate data in current dataset.")
    else:
        map_df = (
            df.groupby("station_name")
            .agg({
                "latitude":      "first",
                "longitude":     "first",
                "state":         "first",
                "temperature_c": "mean",
                "humidity_pct":  "mean",
                "wind_speed_kmh":"mean",
            })
            .reset_index()
        )
        map_df["latitude"]      = pd.to_numeric(map_df["latitude"],      errors="coerce")
        map_df["longitude"]     = pd.to_numeric(map_df["longitude"],     errors="coerce")
        map_df["temperature_c"] = pd.to_numeric(map_df["temperature_c"], errors="coerce")
        map_df = map_df.dropna(subset=["latitude", "longitude"])

        # Map colour variable
        map_col_opts = {}
        for col, lbl in {
            "temperature_c":  "Temperature (°C)",
            "humidity_pct":   "Humidity (%)",
            "wind_speed_kmh": "Wind Speed (km/h)",
        }.items():
            if col in map_df.columns:
                map_col_opts[lbl] = col

        map_colour_label = st.selectbox(
            "Colour stations by", list(map_col_opts.keys()), key="map_col"
        )
        map_colour_col = map_col_opts[map_colour_label]

        fig_map = px.scatter_mapbox(
            map_df,
            lat="latitude",
            lon="longitude",
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
            mapbox_style="carto-positron",
            zoom=3.5,
            center={"lat": -26, "lon": 134},
            height=580,
            title=f"{len(map_df)} BOM Stations - coloured by {map_colour_label}",
            labels={map_colour_col: map_colour_label},
        )
        fig_map.update_layout(
            margin=dict(t=50, b=0, l=0, r=0),
            coloraxis_colorbar=dict(title=map_colour_label),
        )
        st.plotly_chart(fig_map, width='stretch')

        # State breakdown
        st.divider()
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Stations per State")
            sc = (
                df.groupby("state")["station_name"]
                .nunique().reset_index()
            )
            sc.columns = ["State", "Stations"]
            fig_sc = px.bar(
                sc.sort_values("Stations", ascending=False),
                x="State", y="Stations",
                color="Stations",
                color_continuous_scale="Blues",
                text_auto=True, height=320,
            )
            fig_sc.update_layout(
                coloraxis_showscale=False,
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=PLOT_BG,
            )
            st.plotly_chart(fig_sc, width='stretch')

        with c2:
            st.subheader("Temperature Distribution")
            temp_vals = pd.to_numeric(
                df["temperature_c"], errors="coerce"
            ).dropna()
            fig_hist = px.histogram(
                x=temp_vals,
                nbins=40,
                title="Temperature Distribution Across All Stations",
                labels={"x": "Temperature (°C)", "count": "Observations"},
                color_discrete_sequence=["#ef4444"],
                height=320,
            )
            fig_hist.update_layout(
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=PLOT_BG,
                showlegend=False,
            )
            st.plotly_chart(fig_hist, width='stretch')


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 -- ANOMALIES
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Climate Anomaly Detection")

    ml_available = (
        "ml_anomaly_flag" in df.columns
        and df["ml_anomaly_flag"].notna().any()
    )
    anomaly_available = (
        "temp_anomaly_c" in df.columns
        and df["temp_anomaly_c"].notna().any()
    )
    shap_available = (
        "shap_computed" in df.columns
        and df["shap_computed"].any()
    )

    # ── ML results (Isolation Forest) ─────────────────────────────
    if ml_available:
        st.success(
            "Isolation Forest ML model active -- "
            "anomalies detected across 5 climate variables."
        )

        ml_df      = df[df["ml_anomaly_flag"] == True].copy()
        n_total    = len(df)
        n_anomaly  = int(df["ml_anomaly_flag"].sum())
        n_severe   = int((df["ml_severity"] == "Severe").sum())   \
            if "ml_severity" in df.columns else 0
        n_moderate = int((df["ml_severity"] == "Moderate").sum()) \
            if "ml_severity" in df.columns else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Observations Scored", f"{n_total:,}")
        c2.metric("Anomalies Flagged",   f"{n_anomaly:,}")
        c3.metric("Severe",              n_severe)
        c4.metric("Moderate",            n_moderate)

        st.divider()

        col_sev, col_drv = st.columns(2)

        with col_sev:
            if "ml_severity" in df.columns:
                sev_counts = df["ml_severity"].value_counts().reset_index()
                sev_counts.columns = ["Severity", "Count"]
                sev_colours = {
                    "Severe":   "#ef4444",
                    "Moderate": "#f97316",
                    "Mild":     "#eab308",
                    "Normal":   "#22c55e",
                    "Unknown":  "#6b7280",
                }
                fig_sev = px.bar(
                    sev_counts,
                    x="Severity", y="Count",
                    color="Severity",
                    color_discrete_map=sev_colours,
                    title="Anomaly Severity Distribution",
                    height=320,
                    text_auto=True,
                )
                fig_sev.update_layout(
                    showlegend=False,
                    plot_bgcolor=PLOT_BG,
                    paper_bgcolor=PLOT_BG,
                )
                st.plotly_chart(fig_sev, use_container_width=True)

        with col_drv:
            driver_col = (
                "shap_top_driver" if shap_available
                else "ml_primary_driver"
            )
            if driver_col in df.columns:
                driver_counts = (
                    ml_df[driver_col]
                    .replace("", np.nan)
                    .dropna()
                    .value_counts()
                    .reset_index()
                )
                driver_counts.columns = ["Driver", "Count"]
                title_suffix = "(SHAP)" if shap_available else "(Phase 2)"
                if not driver_counts.empty:
                    fig_drv = px.pie(
                        driver_counts,
                        names="Driver", values="Count",
                        title=f"Primary Anomaly Driver {title_suffix}",
                        height=320,
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    st.plotly_chart(fig_drv, use_container_width=True)

        # Station risk map
        st.subheader("Station Risk Map")
        from anomaly.detector import compute_station_anomaly_summary
        station_risk = compute_station_anomaly_summary(df)

        if not station_risk.empty:
            risk_colours = {
                "High":   "#ef4444",
                "Medium": "#f97316",
                "Low":    "#eab308",
                "Normal": "#22c55e",
            }
            map_df = station_risk.dropna(subset=["latitude", "longitude"])
            if not map_df.empty:
                fig_risk = px.scatter_mapbox(
                    map_df,
                    lat="latitude",
                    lon="longitude",
                    color="station_risk",
                    color_discrete_map=risk_colours,
                    size="anomaly_count",
                    size_max=18,
                    hover_name="station_name",
                    hover_data={
                        "state":           True,
                        "anomaly_count":   True,
                        "anomaly_rate_pct":True,
                        "station_risk":    True,
                        "latitude":        False,
                        "longitude":       False,
                    },
                    mapbox_style="carto-darkmatter",
                    zoom=3.5,
                    center={"lat": -26, "lon": 134},
                    height=500,
                    title="Station Risk Map -- dot size = anomaly count",
                )
                st.plotly_chart(fig_risk, use_container_width=True)

            # Risk table
            risk_cols = {
                "station_name":      "Station",
                "state":             "State",
                "total_observations":"Total Obs",
                "anomaly_count":     "Anomalies",
                "anomaly_rate_pct":  "Rate (%)",
                "severe_count":      "Severe",
                "moderate_count":    "Moderate",
                "avg_anomaly_score": "Avg Score",
                "primary_driver":    "Driver",
                "station_risk":      "Risk",
            }
            avail_risk = {
                k: v for k, v in risk_cols.items()
                if k in station_risk.columns
            }
            disp_risk = station_risk[list(avail_risk.keys())].copy()
            disp_risk.rename(columns=avail_risk, inplace=True)
            st.dataframe(
                disp_risk.sort_values("Anomalies", ascending=False)
                if "Anomalies" in disp_risk.columns else disp_risk,
                use_container_width=True,
                hide_index=True,
            )

        # Top anomalies table
        st.subheader("Most Anomalous Observations")
        detail_cols = [c for c in [
            "station_name", "state", "temperature_c",
            "feels_like_c", "humidity_pct", "wind_speed_kmh",
            "ml_anomaly_score", "ml_severity", "ml_primary_driver"
        ] if c in df.columns]

        top_anom = (
            ml_df[detail_cols]
            .sort_values("ml_anomaly_score")
            .head(20)
            .rename(columns={
                "station_name":      "Station",
                "state":             "State",
                "temperature_c":     "Temp (C)",
                "feels_like_c":      "Feels Like (C)",
                "humidity_pct":      "Humidity (%)",
                "wind_speed_kmh":    "Wind (km/h)",
                "ml_anomaly_score":  "Score",
                "ml_severity":       "Severity",
                "ml_primary_driver": "Driver (Phase 2)",
            })
        )
        st.dataframe(top_anom, use_container_width=True, hide_index=True)

    # ── BARRA2 statistical anomalies ──────────────────────────────
    elif anomaly_available:
        st.info(
            "Showing BARRA2-based anomalies. "
            "ML Isolation Forest layer adds more detail."
        )
        anom_df = (
            df.groupby("station_name")["temp_anomaly_c"]
            .mean().dropna().reset_index()
            .sort_values("temp_anomaly_c", ascending=False)
        )
        anom_df["status"] = anom_df["temp_anomaly_c"].apply(
            lambda v:
            "Much warmer" if v >  3 else
            "Warmer"      if v >  1 else
            "Cooler"      if v < -1 else
            "Normal"
        )
        fig_a = px.bar(
            anom_df.head(40),
            x="station_name", y="temp_anomaly_c",
            color="temp_anomaly_c",
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            title="Temperature Anomaly vs BARRA2 Historical Baseline",
            height=420,
        )
        fig_a.add_hline(y=0, line_color="white", opacity=0.2)
        fig_a.update_layout(
            xaxis_tickangle=-45,
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=PLOT_BG,
        )
        st.plotly_chart(fig_a, use_container_width=True)

    else:
        # No ML or BARRA2 -- use z-score fallback
        st.info(
            "Showing statistical outliers from live BOM data. "
            "Run `python main.py` to enable ML anomaly detection."
        )
        if "temperature_c" in df.columns:
            temps = pd.to_numeric(df["temperature_c"], errors="coerce")
            t_mean = temps.mean()
            t_std  = temps.std()

            outlier_df = (
                df.groupby("station_name")
                .agg(
                    temperature_c=("temperature_c", "mean"),
                    state=("state", "first"),
                )
                .reset_index()
            )
            outlier_df["temperature_c"] = pd.to_numeric(
                outlier_df["temperature_c"], errors="coerce"
            )
            outlier_df["z_score"] = (
                (outlier_df["temperature_c"] - t_mean) / t_std
            ).round(2)
            outlier_df["flag"] = outlier_df["z_score"].apply(
                lambda z:
                "Hot outlier"  if z >  2 else
                "Cold outlier" if z < -2 else
                "Normal"
            )

            hot  = outlier_df[outlier_df["z_score"] >  2]
            cold = outlier_df[outlier_df["z_score"] < -2]

            c1, c2, c3 = st.columns(3)
            c1.metric("Hot outliers",  len(hot))
            c2.metric("Normal range",  len(outlier_df) - len(hot) - len(cold))
            c3.metric("Cold outliers", len(cold))

            fig_z = px.bar(
                outlier_df.sort_values("z_score", ascending=False).head(50),
                x="station_name", y="z_score",
                color="z_score",
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                title="Temperature Z-Scores (top 50 stations)",
                height=440,
                text_auto=".1f",
            )
            fig_z.add_hline(y=2,  line_dash="dash",
                            line_color="red",  opacity=0.5)
            fig_z.add_hline(y=-2, line_dash="dash",
                            line_color="blue", opacity=0.5)
            fig_z.add_hline(y=0, line_color="white", opacity=0.15)
            fig_z.update_layout(
                xaxis_tickangle=-45,
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=PLOT_BG,
            )
            st.plotly_chart(fig_z, use_container_width=True)

    # ── SHAP Phase 3 (appended at bottom of tab regardless of path) ────
    if shap_available:
        st.divider()
        st.subheader("Phase 3 -- SHAP Explainability")
        st.caption(
            "Shapley values show exactly how much each climate variable "
            "contributed to every anomaly score. Values are additive and "
            "sum to the total anomaly score. "
            "More negative = stronger push toward anomalous."
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
            and c not in [
                "shap_top_driver", "shap_top_driver_val",
                "shap_explanation", "shap_computed"
            ]
        ]

        # Global importance bar chart
        if not df_shap_global.empty:
            st.markdown("**Global Feature Importance Across All Anomalies**")
            fig_gimp = px.bar(
                df_shap_global,
                x="mean_abs_shap",
                y="label",
                orientation="h",
                color="mean_abs_shap",
                color_continuous_scale="RdYlBu_r",
                text=df_shap_global["mean_abs_shap"].round(4).astype(str),
                height=300,
                labels={
                    "mean_abs_shap": "Mean |SHAP|",
                    "label":         "Variable",
                }
            )
            fig_gimp.update_traces(textposition="outside")
            fig_gimp.update_layout(
                coloraxis_showscale=False,
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=PLOT_BG,
                xaxis_title="Mean Absolute SHAP Contribution",
                yaxis_title="",
            )
            st.plotly_chart(fig_gimp, use_container_width=True)

        # Beeswarm scatter
        if shap_feat_cols and "ml_anomaly_flag" in df.columns:
            st.markdown(
                "**SHAP Distribution -- All Anomalous Observations**"
            )
            st.caption(
                "Each dot is one anomalous observation. "
                "X-axis = SHAP contribution (negative = anomaly driver). "
                "Colour = actual value of that variable."
            )
            anom_df_shap = df[df["ml_anomaly_flag"] == True].copy()
            shap_long_rows = []
            for shap_col in shap_feat_cols:
                feat  = shap_col.replace("shap_", "")
                label = feat_labels.get(feat, feat)
                unit  = feat_units.get(feat, "")
                if shap_col in anom_df_shap.columns and feat in anom_df_shap.columns:
                    sub = anom_df_shap[[
                        shap_col, feat, "station_name"
                    ]].dropna().copy()
                    sub = sub.rename(columns={
                        shap_col: "shap_value",
                        feat:     "feature_value",
                    })
                    sub["feature"] = label
                    shap_long_rows.append(sub)

            if shap_long_rows:
                shap_long = pd.concat(shap_long_rows, ignore_index=True)
                fig_bee = px.scatter(
                    shap_long,
                    x="shap_value",
                    y="feature",
                    color="feature_value",
                    color_continuous_scale="RdYlBu_r",
                    opacity=0.4,
                    height=360,
                    hover_data={
                        "station_name":  True,
                        "shap_value":    ":.5f",
                        "feature_value": ":.2f",
                        "feature":       False,
                    },
                    labels={
                        "shap_value":    "SHAP Contribution",
                        "feature":       "Variable",
                        "feature_value": "Actual Value",
                    }
                )
                fig_bee.add_vline(
                    x=0,
                    line_dash="solid",
                    line_color="white",
                    opacity=0.3,
                )
                fig_bee.update_layout(
                    plot_bgcolor=PLOT_BG,
                    paper_bgcolor=PLOT_BG,
                    coloraxis_colorbar=dict(title="Value"),
                )
                st.plotly_chart(fig_bee, use_container_width=True)

        # Waterfall chart for individual station
        st.markdown("**Station Waterfall Chart**")
        st.caption(
            "Select a station to see how each variable contributed "
            "to its anomaly score."
        )

        anomaly_stations = df[
            (df["ml_anomaly_flag"] == True) &
            (df["shap_computed"] == True)
        ]["station_name"].dropna().unique().tolist() \
            if "shap_computed" in df.columns else []

        if anomaly_stations:
            selected_stn = st.selectbox(
                "Select anomalous station",
                options=sorted(anomaly_stations),
                key="shap_stn_sel"
            )
            stn_rows = df[
                (df["station_name"] == selected_stn) &
                (df["ml_anomaly_flag"] == True)
            ].sort_values("ml_anomaly_score")

            if not stn_rows.empty:
                top_row    = stn_rows.iloc[0]
                wf_labels  = []
                wf_values  = []

                for shap_col in shap_feat_cols:
                    feat  = shap_col.replace("shap_", "")
                    label = feat_labels.get(feat, feat)
                    unit  = feat_units.get(feat, "")
                    if shap_col in top_row.index:
                        sv  = float(top_row[shap_col])
                        av  = float(top_row[feat]) \
                            if feat in top_row.index else np.nan
                        avs = f"{av:.1f}{unit}" \
                            if not np.isnan(av) else "N/A"
                        wf_labels.append(f"{label}<br>({avs})")
                        wf_values.append(sv)

                if wf_values:
                    pairs = sorted(zip(wf_labels, wf_values),
                                   key=lambda x: x[1])
                    wl = [p[0] for p in pairs]
                    wv = [p[1] for p in pairs]
                    colors = [
                        "#ef4444" if v < 0 else "#22c55e"
                        for v in wv
                    ]
                    score = float(top_row["ml_anomaly_score"])
                    fig_wf = go.Figure(go.Bar(
                        x=wl, y=wv,
                        marker_color=colors,
                        text=[f"{v:+.4f}" for v in wv],
                        textposition="outside",
                    ))
                    fig_wf.add_hline(y=0, line_color="white", opacity=0.3)
                    fig_wf.update_layout(
                        title=f"SHAP Waterfall -- {selected_stn} "
                              f"(anomaly score: {score:.4f})",
                        height=400,
                        plot_bgcolor=PLOT_BG,
                        paper_bgcolor=PLOT_BG,
                        yaxis_title="SHAP Contribution",
                        showlegend=False,
                    )
                    st.plotly_chart(fig_wf, use_container_width=True)

                    expl = top_row.get("shap_explanation", "")
                    if expl:
                        st.info(f"Explanation: {expl}")

        # SHAP explanation table
        st.markdown("**Top 20 Anomalies with SHAP Explanations**")
        expl_cols = [c for c in [
            "station_name", "state",
            "temperature_c", "humidity_pct",
            "ml_anomaly_score", "ml_severity",
            "shap_top_driver", "shap_top_driver_val",
            "shap_explanation",
        ] if c in df.columns]

        expl_df = (
            df[df["shap_computed"] == True][expl_cols]
            .sort_values("ml_anomaly_score")
            .head(20)
            .rename(columns={
                "station_name":      "Station",
                "state":             "State",
                "temperature_c":     "Temp (C)",
                "humidity_pct":      "Humidity (%)",
                "ml_anomaly_score":  "Score",
                "ml_severity":       "Severity",
                "shap_top_driver":   "Top Driver",
                "shap_top_driver_val":"Driver Value",
                "shap_explanation":  "SHAP Explanation",
            })
        ) if "shap_computed" in df.columns else pd.DataFrame()

        if not expl_df.empty:
            st.dataframe(
                expl_df,
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            "Score: more negative = more anomalous. "
            "Red bars = variables pushing toward anomalous. "
            "Green bars = variables pulling toward normal."
        )
    elif ml_available:
        st.info(
            "SHAP Phase 3 will appear here after the next pipeline run. "
            "Run: python main.py"
        )
                    

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: BARRA2 COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("📈 BOM Live vs BARRA2 Historical Baseline")

    if not barra2_available:
        st.info(
            "BARRA2 historical baseline is not yet loaded. "
            "This tab will populate once BARRA2 THREDDS access is confirmed."
        )
        st.markdown("""
        **What this tab shows when BARRA2 loads:**
        - BOM current readings vs BARRA2 historical monthly mean
        - Side-by-side comparison for temperature, humidity, and wind
        - Which stations are running above or below their historical norms
        - Spatial distribution of anomalies across Australia
        """)
        st.divider()
        st.subheader("Current BOM Summary: All Stations")
        if not summary.empty:
            disp_cols = {
                "station_name":        "Station",
                "state":               "State",
                "mean_temperature_c":  "Avg Temp (°C)",
                "max_temperature_c":   "Max Temp (°C)",
                "min_temperature_c":   "Min Temp (°C)",
                "temp_range_c":        "Temp Range (°C)",
                "mean_humidity_pct":   "Avg Humidity (%)",
                "mean_wind_speed_kmh": "Avg Wind (km/h)",
                "max_gust_kmh":        "Max Gust (km/h)",
                "heat_stress_score":   "Heat Stress Score",
                "vulnerability_index": "Vulnerability Index",
                "record_count":        "Records",
            }
            avail = {
                k: v for k, v in disp_cols.items()
                if k in summary.columns
            }
            disp = summary[list(avail.keys())].copy()
            disp.rename(columns=avail, inplace=True)
            for col in disp.select_dtypes("float").columns:
                disp[col] = disp[col].round(1)
            st.dataframe(
                disp.sort_values(
                    "Avg Temp (°C)", ascending=False
                ) if "Avg Temp (°C)" in disp.columns else disp,
                width='stretch',
                hide_index=True,
            )
    else:
        # ── BARRA2 comparison available ───────────────────────────
        barra2_cols_present = [
            c for c in df.columns if c.startswith("barra2_")
        ]

        # Build comparison DataFrame
        agg_dict = {
            "temperature_c": "mean",
            "humidity_pct":  "mean",
            "wind_speed_kmh":"mean",
        }
        for col in barra2_cols_present:
            agg_dict[col] = "first"
        if "state" in df.columns:
            agg_dict["state"] = "first"

        comp_df = (
            df.groupby("station_name")
            .agg(agg_dict)
            .reset_index()
        )
        for col in comp_df.columns:
            if col != "station_name" and col != "state":
                comp_df[col] = pd.to_numeric(
                    comp_df[col], errors="coerce"
                ).round(1)

        # Which pairs to show
        pair_options = {}
        if "temperature_c" in comp_df and "barra2_temp_c" in comp_df:
            pair_options["Temperature (°C)"] = (
                "temperature_c", "barra2_temp_c"
            )
        if "humidity_pct" in comp_df and "barra2_humidity_pct" in comp_df:
            pair_options["Humidity (%)"] = (
                "humidity_pct", "barra2_humidity_pct"
            )
        if "wind_speed_kmh" in comp_df and "barra2_wind_kmh" in comp_df:
            pair_options["Wind Speed (km/h)"] = (
                "wind_speed_kmh", "barra2_wind_kmh"
            )

        selected_pair_label = st.selectbox(
            "Compare variable", list(pair_options.keys())
        )
        bom_col, barra2_col = pair_options[selected_pair_label]

        comp_clean = comp_df.dropna(
            subset=[bom_col, barra2_col]
        ).sort_values(bom_col, ascending=False).head(40)

        if not comp_clean.empty:
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                name="BOM Current",
                x=comp_clean["station_name"],
                y=comp_clean[bom_col],
                marker_color="#ef4444",
                opacity=0.85,
            ))
            fig_comp.add_trace(go.Bar(
                name="BARRA2 Historical Mean",
                x=comp_clean["station_name"],
                y=comp_clean[barra2_col],
                marker_color="#3b82f6",
                opacity=0.85,
            ))
            fig_comp.update_layout(
                barmode="group",
                title=f"BOM Current vs BARRA2 Historical"
                      f"{selected_pair_label} (Top 40 stations)",
                xaxis_tickangle=-45,
                height=480,
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=PLOT_BG,
                yaxis_title=selected_pair_label,
                legend=dict(
                    orientation="h",
                    yanchor="bottom", y=1.02,
                    xanchor="right",  x=1,
                )
            )
            st.plotly_chart(fig_comp, width='stretch')

            # Difference chart
            comp_clean["difference"] = (
                comp_clean[bom_col] - comp_clean[barra2_col]
            ).round(2)

            fig_diff = px.bar(
                comp_clean.sort_values("difference", ascending=False),
                x="station_name", y="difference",
                color="difference",
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                title=f"Difference: BOM Current minus BARRA2 Historical "
                      f": {selected_pair_label}",
                height=380,
                labels={
                    "station_name": "Station",
                    "difference":   f"Difference ({selected_pair_label})",
                }
            )
            fig_diff.add_hline(y=0, line_color="white", opacity=0.2)
            fig_diff.update_layout(
                xaxis_tickangle=-45,
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=PLOT_BG,
            )
            st.plotly_chart(fig_diff, width='stretch')

            # BARRA2 metadata
            if "barra2_baseline_year" in df.columns:
                yr = df["barra2_baseline_year"].dropna().iloc[0] \
                    if not df["barra2_baseline_year"].dropna().empty else "?"
                mo = df["barra2_baseline_month"].dropna().iloc[0] \
                    if "barra2_baseline_month" in df.columns else "?"
                freq = df["barra2_freq"].dropna().iloc[0] \
                    if "barra2_freq" in df.columns else "?"
                st.caption(
                    f"BARRA2 baseline: {yr}-{int(mo):02d}, "
                    f"frequency: {freq}, resolution: 12km (AUS-11 grid). "
                    f"Source: NCI THREDDS ob53 - CC-BY 4.0"
                )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: DATA TABLE
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("📋 Station Summary Table")

    if summary.empty:
        st.info("Run `python main.py` to generate station summaries.")
    else:
        display_map = {
            "station_name":        "Station",
            "state":               "State",
            "mean_temperature_c":  "Avg Temp (°C)",
            "max_temperature_c":   "Max Temp (°C)",
            "min_temperature_c":   "Min Temp (°C)",
            "temp_range_c":        "Temp Range (°C)",
            "mean_feels_like_c":   "Avg Feels Like (°C)",
            "mean_humidity_pct":   "Avg Humidity (%)",
            "min_humidity_pct":    "Min Humidity (%)",
            "mean_wind_speed_kmh": "Avg Wind (km/h)",
            "max_gust_kmh":        "Max Gust (km/h)",
            "mean_pressure_hpa":   "Avg Pressure (hPa)",
            "heat_stress_score":   "Heat Stress Score",
            "vulnerability_index": "Vulnerability Index",
            "barra2_temp_c":       "BARRA2 Hist Temp (°C)",
            "temp_anomaly_c":      "Temp Anomaly (°C)",
            "record_count":        "Records",
            "latitude":            "Lat",
            "longitude":           "Lon",
        }
        avail = {
            k: v for k, v in display_map.items()
            if k in summary.columns
        }
        disp = summary[list(avail.keys())].copy()
        disp.rename(columns=avail, inplace=True)

        for col in disp.select_dtypes("float").columns:
            disp[col] = disp[col].round(1)

        # Filters
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            if "State" in disp.columns:
                filter_state = st.multiselect(
                    "Filter by State",
                    options=sorted(
                        disp["State"].dropna().unique().tolist()
                    ),
                    default=[],
                    key="table_state_filter"
                )
                if filter_state:
                    disp = disp[disp["State"].isin(filter_state)]

        with fc2:
            sort_col = st.selectbox(
                "Sort by",
                options=[
                    c for c in disp.columns
                    if c not in ["Station", "State", "Lat", "Lon"]
                ],
                index=0,
                key="table_sort"
            )

        with fc3:
            sort_asc = st.radio(
                "Order", ["Descending", "Ascending"],
                horizontal=True,
                key="table_order"
            )

        if sort_col in disp.columns:
            disp = disp.sort_values(
                sort_col, ascending=(sort_asc == "Ascending")
            )

        st.dataframe(
            disp,
            width='stretch',
            hide_index=True,
            height=580,
        )

        c_dl1, c_dl2, _ = st.columns([1, 1, 4])
        with c_dl1:
            st.download_button(
                "📥 Download CSV",
                data=disp.to_csv(index=False),
                file_name="ausclimate_station_summary.csv",
                mime="text/csv",
                width='stretch',
            )
        with c_dl2:
            st.download_button(
                "📥 Download JSON",
                data=disp.to_json(orient="records", indent=2),
                file_name="ausclimate_station_summary.json",
                mime="application/json",
                width='stretch',
            )

    # Raw data preview
    with st.expander("View raw merged data (first 500 rows)"):
        if not df.empty:
            st.dataframe(
                df.head(500),
                width='stretch',
                hide_index=True,
            )
        else:
            st.info("No merged data available.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 -- ENERGY (AEMO)
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.subheader("Energy Demand vs Climate Conditions")
    st.caption(
        "Australian Energy Market Operator (AEMO) 5-minute dispatch data "
        "for NEM regions, overlaid with live BOM temperature readings. "
        "WA and NT operate separate electricity markets and are not shown."
    )

    aemo_available = (
        not df_aemo.empty
        and "total_demand_mw" in df_aemo.columns
        and df_aemo["total_demand_mw"].notna().any()
    )

    # Refresh button
    if st.button("Refresh AEMO Data", key="aemo_refresh"):
        with st.spinner("Fetching latest AEMO dispatch..."):
            try:
                from ingestion.aemo import fetch_aemo_dispatch, merge_aemo_with_bom
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
            "AEMO dispatch data is not yet loaded or is showing placeholder values. "
            "Click Refresh above or run `python main.py` to fetch fresh data."
        )
        st.markdown("""
        **What this tab shows when AEMO loads:**
        - Current electricity demand (MW) per NEM region
        - Regional spot price ($/MWh) per region
        - Temperature vs demand correlation chart
        - Rolling demand history over the last 7 days
        """)

        # Show BOM summary in the meantime
        st.subheader("Current BOM Station Summary")
        if not summary.empty:
            cols_map = {
                "station_name":       "Station",
                "state":              "State",
                "mean_temperature_c": "Avg Temp (C)",
                "max_temperature_c":  "Max Temp (C)",
                "mean_humidity_pct":  "Avg Humidity (%)",
                "mean_wind_speed_kmh":"Avg Wind (km/h)",
                "record_count":       "Records",
            }
            avail = {k: v for k, v in cols_map.items() if k in summary.columns}
            disp  = summary[list(avail.keys())].copy()
            disp.rename(columns=avail, inplace=True)
            for col in ["Avg Temp (C)", "Max Temp (C)",
                        "Avg Humidity (%)", "Avg Wind (km/h)"]:
                if col in disp.columns:
                    disp[col] = pd.to_numeric(
                        disp[col], errors="coerce"
                    ).round(1)
            st.dataframe(
                disp.sort_values("Avg Temp (C)", ascending=False)
                if "Avg Temp (C)" in disp.columns else disp,
                use_container_width=True,
                hide_index=True,
            )

    else:
        # Top metrics
        total_dem      = df_aemo["total_demand_mw"].sum()
        has_price      = "rrp_per_mwh" in df_aemo.columns and df_aemo["rrp_per_mwh"].notna().any()
        max_price      = df_aemo["rrp_per_mwh"].max() if has_price else 0
        max_price_rgn  = ""
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
        c1.metric("Total NEM Demand",  f"{total_dem:,.0f} MW")
        c2.metric("Highest Spot Price", f"${max_price:,.0f}/MWh", max_price_rgn)
        c3.metric("NEM Regions",       "5")
        c4.metric("Last Updated",      settle_str or "Just now")

        st.divider()

        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Current Demand by Region")
            dem_df = df_aemo.dropna(subset=["total_demand_mw"]).copy()
            dem_df = dem_df.sort_values("total_demand_mw", ascending=False)
            x_col  = "region_name" if "region_name" in dem_df.columns else "region"

            fig_dem = px.bar(
                dem_df, x=x_col, y="total_demand_mw",
                color="total_demand_mw",
                color_continuous_scale="YlOrRd",
                text_auto=".0f", height=360,
                labels={"total_demand_mw": "Demand (MW)", x_col: "Region"}
            )
            fig_dem.update_traces(textposition="outside")
            fig_dem.update_layout(
                coloraxis_showscale=False,
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=PLOT_BG,
                xaxis_title="",
            )
            st.plotly_chart(fig_dem, use_container_width=True)

        with col_r:
            st.subheader("Spot Price by Region")
            if has_price:
                pr_df  = df_aemo.dropna(subset=["rrp_per_mwh"]).copy()
                pr_df  = pr_df.sort_values("rrp_per_mwh", ascending=False)
                x_col2 = "region_name" if "region_name" in pr_df.columns else "region"

                fig_pr = px.bar(
                    pr_df, x=x_col2, y="rrp_per_mwh",
                    color="rrp_per_mwh",
                    color_continuous_scale="RdYlGn_r",
                    text_auto=".0f", height=360,
                    labels={"rrp_per_mwh": "Spot Price ($/MWh)", x_col2: "Region"}
                )
                fig_pr.update_traces(textposition="outside")
                fig_pr.update_layout(
                    coloraxis_showscale=False,
                    plot_bgcolor=PLOT_BG,
                    paper_bgcolor=PLOT_BG,
                    xaxis_title="",
                )
                st.plotly_chart(fig_pr, use_container_width=True)
            else:
                st.info("Spot price data not available in this fetch.")

        st.divider()

        # Temperature vs demand correlation
        st.subheader("Temperature vs Electricity Demand")
        st.caption(
            "Each dot is a NEM region. Shows how temperature drives "
            "electricity demand across Australia right now."
        )

        if (not df_aemo_bom.empty
                and "avg_temp_c" in df_aemo_bom.columns
                and "total_demand_mw" in df_aemo_bom.columns):

            corr_df = df_aemo_bom.dropna(
                subset=["avg_temp_c", "total_demand_mw"]
            )

            if not corr_df.empty:
                x_label = "region_name" \
                    if "region_name" in corr_df.columns else "region"

                # Build hover_data only from columns that actually exist
                hover = {}
                if "region" in corr_df.columns:
                    hover["region"] = True
                if "avg_temp_c" in corr_df.columns:
                    hover["avg_temp_c"] = ":.1f"
                if "total_demand_mw" in corr_df.columns:
                    hover["total_demand_mw"] = ":,.0f"
                if "rrp_per_mwh" in corr_df.columns:
                    hover["rrp_per_mwh"] = ":,.0f"
                if "station_count" in corr_df.columns:
                    hover["station_count"] = True
                if "states" in corr_df.columns:
                    hover["states"] = True

                fig_corr = px.scatter(
                    corr_df,
                    x="avg_temp_c",
                    y="total_demand_mw",
                    text=x_label,
                    size=[40] * len(corr_df),
                    color="total_demand_mw",
                    color_continuous_scale="YlOrRd",
                    hover_data=hover,
                    height=420,
                    labels={
                        "avg_temp_c":      "Average Temperature (C)",
                        "total_demand_mw": "Electricity Demand (MW)",
                    }
                )
                fig_corr.update_traces(
                    textposition="top center",
                    marker=dict(
                        opacity=0.85,
                        line=dict(width=1, color="white")
                    )
                )
                fig_corr.update_layout(
                    coloraxis_showscale=False,
                    plot_bgcolor=PLOT_BG,
                    paper_bgcolor=PLOT_BG,
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Not enough data to show correlation chart.")
        else:
            st.info(
                "Temperature-demand correlation requires both BOM and "
                "AEMO data to be loaded. Run `python main.py` first."
            )

        # Demand history
        if (not df_aemo_history.empty
                and "settlement_date" in df_aemo_history.columns):
            st.divider()
            st.subheader("Demand History (Rolling 7 Days)")

            hist = df_aemo_history.copy()
            hist["settlement_date"] = pd.to_datetime(
                hist["settlement_date"], errors="coerce"
            )
            hist["total_demand_mw"] = pd.to_numeric(
                hist["total_demand_mw"], errors="coerce"
            )
            hist = hist.dropna(subset=["settlement_date", "total_demand_mw"])

            if not hist.empty:
                clr_col = "region_name" \
                    if "region_name" in hist.columns else "region"

                region_opts = hist[clr_col].dropna().unique().tolist()
                sel_regions = st.multiselect(
                    "Regions",
                    options=region_opts,
                    default=region_opts[:3],
                    key="aemo_hist_sel"
                )

                hist_f = hist[hist[clr_col].isin(sel_regions)] \
                    if sel_regions else hist

                fig_hist = px.line(
                    hist_f.sort_values("settlement_date"),
                    x="settlement_date",
                    y="total_demand_mw",
                    color=clr_col,
                    height=360,
                    title="NEM Regional Demand Over Time",
                    labels={
                        "settlement_date": "Time",
                        "total_demand_mw": "Demand (MW)",
                        clr_col:           "Region",
                    }
                )
                fig_hist.update_layout(
                    plot_bgcolor=PLOT_BG,
                    paper_bgcolor=PLOT_BG,
                )
                st.plotly_chart(fig_hist, use_container_width=True)

        # Region detail table
        st.divider()
        st.subheader("Region Detail")

        display_src = df_aemo_bom if not df_aemo_bom.empty else df_aemo
        col_map = {
            "region":          "Region ID",
            "region_name":     "Region",
            "states":          "States",
            "total_demand_mw": "Demand (MW)",
            "rrp_per_mwh":     "Spot Price ($/MWh)",
            "avg_temp_c":      "Avg Temp (C)",
            "max_temp_c":      "Max Temp (C)",
            "station_count":   "BOM Stations",
        }
        avail = {k: v for k, v in col_map.items() if k in display_src.columns}
        if avail:
            disp2 = display_src[list(avail.keys())].copy()
            disp2.rename(columns=avail, inplace=True)
            for col in ["Demand (MW)", "Spot Price ($/MWh)",
                        "Avg Temp (C)", "Max Temp (C)"]:
                if col in disp2.columns:
                    disp2[col] = pd.to_numeric(
                        disp2[col], errors="coerce"
                    ).round(1)
            st.dataframe(disp2, use_container_width=True, hide_index=True)

        st.caption(
            "Source: AEMO NEMWeb public dispatch data | "
            "Updated every 5 minutes | "
            "CC-BY 4.0 | "
            "nemweb.com.au/Reports/Current/DispatchIS_Reports/"
        )
        
# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
col_f1, col_f2 = st.columns([3, 1])
with col_f1:
    barra2_note = (
        "BARRA2 NCI THREDDS (BOM) | "
        if barra2_available else ""
    )
    st.caption(
        f"Data: Bureau of Meteorology Live JSON API | "
        f"{barra2_note}"
        f"AusClimate Capstone Project: "
        f"University of Technology Sydney 2026"
    )
with col_f2:
    from datetime import datetime
    st.caption(f"Last loaded: {datetime.now().strftime('%H:%M:%S')}")