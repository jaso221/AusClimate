import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add project root to path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DATA_PATH

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AusClimate",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def load_data():
    if not os.path.exists(PROCESSED_DATA_PATH):
        return pd.DataFrame()
    return pd.read_parquet(PROCESSED_DATA_PATH)

df = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌡️ AusClimate")
st.caption(
    "Real-Time Australian Climate Analytics Platform  |  "
    "Bureau of Meteorology Live Observation Data"
)
st.divider()

# ── Empty state ───────────────────────────────────────────────────────────────
if df.empty:
    st.warning(
        "No data loaded yet. Run `python main.py` first to ingest BOM data, "
        "then refresh this page."
    )
    st.stop()

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.header("Controls")

# Refresh button
if st.sidebar.button("🔄 Refresh BOM Data", use_container_width=True):
    with st.spinner("Fetching latest BOM data..."):
        try:
            from ingestion.bom import fetch_all_stations
            from pipeline.cleaner import clean_bom_data
            from pipeline.feature_eng import compute_station_summary
            from pipeline.storage import save_parquet

            df_raw  = fetch_all_stations()
            df_clean = clean_bom_data(df_raw)
            save_parquet(df_clean, PROCESSED_DATA_PATH)
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Refresh failed: {e}")

st.sidebar.divider()

# Station selector
all_stations = sorted(df["station_name"].unique().tolist())
selected_stations = st.sidebar.multiselect(
    "Select Stations",
    options=all_stations,
    default=all_stations[:6]
)

# Variable selector
feature_options = {}
if "temperature_c"  in df.columns: feature_options["Temperature (°C)"]      = "temperature_c"
if "feels_like_c"   in df.columns: feature_options["Feels Like (°C)"]        = "feels_like_c"
if "humidity_pct"   in df.columns: feature_options["Humidity (%)"]           = "humidity_pct"
if "wind_speed_kmh" in df.columns: feature_options["Wind Speed (km/h)"]      = "wind_speed_kmh"
if "gust_kmh"       in df.columns: feature_options["Max Gust (km/h)"]        = "gust_kmh"
if "pressure_hpa"   in df.columns: feature_options["Pressure (hPa)"]         = "pressure_hpa"

selected_label = st.sidebar.selectbox(
    "Select Variable",
    options=list(feature_options.keys())
)
selected_col = feature_options[selected_label]

# Chart type
chart_type = st.sidebar.radio(
    "Chart Type",
    options=["Bar", "Horizontal Bar", "Line", "Scatter"],
    horizontal=True
)

# ── Filter data ───────────────────────────────────────────────────────────────
if not selected_stations:
    st.info("Select at least one station from the sidebar.")
    st.stop()

df_filtered = df[df["station_name"].isin(selected_stations)].copy()
df_filtered[selected_col] = pd.to_numeric(df_filtered[selected_col], errors="coerce")

summary = (
    df_filtered.groupby("station_name")[selected_col]
    .mean()
    .dropna()
    .reset_index()
    .sort_values(selected_col, ascending=False)
)
summary.columns = ["Station", selected_label]

# ── Stats row ─────────────────────────────────────────────────────────────────
if not summary.empty:
    c1, c2, c3, c4 = st.columns(4)
    avg_val     = summary[selected_label].mean()
    max_val     = summary[selected_label].max()
    min_val     = summary[selected_label].min()
    max_station = summary.loc[summary[selected_label].idxmax(), "Station"]
    min_station = summary.loc[summary[selected_label].idxmin(), "Station"]

    c1.metric("Average",             f"{avg_val:.1f}")
    c2.metric("Highest",             f"{max_val:.1f}",  max_station)
    c3.metric("Lowest",              f"{min_val:.1f}",  min_station)
    c4.metric("Range",               f"{max_val - min_val:.1f}")

st.divider()

# ── Chart ─────────────────────────────────────────────────────────────────────
COLOUR_SCALE = "RdYlBu_r"

if chart_type == "Bar":
    fig = px.bar(
        summary,
        x="Station", y=selected_label,
        color=selected_label,
        color_continuous_scale=COLOUR_SCALE,
        title=f"Average {selected_label} by Station",
        text_auto=".1f"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-30)

elif chart_type == "Horizontal Bar":
    fig = px.bar(
        summary.sort_values(selected_label),
        x=selected_label, y="Station",
        orientation="h",
        color=selected_label,
        color_continuous_scale=COLOUR_SCALE,
        title=f"Average {selected_label} by Station",
        text_auto=".1f"
    )
    fig.update_layout(coloraxis_showscale=False)

elif chart_type == "Line":
    if "local_date_time_full" in df_filtered.columns:
        df_line = df_filtered.sort_values("local_date_time_full")
        fig = px.line(
            df_line,
            x="local_date_time_full",
            y=selected_col,
            color="station_name",
            title=f"{selected_label} Over Time",
            labels={
                selected_col: selected_label,
                "local_date_time_full": "Time",
                "station_name": "Station"
            }
        )
    else:
        fig = px.line(
            summary,
            x="Station", y=selected_label,
            markers=True,
            title=f"{selected_label} by Station"
        )

elif chart_type == "Scatter":
    scatter_df = df_filtered[["station_name", selected_col]].dropna()
    scatter_df = scatter_df.reset_index()
    fig = px.scatter(
        scatter_df,
        x="index", y=selected_col,
        color="station_name",
        title=f"{selected_label} — All Observations",
        labels={
            selected_col: selected_label,
            "index": "Observation Index",
            "station_name": "Station"
        },
        opacity=0.6
    )

fig.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_family="Arial",
    height=420,
    margin=dict(t=50, b=40, l=40, r=20)
)
st.plotly_chart(fig, use_container_width=True)

# ── Dashboard tiles ───────────────────────────────────────────────────────────
st.divider()
st.subheader("My Dashboard")

if "tiles" not in st.session_state:
    st.session_state.tiles = []

col_add, col_clear = st.columns([1, 5])
with col_add:
    if st.button("➕ Add Chart to Dashboard", use_container_width=True):
        st.session_state.tiles.append({
            "stations": selected_stations.copy(),
            "col": selected_col,
            "label": selected_label,
            "chart": chart_type
        })
with col_clear:
    if st.button("Clear Dashboard", use_container_width=True):
        st.session_state.tiles = []

if st.session_state.tiles:
    cols = st.columns(2)
    for i, tile in enumerate(st.session_state.tiles):
        with cols[i % 2]:
            with st.container(border=True):
                tile_df = df[df["station_name"].isin(tile["stations"])].copy()
                tile_df[tile["col"]] = pd.to_numeric(tile_df[tile["col"]], errors="coerce")
                tile_summary = (
                    tile_df.groupby("station_name")[tile["col"]]
                    .mean().dropna().reset_index()
                )
                tile_summary.columns = ["Station", tile["label"]]

                if tile["chart"] == "Horizontal Bar":
                    tile_fig = px.bar(
                        tile_summary.sort_values(tile["label"]),
                        x=tile["label"], y="Station",
                        orientation="h",
                        color=tile["label"],
                        color_continuous_scale=COLOUR_SCALE,
                        title=f"{tile['label']}",
                        height=280
                    )
                elif tile["chart"] == "Line" and "local_date_time_full" in tile_df.columns:
                    tile_fig = px.line(
                        tile_df.sort_values("local_date_time_full"),
                        x="local_date_time_full", y=tile["col"],
                        color="station_name",
                        title=f"{tile['label']}",
                        height=280
                    )
                else:
                    tile_fig = px.bar(
                        tile_summary.sort_values(tile["label"], ascending=False),
                        x="Station", y=tile["label"],
                        color=tile["label"],
                        color_continuous_scale=COLOUR_SCALE,
                        title=f"{tile['label']}",
                        height=280
                    )

                tile_fig.update_layout(
                    coloraxis_showscale=False,
                    margin=dict(t=40, b=20, l=20, r=20),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(tile_fig, use_container_width=True)

                if st.button(f"Remove", key=f"remove_{i}"):
                    st.session_state.tiles.pop(i)
                    st.rerun()

# ── Station summary table ─────────────────────────────────────────────────────
st.divider()
st.subheader("Station Summary Table")

display_cols = {
    "station_name":    "Station",
    "temperature_c":   "Temp (°C)",
    "feels_like_c":    "Feels Like (°C)",
    "humidity_pct":    "Humidity (%)",
    "wind_speed_kmh":  "Wind (km/h)",
    "gust_kmh":        "Gust (km/h)",
    "pressure_hpa":    "Pressure (hPa)",
}

available = [c for c in display_cols if c in df_filtered.columns]
table_df  = df_filtered[available].copy()

for col in available[1:]:
    table_df[col] = pd.to_numeric(table_df[col], errors="coerce")

table_summary = (
    table_df.groupby("station_name")
    .mean(numeric_only=True)
    .round(1)
    .reset_index()
)
table_summary.columns = [display_cols.get(c, c) for c in table_summary.columns]
st.dataframe(table_summary, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Data source: Bureau of Meteorology Live JSON API  |  "
    "Updated every 30 minutes  |  "
    "AusClimate Capstone Project — University of Technology Sydney 2026"
)