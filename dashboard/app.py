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

from config import (
    PROCESSED_DATA_PATH,
    MERGED_DATA_PATH,
    STATION_SUMMARY_PATH,
    BARRA2_CACHE_PATH,
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


df, summary = load_all_data()


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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌡️ AusClimate")
    st.caption("Real-Time Australian Climate Analytics")
    st.divider()

    # Data source status
    st.markdown('<p class="section-label">Data Sources</p>',
                unsafe_allow_html=True)

    n_stations = df["station_name"].nunique() if "station_name" in df.columns else 0
    bom_status = (
        f'<span class="status-ok">● Live</span> '
        f'— {n_stations} stations'
        if n_stations > 0
        else '<span class="status-err">● No data</span>'
    )
    barra2_status = (
        '<span class="status-ok">● Loaded</span>'
        if barra2_available
        else '<span class="status-warn">● Pending NCI access</span>'
    )

    st.markdown(
        f"**BOM** {bom_status}<br>"
        f"**BARRA2** {barra2_status}<br>"
        f"**ERA5** <span class=\"status-warn\">● Needs CDS key</span>",
        unsafe_allow_html=True
    )
    st.divider()

    # Refresh
    if st.button("🔄 Refresh Data", width='stretch'):
        with st.spinner("Running BOM ingestion..."):
            try:
                from ingestion.bom import fetch_all_stations, get_station_coords
                from pipeline.cleaner import clean_bom_data
                from pipeline.feature_eng import compute_station_summary
                from pipeline.storage import save_parquet
                from pipeline.spatial_join import merge_bom_barra2, merge_era5

                df_raw   = fetch_all_stations()
                df_clean = clean_bom_data(df_raw)
                save_parquet(df_clean, PROCESSED_DATA_PATH)

                # Reload BARRA2 from cache if available
                barra2_df = pd.DataFrame()
                if os.path.exists(BARRA2_CACHE_PATH):
                    try:
                        barra2_df = pd.read_parquet(BARRA2_CACHE_PATH)
                    except Exception:
                        pass

                merged = merge_bom_barra2(df_clean, barra2_df)
                save_parquet(merged, MERGED_DATA_PATH)
                feat = compute_station_summary(merged)
                save_parquet(feat, STATION_SUMMARY_PATH)

                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Refresh failed: {e}")

    st.divider()

    # ── Filters ───────────────────────────────────────────────────
    st.markdown('<p class="section-label">Filters</p>',
                unsafe_allow_html=True)

    if df.empty:
        st.warning("No data. Run `python main.py` first.")
        st.stop()

    all_states = sorted(df["state"].dropna().unique().tolist()) \
        if "state" in df.columns else []

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

    all_stations_in_state = sorted(
        df_state["station_name"].dropna().unique().tolist()
    ) if "station_name" in df_state.columns else []

    selected_stations = st.multiselect(
        "Stations",
        options=all_stations_in_state,
        default=all_stations_in_state[:12]
        if len(all_stations_in_state) > 12
        else all_stations_in_state,
    )

    # Variable picker
    FEATURE_OPTIONS = {}
    for col, label in {
        "temperature_c":  "Temperature (°C)",
        "feels_like_c":   "Feels Like (°C)",
        "humidity_pct":   "Humidity (%)",
        "wind_speed_kmh": "Wind Speed (km/h)",
        "gust_kmh":       "Max Gust (km/h)",
        "pressure_hpa":   "Pressure (hPa)",
        "dewpoint_c":     "Dewpoint (°C)",
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
        ["Bar", "Horizontal Bar", "Line", "Scatter", "Box", "Violin",
         "Radar", "Heatmap", "Bubble"],
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


# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🗺️ Station Map",
    "⚠️ Anomalies",
    "📈 BARRA2 Comparison",
    "📋 Data Table",
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
# TAB 3: ANOMALIES
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("⚠️ Climate Anomaly Detection")
    
        # ── ML Results (Isolation Forest) ────────────────────────────
    ml_available = (
        "ml_anomaly_flag" in df.columns and
        df["ml_anomaly_flag"].notna().any()
    )

    if ml_available:
        st.success(
            "Isolation Forest ML model active: "
            "anomalies detected across 5 climate variables."
        )

        ml_df = df[df["ml_anomaly_flag"] == True].copy()

        # Summary metrics
        n_total    = len(df)
        n_anomaly  = int(df["ml_anomaly_flag"].sum())
        n_severe   = len(df[df["ml_severity"] == "Severe"])
        n_moderate = len(df[df["ml_severity"] == "Moderate"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Observations Scored",  f"{n_total:,}")
        c2.metric("Anomalies Flagged",    f"{n_anomaly:,}")
        c3.metric("🔴 Severe",            n_severe)
        c4.metric("🟠 Moderate",          n_moderate)

        st.divider()

        # Severity distribution
        sev_counts = (
            df["ml_severity"].value_counts().reset_index()
        )
        sev_counts.columns = ["Severity", "Count"]
        sev_order   = ["Severe", "Moderate", "Mild", "Normal"]
        sev_colours = {
            "Severe":   "#ef4444",
            "Moderate": "#f97316",
            "Mild":     "#eab308",
            "Normal":   "#22c55e",
        }
        sev_counts = sev_counts[
            sev_counts["Severity"].isin(sev_order)
        ]

        col_sev, col_driver = st.columns(2)

        with col_sev:
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
            st.plotly_chart(fig_sev, width='stretch')

        with col_driver:
            driver_counts = (
                ml_df["ml_primary_driver"]
                .value_counts().reset_index()
            )
            driver_counts.columns = ["Driver", "Count"]
            if not driver_counts.empty:
                fig_drv = px.pie(
                    driver_counts,
                    names="Driver", values="Count",
                    title="Primary Anomaly Driver",
                    height=320,
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                st.plotly_chart(fig_drv, width='stretch')

        # Station anomaly summary
        st.subheader("Station Risk Summary")
        from anomaly.detector import compute_station_anomaly_summary
        station_risk = compute_station_anomaly_summary(df)

        if not station_risk.empty:
            risk_colours = {
                "High": "#ef4444", "Medium": "#f97316",
                "Low":  "#eab308", "Normal": "#22c55e"
            }

            fig_risk = px.scatter_mapbox(
                station_risk.dropna(
                    subset=["latitude", "longitude"]
                ),
                lat="latitude", lon="longitude",
                color="station_risk",
                color_discrete_map=risk_colours,
                size="anomaly_count",
                size_max=18,
                hover_name="station_name",
                hover_data={
                    "state":           True,
                    "anomaly_count":   True,
                    "anomaly_rate_pct":True,
                    "primary_driver":  True,
                    "station_risk":    True,
                    "latitude":        False,
                    "longitude":       False,
                },
                mapbox_style="carto-darkmatter",
                zoom=3.5,
                center={"lat": -26, "lon": 134},
                height=520,
                title="Station Risk Map: "
                      "dot size = anomaly count, colour = risk level",
            )
            st.plotly_chart(fig_risk, width='stretch')

            # Risk table
            display_risk = station_risk.rename(columns={
                "station_name":      "Station",
                "state":             "State",
                "total_observations":"Total Obs",
                "anomaly_count":     "Anomalies",
                "anomaly_rate_pct":  "Anomaly Rate (%)",
                "severe_count":      "Severe",
                "moderate_count":    "Moderate",
                "avg_anomaly_score": "Avg Score",
                "primary_driver":    "Primary Driver",
                "station_risk":      "Risk Level",
            })
            keep_cols = [
                "Station", "State", "Total Obs", "Anomalies",
                "Anomaly Rate (%)", "Severe", "Moderate",
                "Primary Driver", "Risk Level"
            ]
            keep_cols = [c for c in keep_cols if c in display_risk.columns]
            st.dataframe(
                display_risk[keep_cols].sort_values(
                    "Anomalies", ascending=False
                ),
                width='stretch',
                hide_index=True,
            )

        # Anomalous observations detail
        st.subheader("Most Anomalous Observations")
        detail_cols = [
            "station_name", "state", "temperature_c",
            "feels_like_c", "humidity_pct", "wind_speed_kmh",
            "ml_anomaly_score", "ml_severity", "ml_primary_driver"
        ]
        detail_cols = [c for c in detail_cols if c in ml_df.columns]

        top_anomalies = (
            ml_df[detail_cols]
            .sort_values("ml_anomaly_score")
            .head(20)
            .rename(columns={
                "station_name":      "Station",
                "state":             "State",
                "temperature_c":     "Temp (°C)",
                "feels_like_c":      "Feels Like (°C)",
                "humidity_pct":      "Humidity (%)",
                "wind_speed_kmh":    "Wind (km/h)",
                "ml_anomaly_score":  "Anomaly Score",
                "ml_severity":       "Severity",
                "ml_primary_driver": "Primary Driver",
            })
        )
        st.dataframe(
            top_anomalies,
            width='stretch',
            hide_index=True,
        )

        st.caption(
            "Anomaly Score: more negative = more anomalous. "
            "Primary Driver: the climate variable contributing "
            "most to the anomaly flag (full SHAP attribution in Phase 3)."
        )
        st.divider()

    if anomaly_available:
        # ── BARRA2-based anomalies ────────────────────────────────
        st.success(
            "BARRA2 historical baseline loaded. Anomalies show "
            "deviation from historical climate norms."
        )

        anom_df = (
            df.groupby("station_name")
            .agg({
                "temp_anomaly_c":       "mean",
                "humidity_anomaly_pct": "mean",
                "wind_anomaly_kmh":     "mean",
                "pressure_anomaly_hpa": "mean",
                "state":                "first",
            })
            .reset_index()
            .dropna(subset=["temp_anomaly_c"])
        )
        anom_df["temp_anomaly_c"] = anom_df["temp_anomaly_c"].round(2)

        anom_df["status"] = anom_df["temp_anomaly_c"].apply(
            lambda v:
            "🔴 Much warmer" if v >  3 else
            "🟠 Warmer"      if v >  1 else
            "🔵 Cooler"      if v < -1 else
            "🟢 Normal"
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Much warmer (>3°C)",
                  len(anom_df[anom_df["temp_anomaly_c"] > 3]))
        c2.metric("Warmer than normal",
                  len(anom_df[anom_df["temp_anomaly_c"] > 1]))
        c3.metric("Near normal",
                  len(anom_df[anom_df["temp_anomaly_c"].abs() <= 1]))
        c4.metric("Cooler than normal",
                  len(anom_df[anom_df["temp_anomaly_c"] < -1]))

        st.divider()

        # Temperature anomaly chart
        top_anom = anom_df.reindex(
            anom_df["temp_anomaly_c"].abs().sort_values(
                ascending=False
            ).index
        ).head(40)

        fig_anom = px.bar(
            top_anom,
            x="station_name",
            y="temp_anomaly_c",
            color="temp_anomaly_c",
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            title="Temperature Anomaly vs BARRA2 Historical Baseline (°C) "
                  "- Top 40 Stations by Magnitude",
            height=440,
            labels={
                "temp_anomaly_c":  "Anomaly (°C)",
                "station_name":    "Station",
            }
        )
        fig_anom.add_hline(
            y=0, line_dash="solid",
            line_color="white", opacity=0.2
        )
        fig_anom.add_hline(
            y=3, line_dash="dash",
            line_color="red", opacity=0.4,
            annotation_text="+3°C threshold"
        )
        fig_anom.add_hline(
            y=-3, line_dash="dash",
            line_color="blue", opacity=0.4,
            annotation_text="-3°C threshold"
        )
        fig_anom.update_layout(
            xaxis_tickangle=-45,
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=PLOT_BG,
            coloraxis_colorbar=dict(title="Anomaly (°C)"),
        )
        st.plotly_chart(fig_anom, width='stretch')

        # Multi-variable anomaly
        st.subheader("Multi-Variable Anomaly Overview")
        anomaly_var_opts = {}
        for col, lbl in {
            "temp_anomaly_c":       "Temperature Anomaly (°C)",
            "humidity_anomaly_pct": "Humidity Anomaly (%)",
            "wind_anomaly_kmh":     "Wind Anomaly (km/h)",
            "pressure_anomaly_hpa": "Pressure Anomaly (hPa)",
        }.items():
            if col in df.columns and df[col].notna().any():
                anomaly_var_opts[lbl] = col

        if len(anomaly_var_opts) > 1:
            selected_anom_var = st.selectbox(
                "Anomaly variable", list(anomaly_var_opts.keys())
            )
            anom_col = anomaly_var_opts[selected_anom_var]
            av_df = (
                df.groupby("station_name")[anom_col]
                .mean().dropna().reset_index()
                .sort_values(anom_col, ascending=False)
            )
            fig_av = px.bar(
                av_df.head(40),
                x="station_name", y=anom_col,
                color=anom_col,
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                height=380,
                title=f"{selected_anom_var} - Top 40 Stations",
                labels={"station_name": "Station", anom_col: selected_anom_var}
            )
            fig_av.add_hline(y=0, line_color="white", opacity=0.2)
            fig_av.update_layout(
                xaxis_tickangle=-45,
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=PLOT_BG,
                coloraxis_colorbar=dict(title=selected_anom_var),
            )
            st.plotly_chart(fig_av, width='stretch')

        # Anomaly detail table
        st.subheader("Anomaly Detail Table")
        display_anom = anom_df.rename(columns={
            "station_name":       "Station",
            "state":              "State",
            "temp_anomaly_c":     "Temp Anomaly (°C)",
            "humidity_anomaly_pct": "Humidity Anomaly (%)",
            "wind_anomaly_kmh":   "Wind Anomaly (km/h)",
            "pressure_anomaly_hpa":"Pressure Anomaly (hPa)",
            "status":             "Status",
        })
        st.dataframe(
            display_anom.sort_values(
                "Temp Anomaly (°C)", ascending=False
            ),
            width='stretch',
            hide_index=True,
        )

    else:
        # ── BARRA2 not available - statistical outliers ───────────
        st.info(
            "BARRA2 historical baseline is not yet loaded. "
            "Showing statistical outliers from live BOM data. "
            "Once BARRA2 connects, BARRA2-based anomaly columns "
            "will populate automatically."
        )

        if "temperature_c" not in df.columns:
            st.warning("No temperature data available.")
        else:
            temp_col = pd.to_numeric(df["temperature_c"], errors="coerce")
            t_mean   = temp_col.mean()
            t_std    = temp_col.std()

            outlier_df = (
                df.groupby("station_name")
                .agg({
                    "temperature_c": "mean",
                    "humidity_pct":  "mean",
                    "wind_speed_kmh":"mean",
                    "state":         "first",
                })
                .reset_index()
            )
            for col in ["temperature_c", "humidity_pct", "wind_speed_kmh"]:
                outlier_df[col] = pd.to_numeric(
                    outlier_df[col], errors="coerce"
                ).round(1)

            outlier_df["z_score"] = (
                (outlier_df["temperature_c"] - t_mean) / t_std
            ).round(2)
            outlier_df["flag"] = outlier_df["z_score"].apply(
                lambda z:
                "🔴 Hot outlier"  if z >  2 else
                "🔵 Cold outlier" if z < -2 else
                "🟢 Normal"
            )

            hot   = outlier_df[outlier_df["z_score"] >  2]
            cold  = outlier_df[outlier_df["z_score"] < -2]
            norm  = outlier_df[outlier_df["z_score"].abs() <= 2]

            c1, c2, c3 = st.columns(3)
            c1.metric("🔴 Hot outliers",  len(hot))
            c2.metric("🟢 Normal range",  len(norm))
            c3.metric("🔵 Cold outliers", len(cold))

            st.divider()

            # Z-score chart
            plot_df = outlier_df.sort_values(
                "z_score", ascending=False
            ).head(50)

            fig_z = px.bar(
                plot_df,
                x="station_name",
                y="z_score",
                color="z_score",
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                title="Temperature Z-Scores - Top 50 Stations "
                      "(|z| > 2 = statistical outlier)",
                height=440,
                text_auto=".1f",
                labels={
                    "z_score":      "Z-Score",
                    "station_name": "Station",
                }
            )
            fig_z.add_hline(
                y=2,  line_dash="dash",
                line_color="red",  opacity=0.5,
                annotation_text="+2σ"
            )
            fig_z.add_hline(
                y=-2, line_dash="dash",
                line_color="blue", opacity=0.5,
                annotation_text="-2σ"
            )
            fig_z.add_hline(
                y=0, line_color="white", opacity=0.15
            )
            fig_z.update_layout(
                xaxis_tickangle=-45,
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=PLOT_BG,
                coloraxis_colorbar=dict(title="Z-Score"),
            )
            st.plotly_chart(fig_z, width='stretch')

            # Top outlier tables
            st.divider()
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("**🔴 Hottest outlier stations**")
                if not hot.empty:
                    st.dataframe(
                        hot[["station_name", "state",
                             "temperature_c", "z_score"]]
                        .sort_values("z_score", ascending=False)
                        .rename(columns={
                            "station_name":  "Station",
                            "state":         "State",
                            "temperature_c": "Avg Temp (°C)",
                            "z_score":       "Z-Score",
                        }),
                        width='stretch',
                        hide_index=True,
                    )
                else:
                    st.caption("No hot outliers in current selection.")

            with col_right:
                st.markdown("**🔵 Coldest outlier stations**")
                if not cold.empty:
                    st.dataframe(
                        cold[["station_name", "state",
                              "temperature_c", "z_score"]]
                        .sort_values("z_score")
                        .rename(columns={
                            "station_name":  "Station",
                            "state":         "State",
                            "temperature_c": "Avg Temp (°C)",
                            "z_score":       "Z-Score",
                        }),
                        width='stretch',
                        hide_index=True,
                    )
                else:
                    st.caption("No cold outliers in current selection.")


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
                      f"— {selected_pair_label}",
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