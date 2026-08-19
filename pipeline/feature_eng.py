"""
Feature engineering module.
Computes station-level aggregate features including anomaly context
when BARRA2 data is available.
"""

import pandas as pd
import numpy as np


def compute_station_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-station aggregate features.
    Works on BOM-only data or BOM+BARRA2 merged data.
    """
    if df.empty:
        return df

    # Primary BOM variables to aggregate
    bom_agg_cols = {
        "temperature_c":  ["mean", "max", "min"],
        "feels_like_c":   ["mean", "max"],
        "humidity_pct":   ["mean", "min"],
        "wind_speed_kmh": ["mean", "max"],
        "gust_kmh":       ["max"],
        "pressure_hpa":   ["mean"],
    }

    agg_dict = {}
    for col, fns in bom_agg_cols.items():
        if col in df.columns:
            for fn in fns:
                label = f"{fn}_{col}"
                agg_dict[label] = (col, fn)

    agg_dict["record_count"] = ("temperature_c", "count")

    # Aggregate BARRA2 columns if present
    barra2_cols = [c for c in df.columns if c.startswith("barra2_")]
    for col in barra2_cols:
        agg_dict[col] = (col, "first")  # BARRA2 is already per-station

    # ERA5 columns if present
    era5_cols = [c for c in df.columns if c.startswith("era5_")]
    for col in era5_cols:
        agg_dict[col] = (col, "first")

    # Anomaly columns if present
    anomaly_cols = [c for c in df.columns if c.endswith("_anomaly")]
    for col in anomaly_cols:
        agg_dict[col] = (col, "mean")

    # State column
    if "state" in df.columns:
        agg_dict["state"] = ("state", "first")

    # Lat/lon
    if "latitude" in df.columns:
        agg_dict["latitude"]  = ("latitude",  "first")
        agg_dict["longitude"] = ("longitude", "first")

    summary = df.groupby("station_name").agg(**agg_dict).reset_index()

    # Temperature range
    if "max_temperature_c" in summary and "min_temperature_c" in summary:
        summary["temp_range_c"] = (
            summary["max_temperature_c"] - summary["min_temperature_c"]
        ).round(1)

    # Heat stress score (0-100 scale)
    # Combines temperature and humidity — higher = more stressful
    # Based on Steadman apparent temperature relationship
    if ("mean_temperature_c" in summary and
            "mean_humidity_pct" in summary):
        t   = summary["mean_temperature_c"]
        rh  = summary["mean_humidity_pct"]
        summary["heat_stress_score"] = (
            (t * 0.55) + (rh / 100 * t * 0.35) + (t * 0.10)
        ).clip(0, 100).round(1)

    # Vulnerability index: combines temp, humidity, wind
    # Higher = more vulnerable conditions
    if ("mean_temperature_c" in summary and
            "mean_humidity_pct" in summary and
            "mean_wind_speed_kmh" in summary):
        t   = summary["mean_temperature_c"].fillna(0)
        rh  = summary["mean_humidity_pct"].fillna(50)
        w   = summary["mean_wind_speed_kmh"].fillna(0)
        # Normalise each to 0-1 range (approximate Australian ranges)
        t_n = (t  - (-5)) / (50  - (-5))
        r_n = rh  / 100
        w_n = w   / 150
        summary["vulnerability_index"] = (
            (t_n * 0.5 + r_n * 0.3 + w_n * 0.2) * 100
        ).clip(0, 100).round(1)

    summary = summary.round(2)
    print(f"Feature engineering: {len(summary)} station summaries.")
    return summary