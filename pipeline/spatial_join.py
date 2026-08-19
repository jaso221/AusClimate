"""
Spatial join module.

How it works:
BOM provides real-time point observations.
BARRA2 provides historical gridded averages (pre-extracted per station).
This module merges them on station_name to create an enriched DataFrame
that contains both current conditions and historical context side by side.

The anomaly columns computed here (e.g. temp_anomaly_c) are the difference
between what BOM is reading right now and what BARRA2 says is historically
normal for that location and time of year.
"""

import pandas as pd
import numpy as np


def merge_bom_barra2(df_bom: pd.DataFrame,
                     df_barra2: pd.DataFrame) -> pd.DataFrame:
    """
    Merge BOM live observations with BARRA2 historical baseline.

    df_bom:    cleaned BOM DataFrame (one row per observation)
    df_barra2: BARRA2 DataFrame (one row per station, monthly averages)

    Returns enriched DataFrame with BARRA2 columns added to each
    BOM observation, plus computed anomaly columns.
    """
    if df_bom.empty:
        return df_bom

    if df_barra2.empty:
        print("No BARRA2 data — returning BOM data without historical context.")
        return df_bom

    # BARRA2 is already one row per station
    # Merge on station_name (left join keeps all BOM records)
    merged = df_bom.merge(
        df_barra2.drop(columns=["barra2_lat", "barra2_lon"],
                        errors="ignore"),
        on="station_name",
        how="left"
    )

    # Compute anomalies: current BOM reading minus BARRA2 historical mean
    # Positive = warmer/wetter/windier than normal
    # Negative = cooler/drier/calmer than normal

    if ("temperature_c" in merged.columns and
            "barra2_temp_c" in merged.columns):
        merged["temp_anomaly_c"] = (
            merged["temperature_c"] - merged["barra2_temp_c"]
        ).round(2)

    if ("humidity_pct" in merged.columns and
            "barra2_humidity_pct" in merged.columns):
        merged["humidity_anomaly_pct"] = (
            merged["humidity_pct"] - merged["barra2_humidity_pct"]
        ).round(2)

    if ("wind_speed_kmh" in merged.columns and
            "barra2_wind_kmh" in merged.columns):
        merged["wind_anomaly_kmh"] = (
            merged["wind_speed_kmh"] - merged["barra2_wind_kmh"]
        ).round(2)

    if ("pressure_hpa" in merged.columns and
            "barra2_pressure_hpa" in merged.columns):
        merged["pressure_anomaly_hpa"] = (
            merged["pressure_hpa"] - merged["barra2_pressure_hpa"]
        ).round(2)

    # Compound anomaly flag: True if temperature AND humidity
    # both deviate significantly from historical baseline
    if ("temp_anomaly_c" in merged.columns and
            "humidity_anomaly_pct" in merged.columns):
        merged["compound_anomaly"] = (
            (merged["temp_anomaly_c"].abs() > 3.0) &
            (merged["humidity_anomaly_pct"].abs() > 15.0)
        )

    n_barra2 = merged["barra2_temp_c"].notna().sum() if "barra2_temp_c" in merged.columns else 0
    print(f"Merged: {len(merged)} records | "
          f"{n_barra2} with BARRA2 context.")

    return merged.reset_index(drop=True)


def merge_era5(df_merged: pd.DataFrame,
               df_era5: pd.DataFrame) -> pd.DataFrame:
    """
    Add ERA5 supplementary data to the merged DataFrame.
    Only adds columns for stations where BARRA2 context is missing.
    """
    if df_era5.empty or df_merged.empty:
        return df_merged

    era5_cols = ["station_name"] + [
        c for c in df_era5.columns if c.startswith("era5_")
    ]
    df_era5_subset = df_era5[era5_cols]

    merged = df_merged.merge(df_era5_subset, on="station_name", how="left")
    print(f"ERA5 added: {len(merged)} records.")
    return merged