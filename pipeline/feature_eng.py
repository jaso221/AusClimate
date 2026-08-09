import pandas as pd
import numpy as np


def compute_station_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute station-level aggregate features.
    Returns one row per station with summary statistics.
    """
    if df.empty:
        return df

    numeric_cols = [
        "temperature_c", "feels_like_c", "humidity_pct",
        "wind_speed_kmh", "gust_kmh", "pressure_hpa"
    ]

    # Only use columns that exist
    available = [c for c in numeric_cols if c in df.columns]

    agg_dict = {}
    for col in available:
        agg_dict[f"avg_{col}"]  = (col, "mean")
        agg_dict[f"max_{col}"]  = (col, "max")
        agg_dict[f"min_{col}"]  = (col, "min")

    agg_dict["record_count"] = ("temperature_c", "count")

    summary = df.groupby("station_name").agg(**agg_dict).reset_index()

    # Temperature range
    if "max_temperature_c" in summary and "min_temperature_c" in summary:
        summary["temp_range_c"] = (
            summary["max_temperature_c"] - summary["min_temperature_c"]
        ).round(1)

    # Heat stress score: combines temperature and humidity
    # Simple index: (avg_temp * 0.6) + (avg_humidity * 0.4 / 10)
    if "avg_temperature_c" in summary and "avg_humidity_pct" in summary:
        summary["heat_stress_score"] = (
            (summary["avg_temperature_c"] * 0.6) +
            (summary["avg_humidity_pct"] * 0.4 / 10)
        ).round(2)

    summary = summary.round(2)
    print(f"Feature engineering complete: {len(summary)} station summaries")
    return summary