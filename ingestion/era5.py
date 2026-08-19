"""
ERA5 ingestion module — supplementary fallback.

How it works:
- ERA5 is the ECMWF global reanalysis at 31km resolution.
- Requires a CDS API key in ~/.cdsapirc
- Slower than BARRA2 due to CDS queue times (10-30 min)
- Used when BARRA2 is unavailable or for supplementary global context
- Cached by date to avoid repeated slow queue requests

If cdsapi is not installed or key is not configured, this module
returns empty DataFrames gracefully without crashing anything.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from config import ERA5_VARIABLES, ERA5_CACHE_PATH, ERA5_CACHE_HOURS


def _is_cache_valid() -> bool:
    """Check if today's ERA5 cache exists and is recent."""
    if not os.path.exists(ERA5_CACHE_PATH):
        return False
    age_hours = (
        datetime.now().timestamp() -
        os.path.getmtime(ERA5_CACHE_PATH)
    ) / 3600
    return age_hours < ERA5_CACHE_HOURS


def fetch_era5_for_stations(station_coords: dict,
                             date_str: str = None,
                             force_refresh: bool = False
                             ) -> pd.DataFrame:
    """
    Fetch ERA5 data for all station coordinates.
    Falls back gracefully if cdsapi not configured.

    Caches results to avoid repeated CDS queue waits.
    """
    # Return cache if fresh
    if not force_refresh and _is_cache_valid():
        print("ERA5 cache is fresh — loading from cache.")
        try:
            return pd.read_parquet(ERA5_CACHE_PATH)
        except Exception:
            pass

    # Check dependencies
    try:
        import cdsapi
        import xarray as xr
    except ImportError:
        print("cdsapi or xarray not installed. Skipping ERA5.")
        return pd.DataFrame()

    # Check for API key config
    cdsapirc = os.path.expanduser("~/.cdsapirc")
    if not os.path.exists(cdsapirc):
        print("No ~/.cdsapirc found. Skipping ERA5.")
        print("Set up your Copernicus CDS API key to enable ERA5.")
        return pd.DataFrame()

    if not station_coords:
        return pd.DataFrame()

    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    year  = date_str[:4]
    month = date_str[5:7]
    day   = date_str[8:10]

    output_path = "data/raw/era5_temp.nc"
    os.makedirs("data/raw", exist_ok=True)

    try:
        print(f"Fetching ERA5 for {date_str} "
              f"({len(station_coords)} stations)...")
        print("Note: CDS queue may take 10-30 minutes.")

        c = cdsapi.Client(quiet=True)
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable":     ERA5_VARIABLES,
                "year":         year,
                "month":        month,
                "day":          day,
                "time":         ["00:00", "06:00", "12:00", "18:00"],
                "area":         [-10, 112, -45, 154],  # Australia
                "format":       "netcdf",
            },
            output_path,
        )

        ds   = xr.open_dataset(output_path)
        rows = []

        station_names = list(station_coords.keys())
        lats = [station_coords[s]["lat"] for s in station_names]
        lons = [station_coords[s]["lon"] for s in station_names]

        for i, station_name in enumerate(station_names):
            point = ds.sel(
                latitude=lats[i],
                longitude=lons[i],
                method="nearest"
            ).mean(dim="time")   # daily average

            row = {
                "station_name":      station_name,
                "era5_temp_c":       float(point["t2m"].values) - 273.15,
                "era5_dewpt_c":      float(point["d2m"].values) - 273.15,
                "era5_pressure_hpa": float(point["sp"].values)  / 100.0,
                "era5_date":         date_str,
            }
            # Wind speed from u and v components
            if "u10" in point and "v10" in point:
                u = float(point["u10"].values)
                v = float(point["v10"].values)
                row["era5_wind_kmh"] = np.sqrt(u**2 + v**2) * 3.6
            rows.append(row)

        df = pd.DataFrame(rows)
        ds.close()

        # Cache
        os.makedirs(os.path.dirname(ERA5_CACHE_PATH), exist_ok=True)
        df.to_parquet(ERA5_CACHE_PATH, index=False)
        print(f"ERA5 complete: {len(df)} stations.")
        return df

    except Exception as e:
        print(f"ERA5 fetch failed: {e}")
        print("Continuing without ERA5 — not required for core pipeline.")
        return pd.DataFrame()