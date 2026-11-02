"""
BARRA2 ingestion module.

BARRA-R2 is BOM's own high-resolution atmospheric reanalysis for Australia.
Coverage: 1979 to present, updated monthly, 3-4 months behind present day.
Resolution: 12km horizontal (AUS-11 grid), 1-hourly data.
Access: NCI THREDDS public server - no login required.
Licence: Creative Commons Attribution 4.0 (CC-BY 4.0)

Confirmed URL format from github.com/akarich73/barra2-dl:
{var}_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{YYYYMM}-{YYYYMM}.nc

Performance strategy:
- Download the full Australian grid ONCE per variable (one HTTP request)
- Extract all 700+ station values simultaneously using vectorized
  xarray .sel() — this is fast regardless of station count
- Cache results for 24 hours — historical baseline does not change hourly
- Try multiple months automatically until a working date is found

Key known issue: requests for dates within the last 6 months will 404.
Safe baseline is 2022 data which is fully confirmed available.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import re
import math
import tempfile
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config import (
    BARRA2_BASE,
    BARRA2_VARIABLES,
    AUSTRALIA_BOUNDS,
    BARRA2_CACHE_PATH,
    BARRA2_CACHE_HOURS,
)

# ── Constants ─────────────────────────────────────────────────────────────────

# BARRA2 is 3-4 months behind present. We go back 6 months to be safe,
# then fall back to known-good dates if those also fail.
SAFE_FALLBACK_DATES = [
    # Try recent months first (may exist if 3-4 month lag allows)
    # Then fall back to confirmed complete years
    (2024, 6),
    (2024, 3),
    (2024, 1),
    (2023, 12),
    (2023, 9),
    (2023, 6),
    (2023, 1),
    (2022, 12),  # Confirmed complete historical run ends at 2022
    (2022, 9),
    (2022, 6),
    (2022, 1),
]

HEADERS = {
    "User-Agent": "AusClimate/1.0 UTS Capstone Research Project"
}

# Supported THREDDS frequencies to try in order
FREQUENCIES_TO_TRY = ["1hr", "day"]


# ── Unit conversion ───────────────────────────────────────────────────────────

def _convert_units(series: pd.Series, conversion: str) -> pd.Series:
    """Apply unit conversion to a pandas Series."""
    if conversion is None:
        return series
    if conversion == "kelvin_to_celsius":
        return series - 273.15
    if conversion == "ms_to_kmh":
        return series * 3.6
    if conversion == "pa_to_hpa":
        return series / 100.0
    if conversion == "kgm2s_to_mm":
        # kg/m2/s * 86400 = mm/day (for daily data)
        # kg/m2/s * 3600  = mm/hr  (for hourly data)
        return series * 86400.0
    return series


# ── URL builders ──────────────────────────────────────────────────────────────

def _build_filename(variable: str, year: int,
                    month: int, freq: str) -> str:
    """
    Build BARRA2 NetCDF filename.

    Confirmed format from github.com/akarich73/barra2-dl:
    For 1hr: {var}_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_{YYYYMM}-{YYYYMM}.nc
    For day: {var}_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_day_{YYYYMM}.nc
    """
    ym = f"{year}{month:02d}"
    if freq == "1hr":
        return (
            f"{variable}_AUS-11_ERA5_historical_hres_"
            f"BOM_BARRA-R2_v1_1hr_{ym}-{ym}.nc"
        )
    else:
        # Daily format — try both with and without end date
        return (
            f"{variable}_AUS-11_ERA5_historical_hres_"
            f"BOM_BARRA-R2_v1_day_{ym}.nc"
        )


def _build_ncss_url(variable: str, year: int, month: int,
                    freq: str) -> str:
    """
    Build the full NCI THREDDS NCSS query URL for an Australian
    spatial subset.

    Adds query parameters:
    - var: the variable name
    - spatial bounds covering all of Australia
    - time range for the full month
    - horizStride=2: every other grid point (reduces download size ~4x)
    - accept=netcdf3: most compatible format
    """
    filename  = _build_filename(variable, year, month, freq)
    base_path = (
        f"{BARRA2_BASE}/{freq}/{variable}/latest/{filename}"
    )

    # Build start/end times for the month
    start = f"{year}-{month:02d}-01T00:00:00Z"
    # Use day 28 as end to safely stay within any month
    end   = f"{year}-{month:02d}-28T23:59:59Z"

    query = (
        f"?var={variable}"
        f"&north={AUSTRALIA_BOUNDS['north']}"
        f"&south={AUSTRALIA_BOUNDS['south']}"
        f"&east={AUSTRALIA_BOUNDS['east']}"
        f"&west={AUSTRALIA_BOUNDS['west']}"
        f"&horizStride=2"       # Every 2nd grid point — reduces size ~4x
        f"&time_start={start}"
        f"&time_end={end}"
        f"&accept=netcdf3"
    )

    return base_path + query


def _build_point_url(variable: str, year: int, month: int,
                     freq: str, lat: float, lon: float) -> str:
    """
    Build URL to fetch a single point as CSV.
    Used for testing connectivity and as fallback.
    """
    filename  = _build_filename(variable, year, month, freq)
    base_path = (
        f"{BARRA2_BASE}/{freq}/{variable}/latest/{filename}"
    )

    start = f"{year}-{month:02d}-01T00:00:00Z"
    end   = f"{year}-{month:02d}-07T23:59:59Z"  # First week only

    query = (
        f"?var={variable}"
        f"&latitude={lat}"
        f"&longitude={lon}"
        f"&time_start={start}"
        f"&time_end={end}"
        f"&accept=csv"
    )

    return base_path + query


# ── Connectivity test ─────────────────────────────────────────────────────────

def test_barra2_connectivity() -> tuple:
    """
    Test BARRA2 THREDDS connectivity by trying a small point request.
    Returns (success: bool, working_year: int, working_month: int,
             working_freq: str).

    Tries dates from SAFE_FALLBACK_DATES until one works.
    """
    test_lat = -33.86   # Sydney
    test_lon = 151.21

    for freq in FREQUENCIES_TO_TRY:
        for year, month in SAFE_FALLBACK_DATES:
            url = _build_point_url(
                "tas", year, month, freq, test_lat, test_lon
            )
            try:
                r = requests.get(url, headers=HEADERS, timeout=30)
                if r.status_code == 200 and len(r.content) > 100:
                    print(f"  BARRA2 connectivity OK: "
                          f"{year}-{month:02d}, freq={freq}")
                    return True, year, month, freq
                elif r.status_code == 404:
                    continue
                else:
                    print(f"  BARRA2 HTTP {r.status_code} for "
                          f"{year}-{month:02d} ({freq})")
                    continue
            except Exception as e:
                print(f"  BARRA2 connection error: {e}")
                continue

    print("  BARRA2: no working date found in SAFE_FALLBACK_DATES.")
    return False, None, None, None


# ── Cache management ──────────────────────────────────────────────────────────

def _cache_is_valid() -> bool:
    """
    True only if cache file exists, is recent enough,
    AND contains actual non-NaN BARRA2 climate values.
    A cache full of NaN values is not valid.
    """
    if not os.path.exists(BARRA2_CACHE_PATH):
        return False

    age_h = (
        datetime.now().timestamp() -
        os.path.getmtime(BARRA2_CACHE_PATH)
    ) / 3600.0

    if age_h >= BARRA2_CACHE_HOURS:
        return False

    # Check the cache actually has real data not just NaN
    try:
        df = pd.read_parquet(BARRA2_CACHE_PATH)
        barra2_data_cols = [
            v["clean"] for v in BARRA2_VARIABLES.values()
            if v["clean"] in df.columns
        ]
        if not barra2_data_cols:
            return False
        # At least one variable must have real values
        has_data = df[barra2_data_cols].notna().any().any()
        if not has_data:
            print("  BARRA2 cache exists but contains only NaN - will re-fetch.")
        return has_data
    except Exception:
        return False


def _save_cache(df: pd.DataFrame) -> None:
    """
    Save BARRA2 cache only if it contains actual data.
    Never cache a DataFrame that is entirely NaN.
    """
    barra2_data_cols = [
        v["clean"] for v in BARRA2_VARIABLES.values()
        if v["clean"] in df.columns
    ]

    if not barra2_data_cols:
        print("  BARRA2: no data columns to cache - skipping.")
        return

    has_data = df[barra2_data_cols].notna().any().any()
    if not has_data:
        print("  BARRA2: all values are NaN - not caching empty result.")
        print("  This means THREDDS returned no data for any variable.")
        print("  Pipeline will continue with BOM data only.")
        return

    os.makedirs(os.path.dirname(BARRA2_CACHE_PATH), exist_ok=True)
    df.to_parquet(BARRA2_CACHE_PATH, index=False)
    size_kb = os.path.getsize(BARRA2_CACHE_PATH) / 1024

    valid_cols = df[barra2_data_cols].notna().sum()
    print(f"  BARRA2 cache saved: {BARRA2_CACHE_PATH} "
          f"({len(df)} rows, {size_kb:.1f} KB)")
    print(f"  Valid values per variable:")
    for col, count in valid_cols.items():
        print(f"    {col}: {count}/{len(df)} stations")


def _load_cache() -> pd.DataFrame:
    try:
        df = pd.read_parquet(BARRA2_CACHE_PATH)
        print(f"  BARRA2 cache loaded: {len(df)} stations.")
        return df
    except Exception as e:
        print(f"  Cache read failed: {e}")
        return pd.DataFrame()


# ── Core fetching ─────────────────────────────────────────────────────────────

def _fetch_variable_grid(variable: str, year: int,
                          month: int, freq: str) -> bytes:
    """
    Download the BARRA2 NetCDF grid for one variable covering
    all of Australia. Returns raw bytes or raises on failure.
    """
    url = _build_ncss_url(variable, year, month, freq)
    print(f"    GET {variable} {year}-{month:02d} ({freq})...")

    r = requests.get(url, headers=HEADERS, timeout=180)
    r.raise_for_status()

    # Sanity check: NetCDF files start with 'CDF' magic bytes
    if len(r.content) < 500:
        raise ValueError(
            f"Response too small ({len(r.content)} bytes) - "
            f"likely an error page not a NetCDF file."
        )

    return r.content


def _extract_all_stations(nc_bytes: bytes,
                           variable: str,
                           station_names: list,
                           lats: np.ndarray,
                           lons: np.ndarray,
                           conversion: str) -> np.ndarray:
    """
    Extract the variable's monthly mean for all stations simultaneously
    from a BARRA2 NetCDF grid, using vectorized xarray selection.

    This is the core performance trick: instead of looping over stations
    (which would mean hundreds of array lookups), we pass lists of
    coordinates and xarray selects all of them in a single operation.

    Returns array of monthly means, one value per station.
    """
    import xarray as xr

    tmp_path = None
    try:
        # Write bytes to a temp file so xarray can open it
        with tempfile.NamedTemporaryFile(
            suffix=".nc", delete=False
        ) as tmp:
            tmp.write(nc_bytes)
            tmp_path = tmp.name

        ds = xr.open_dataset(tmp_path, engine="netcdf4")

        # Find the correct dimension names (vary between files)
        dims      = list(ds.dims)
        lat_dim   = next(
            (d for d in ["lat", "latitude", "y", "rlat"]
             if d in dims), None
        )
        lon_dim   = next(
            (d for d in ["lon", "longitude", "x", "rlon"]
             if d in dims), None
        )
        time_dim  = next(
            (d for d in ["time", "t"] if d in dims), None
        )

        if lat_dim is None or lon_dim is None:
            raise ValueError(
                f"Cannot identify lat/lon dims. Available: {dims}"
            )

        # Vectorized selection: pass all station coords as DataArrays
        # xarray selects the nearest grid cell for each simultaneously
        points = ds[variable].sel(
            {
                lat_dim: xr.DataArray(lats, dims="station"),
                lon_dim: xr.DataArray(lons, dims="station"),
            },
            method="nearest"
        )

        # Average over the time dimension to get the monthly mean
        if time_dim and time_dim in points.dims:
            monthly_mean = points.mean(dim=time_dim).values
        else:
            monthly_mean = points.values

        ds.close()

        # Apply unit conversion
        series = pd.Series(monthly_mean.astype(float))
        series = _convert_units(series, conversion)

        return series.round(3).values

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ── Main public function ──────────────────────────────────────────────────────

def fetch_barra2_all_stations(station_coords: dict,
                               force_refresh: bool = False
                               ) -> pd.DataFrame:
    """
    Fetch BARRA2 historical baseline data for all BOM stations.

    Parameters
    ----------
    station_coords : dict
        {station_name: {'lat': float, 'lon': float}}
        Typically 600-700 stations from the BOM pipeline.
    force_refresh : bool
        If True, ignore cache and re-fetch from THREDDS.

    Returns
    -------
    pd.DataFrame
        One row per station. Columns include barra2_temp_c,
        barra2_temp_max_c, barra2_temp_min_c, barra2_humidity_pct,
        barra2_wind_kmh, barra2_precip_mm, barra2_pressure_hpa.
        Returns empty DataFrame if THREDDS is unreachable.

    Notes
    -----
    - BARRA2 is 3-4 months behind present day.
    - The function automatically finds the most recent available month.
    - Results are cached for BARRA2_CACHE_HOURS (default 24h).
    - Grid download is done ONCE per variable — all stations are
      extracted simultaneously using vectorized xarray.sel().
    - With horizStride=2, each grid file is ~50-200 MB.
    """

    # ── Check cache ───────────────────────────────────────────────
    if not force_refresh and _cache_is_valid():
        print("BARRA2 cache is fresh - loading from cache.")
        df = _load_cache()
        if not df.empty:
            return df

    if not station_coords:
        print("BARRA2: no station coordinates provided.")
        return pd.DataFrame()

    # ── Check xarray is available ─────────────────────────────────
    try:
        import xarray as xr
    except ImportError:
        print("xarray not installed. Run: pip install xarray netCDF4")
        return pd.DataFrame()

    # ── Test connectivity and find a working date ─────────────────
    print("\nBARRA2: testing THREDDS connectivity...")
    ok, year, month, freq = test_barra2_connectivity()

    if not ok:
        print("BARRA2: THREDDS unreachable or no available data found.")
        print("  Pipeline will continue with BOM data only.")
        print("  Possible reasons:")
        print("  1. NCI THREDDS is temporarily down")
        print("  2. Network issue from your location")
        print("  3. All tried dates are not yet released")
        # Return empty DataFrame with correct columns so downstream
        # code does not break
        empty = pd.DataFrame({"station_name": list(station_coords.keys())})
        for var_info in BARRA2_VARIABLES.values():
            empty[var_info["clean"]] = np.nan
        return empty

    print(f"\nBARRA2: fetching baseline ({year}-{month:02d}, "
          f"freq={freq}) for {len(station_coords)} stations...")

    # ── Prepare station arrays ────────────────────────────────────
    station_names = list(station_coords.keys())
    lats          = np.array(
        [station_coords[s]["lat"] for s in station_names]
    )
    lons          = np.array(
        [station_coords[s]["lon"] for s in station_names]
    )

    # ── Build result DataFrame ────────────────────────────────────
    result = pd.DataFrame({
        "station_name":    station_names,
        "barra2_lat":      lats,
        "barra2_lon":      lons,
        "barra2_baseline_year":  year,
        "barra2_baseline_month": month,
        "barra2_freq":     freq,
    })

    # Initialise all variable columns with NaN
    for var_info in BARRA2_VARIABLES.values():
        result[var_info["clean"]] = np.nan

    # ── Fetch each variable ───────────────────────────────────────
    success_count = 0
    fail_count    = 0

    for var_name, var_info in BARRA2_VARIABLES.items():
        clean_name = var_info["clean"]
        conversion = var_info["convert"]
        desc       = var_info["desc"]
        unit       = var_info["unit"]

        # Try the confirmed working date first, then fall back
        dates_to_try = [(year, month, freq)]

        # Add fallback dates in case this specific variable has gaps
        for fy, fm in SAFE_FALLBACK_DATES:
            if (fy, fm) != (year, month):
                dates_to_try.append((fy, fm, freq))
                if freq != "1hr":
                    dates_to_try.append((fy, fm, "1hr"))

        fetched = False
        for try_year, try_month, try_freq in dates_to_try:
            try:
                nc_bytes = _fetch_variable_grid(
                    var_name, try_year, try_month, try_freq
                )

                values = _extract_all_stations(
                    nc_bytes, var_name,
                    station_names, lats, lons, conversion
                )

                result[clean_name] = values

                valid = np.sum(~np.isnan(values))
                v_min = np.nanmin(values)
                v_max = np.nanmax(values)
                print(
                    f"    {desc} ({unit}): "
                    f"{v_min:.1f} to {v_max:.1f} | "
                    f"{valid}/{len(values)} valid"
                )

                success_count += 1
                fetched        = True
                break

            except requests.exceptions.HTTPError as e:
                code = e.response.status_code if e.response else "?"
                if code == 404:
                    # Expected for dates that don't exist yet
                    continue
                else:
                    print(f"    HTTP {code} for {var_name} "
                          f"({try_year}-{try_month:02d}): {e}")
                    break

            except requests.exceptions.Timeout:
                print(f"    Timeout for {var_name} "
                      f"({try_year}-{try_month:02d}) = skipping.")
                break

            except ValueError as e:
                print(f"    Data error for {var_name}: {e}")
                break

            except Exception as e:
                print(f"    Unexpected error for {var_name}: {e}")
                break

        if not fetched:
            print(f"    {desc}: no data found - column set to NaN.")
            fail_count += 1

    # ── Save cache ────────────────────────────────────────────────
    _save_cache(result)

    # ── Summary ───────────────────────────────────────────────────
    barra2_data_cols = [
        v["clean"] for v in BARRA2_VARIABLES.values()
        if v["clean"] in result.columns
    ]
    stations_with_data = result[barra2_data_cols].notna().any(axis=1).sum()

    print(f"\nBARRA2 complete:")
    print(f"  Stations         : {len(result)}")
    print(f"  Stations with data: {stations_with_data}")
    print(f"  Variables fetched: {success_count}/{len(BARRA2_VARIABLES)}")
    print(f"  Variables failed : {fail_count}")
    print(f"  Baseline period  : {year}-{month:02d} ({freq})")

    return result


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run this file directly to test BARRA2 connectivity:
    python ingestion/barra2.py
    """
    print("=" * 55)
    print("BARRA2 Connectivity Test")
    print("=" * 55)

    ok, year, month, freq = test_barra2_connectivity()

    if ok:
        print(f"\nWorking date: {year}-{month:02d}, freq={freq}")
        print("\nTesting small fetch for Sydney (tas only)...")

        url = _build_point_url("tas", year, month, freq,
                               lat=-33.86, lon=151.21)
        print(f"URL: {url}")

        r = requests.get(url, headers=HEADERS, timeout=30)
        print(f"Status: {r.status_code}")
        print(f"Content preview:\n{r.text[:500]}")
    else:
        print("\nNo working BARRA2 connection found.")
        print("Check network and NCI THREDDS status.")