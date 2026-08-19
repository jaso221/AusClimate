"""
BOM ingestion module.

How it works:
1. Scrape BOM state observation index pages to discover all active
   station product codes and WMO IDs across Australia.
2. Build individual JSON feed URLs for each station.
3. Fetch all stations in parallel using ThreadPoolExecutor.
4. Combine into a single DataFrame covering all of Australia.

Station discovery is cached for 6 hours to avoid re-scraping on
every pipeline run. Actual observation data is always fresh.

No API key required. User-Agent header required by BOM.
"""

import requests
import pandas as pd
import os
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from config import (
    BOM_STATE_INDEX_PAGES,
    BOM_JSON_BASE,
    BOM_MAX_WORKERS,
    BOM_TIMEOUT,
    BOM_STATION_CACHE_HOURS,
    COLUMN_MAPPING,
    NUMERIC_COLS,
    BOM_REQUIRE_TEMPERATURE,
    BOM_MIN_LAT, BOM_MAX_LAT,
    BOM_MIN_LON, BOM_MAX_LON,
    STATION_LIST_PATH,
)

HEADERS = {
    "User-Agent": "AusClimate/1.0 UTS Capstone Research Project"
}

# Where to cache the discovered station URL list
STATION_URL_CACHE = "data/processed/bom_station_urls.json"


def _discover_station_urls_for_state(state: str,
                                      index_url: str) -> list:
    """
    Scrape a BOM state observation index page to find all station
    product codes and WMO IDs, then build their JSON feed URLs.

    BOM observation pages contain links like:
    /products/IDN60901/IDN60901.94768.shtml

    We extract the product code (IDN60901) and station ID (94768)
    and build the JSON URL:
    http://www.bom.gov.au/fwo/IDN60901/IDN60901.94768.json
    """
    try:
        response = requests.get(
            index_url, headers=HEADERS, timeout=15
        )
        response.raise_for_status()

        soup    = BeautifulSoup(response.text, "html.parser")
        links   = soup.find_all("a", href=True)
        urls    = []

        # BOM station links follow the pattern:
        # /products/{PRODUCT_CODE}/{PRODUCT_CODE}.{WMO_ID}.shtml
        pattern = re.compile(
            r"/products/(ID[A-Z]\d+)/(ID[A-Z]\d+)\.(\d+)\.shtml"
        )

        seen = set()
        for link in links:
            href  = link["href"]
            match = pattern.search(href)
            if match:
                product_code = match.group(1)
                wmo_id       = match.group(3)
                key          = f"{product_code}.{wmo_id}"

                if key not in seen:
                    seen.add(key)
                    json_url = (
                        f"{BOM_JSON_BASE}/{product_code}/"
                        f"{product_code}.{wmo_id}.json"
                    )
                    urls.append({
                        "state":        state,
                        "product_code": product_code,
                        "wmo_id":       wmo_id,
                        "json_url":     json_url,
                    })

        print(f"  [{state}] Discovered {len(urls)} stations.")
        return urls

    except Exception as e:
        print(f"  [{state}] Failed to discover stations: {e}")
        return []


def _is_station_cache_valid() -> bool:
    """Check if the station URL cache is fresh enough to reuse."""
    if not os.path.exists(STATION_URL_CACHE):
        return False
    age_hours = (
        __import__("datetime").datetime.now().timestamp() -
        os.path.getmtime(STATION_URL_CACHE)
    ) / 3600
    return age_hours < BOM_STATION_CACHE_HOURS


def discover_all_stations(force_refresh: bool = False) -> list:
    """
    Discover all active BOM station URLs across Australia.

    Returns a list of dicts with state, product_code, wmo_id, json_url.
    Results are cached for BOM_STATION_CACHE_HOURS hours.
    """
    # Load from cache if fresh
    if not force_refresh and _is_station_cache_valid():
        try:
            with open(STATION_URL_CACHE, "r") as f:
                stations = json.load(f)
            print(f"Station list loaded from cache "
                  f"({len(stations)} stations).")
            return stations
        except Exception:
            pass

    print("Discovering BOM stations across all states...")
    all_stations = []

    # Scrape state pages in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_state = {
            executor.submit(
                _discover_station_urls_for_state, state, url
            ): state
            for state, url in BOM_STATE_INDEX_PAGES.items()
        }
        for future in as_completed(future_to_state):
            stations = future.result()
            all_stations.extend(stations)

    print(f"Total stations discovered: {len(all_stations)}")

    # Save cache
    os.makedirs(os.path.dirname(STATION_URL_CACHE), exist_ok=True)
    with open(STATION_URL_CACHE, "w") as f:
        json.dump(all_stations, f)

    return all_stations


def _fetch_single_station(station_info: dict) -> pd.DataFrame:
    """
    Fetch observations from a single BOM station JSON feed.
    Returns empty DataFrame on any failure.
    """
    url   = station_info["json_url"]
    state = station_info["state"]

    try:
        response = requests.get(
            url, headers=HEADERS, timeout=BOM_TIMEOUT
        )
        response.raise_for_status()

        data         = response.json()
        observations = data.get("observations", {}).get("data", [])

        if not observations:
            return pd.DataFrame()

        df          = pd.DataFrame(observations)
        df["state"] = state
        return df

    except requests.exceptions.Timeout:
        return pd.DataFrame()
    except requests.exceptions.HTTPError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def fetch_all_stations(force_station_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch live observations from all active BOM stations in Australia.

    Step 1: Discover all station URLs (cached, runs once per 6 hours)
    Step 2: Fetch all stations in parallel (fresh every run)

    With BOM_MAX_WORKERS=20 and ~700 stations, this typically
    completes in 15-30 seconds.

    Returns raw combined DataFrame ready for cleaning.
    """
    # Step 1: Get station list
    stations = discover_all_stations(force_refresh=force_station_refresh)

    if not stations:
        print("No stations discovered. Check network connection.")
        return pd.DataFrame()

    print(f"\nFetching observations from {len(stations)} stations "
          f"with {BOM_MAX_WORKERS} parallel workers...")

    frames      = []
    success     = 0
    failed      = 0

    # Step 2: Fetch all stations in parallel
    with ThreadPoolExecutor(max_workers=BOM_MAX_WORKERS) as executor:
        futures = {
            executor.submit(_fetch_single_station, s): s
            for s in stations
        }
        for i, future in enumerate(as_completed(futures), 1):
            df = future.result()
            if not df.empty:
                frames.append(df)
                success += 1
            else:
                failed += 1

            # Progress update every 100 stations
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(stations)} "
                      f"({success} ok, {failed} failed)...")

    if not frames:
        print("No observations fetched from any station.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Filter to Australian geographic bounds
    if "lat" in combined.columns and "lon" in combined.columns:
        combined["lat"] = pd.to_numeric(combined["lat"], errors="coerce")
        combined["lon"] = pd.to_numeric(combined["lon"], errors="coerce")
        combined = combined[
            (combined["lat"] >= BOM_MIN_LAT) &
            (combined["lat"] <= BOM_MAX_LAT) &
            (combined["lon"] >= BOM_MIN_LON) &
            (combined["lon"] <= BOM_MAX_LON)
        ]

    # Remove duplicates
    dedup_cols = [c for c in ["wmo", "local_date_time_full"]
                  if c in combined.columns]
    if dedup_cols:
        combined = combined.drop_duplicates(subset=dedup_cols)

    # Drop rows missing temperature
    if BOM_REQUIRE_TEMPERATURE and "air_temp" in combined.columns:
        combined = combined[combined["air_temp"].notna()]

    n_stations = (combined["name"].nunique()
                  if "name" in combined.columns else "?")
    n_states   = (combined["state"].nunique()
                  if "state" in combined.columns else "?")

    print(f"\nBOM fetch complete:")
    print(f"  Stations fetched : {success}/{len(stations)}")
    print(f"  Total records    : {len(combined)}")
    print(f"  Unique stations  : {n_stations}")
    print(f"  States covered   : {n_states}")

    return combined.reset_index(drop=True)


def get_station_coords(df_bom: pd.DataFrame) -> dict:
    """
    Extract {station_name: {lat, lon}} from raw BOM DataFrame.
    Used to pass station locations to BARRA2 and ERA5 fetchers.
    """
    if df_bom.empty:
        return {}

    name_col = "name" if "name" in df_bom.columns else None
    lat_col  = "lat"  if "lat"  in df_bom.columns else None
    lon_col  = "lon"  if "lon"  in df_bom.columns else None

    if not all([name_col, lat_col, lon_col]):
        return {}

    df_coords = df_bom[[name_col, lat_col, lon_col]].copy()
    df_coords[lat_col] = pd.to_numeric(df_coords[lat_col], errors="coerce")
    df_coords[lon_col] = pd.to_numeric(df_coords[lon_col], errors="coerce")
    df_coords = (df_coords.dropna()
                          .drop_duplicates(subset=[name_col]))

    coords = {
        row[name_col]: {
            "lat": float(row[lat_col]),
            "lon": float(row[lon_col])
        }
        for _, row in df_coords.iterrows()
    }

    print(f"Extracted coordinates for {len(coords)} stations.")
    return coords