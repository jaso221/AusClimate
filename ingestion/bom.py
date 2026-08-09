import requests
import pandas as pd
from config import BOM_STATIONS

HEADERS = {"User-Agent": "AusClimate/1.0 UTS Capstone Project"}


def fetch_bom_station(url: str) -> pd.DataFrame:
    """
    Fetch live observations from a single BOM station JSON feed.
    Returns an empty DataFrame on failure.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        observations = data.get("observations", {}).get("data", [])
        if not observations:
            return pd.DataFrame()
        return pd.DataFrame(observations)
    except Exception as e:
        print(f"  Failed to fetch {url}: {e}")
        return pd.DataFrame()


def fetch_all_stations() -> pd.DataFrame:
    """
    Fetch live observations from all configured BOM stations.
    Returns a single concatenated DataFrame.
    """
    frames = []
    print(f"Fetching BOM data from {len(BOM_STATIONS)} stations...")

    for station_name, url in BOM_STATIONS.items():
        print(f"  Fetching: {station_name}")
        df = fetch_bom_station(url)
        if not df.empty:
            df["station_name"] = station_name
            frames.append(df)

    if not frames:
        print("No data fetched from any station.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    print(f"Fetched {len(combined)} total records from {len(frames)} stations.")
    return combined