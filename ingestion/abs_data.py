"""
ingestion/abs_data.py

ABS (Australian Bureau of Statistics) integration for AusClimate.

What this module does:
- Fetches population per SA2 region from the ABS API
- Fetches SEIFA disadvantage score per SA2 from the ABS API
- Matches each BOM station to the nearest known population centre
- Computes a simple Population Risk Score

Population Risk Score:
    How much risk a station poses to people nearby.
    It combines climate stress (from the ML layer) with population
    exposure so a dangerous reading near 100,000 people ranks higher
    than the same reading near 500 people.

SEIFA IRSD (Socio-Economic Disadvantage):
    Score of 1000 = national average
    Below 1000 = more disadvantaged than average
    Above 1000 = less disadvantaged than average
    Disadvantaged communities matter more because they have fewer
    resources to cope with extreme climate events.

No API key required. Data is freely available under CC-BY 4.0.
Source: https://data.api.abs.gov.au
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import requests
from datetime import datetime
from io import StringIO

# ── File paths ────────────────────────────────────────────────────────────────

ABS_CACHE_PATH   = "data/processed/abs_station_data.parquet"
ABS_CACHE_HOURS  = 168   # 7 days -- population data changes very slowly

# ── ABS API URLs ──────────────────────────────────────────────────────────────
# No API key needed. Returns CSV directly.
# URL changed November 2024 to data.api.abs.gov.au

ABS_ERP_URL = (
    "https://data.api.abs.gov.au/rest/data/"
    "ABS,ABS_ANNUAL_ERP_ASGS2021,1.0.0/"
    "A.TOT_P.SA2.AUS.?"
    "format=csvfilewithlabels&startPeriod=2022&endPeriod=2022"
)

ABS_SEIFA_URL = (
    "https://data.api.abs.gov.au/rest/data/"
    "ABS,SEIFA_2021_SA2,1.0.0/"
    "A.IRSD_SCORE.SA2.AUS.?"
    "format=csvfilewithlabels"
)

HEADERS = {
    "User-Agent": "AusClimate/1.0 UTS Capstone Research Project",
}

# ── Population lookup for major Australian centres ────────────────────────────
# Used when the ABS API is unavailable.
# Sources: ABS 2022 ERP and SEIFA 2021 published figures.
# Each entry is one major population centre per region.

POPULATION_CENTRES = [
    # State,  Name,               Lat,     Lon,      Population, SEIFA IRSD
    ("NSW",  "Sydney",           -33.87,  151.21,  5_259_764,  1002),
    ("NSW",  "Newcastle",        -32.93,  151.78,    319_278,   977),
    ("NSW",  "Wollongong",       -34.42,  150.89,    314_647,   970),
    ("NSW",  "Central Coast",    -33.43,  151.34,    341_625,   966),
    ("NSW",  "Wagga Wagga",      -35.12,  147.37,     67_085,   960),
    ("NSW",  "Albury",           -36.08,  146.92,     56_093,   967),
    ("NSW",  "Orange",           -33.28,  149.10,     44_321,   942),
    ("NSW",  "Dubbo",            -32.26,  148.60,     46_390,   919),
    ("NSW",  "Tamworth",         -31.09,  150.93,     62_345,   940),
    ("NSW",  "Broken Hill",      -31.95,  141.47,     17_123,   895),
    ("VIC",  "Melbourne",        -37.81,  144.96,  5_192_624,  1012),
    ("VIC",  "Geelong",          -38.15,  144.36,    270_173,   979),
    ("VIC",  "Ballarat",         -37.56,  143.85,    116_953,   966),
    ("VIC",  "Bendigo",          -36.76,  144.28,    116_345,   958),
    ("VIC",  "Shepparton",       -36.38,  145.40,     67_080,   932),
    ("VIC",  "Warrnambool",      -38.38,  142.48,     37_622,   973),
    ("VIC",  "Wodonga",          -36.12,  146.89,     43_561,   963),
    ("QLD",  "Brisbane",         -27.47,  153.02,  2_567_447,  997),
    ("QLD",  "Gold Coast",       -28.02,  153.40,    763_493,   988),
    ("QLD",  "Sunshine Coast",   -26.65,  153.07,    379_087,   1011),
    ("QLD",  "Townsville",       -19.26,  146.82,    234_783,   951),
    ("QLD",  "Cairns",           -16.92,  145.78,    163_654,   951),
    ("QLD",  "Toowoomba",        -27.56,  151.95,    181_430,   963),
    ("QLD",  "Rockhampton",      -23.38,  150.51,     83_293,   943),
    ("QLD",  "Mackay",           -21.14,  149.19,    128_756,   960),
    ("QLD",  "Bundaberg",        -24.87,  152.35,     98_321,   929),
    ("QLD",  "Mount Isa",        -20.73,  139.49,     21_942,   906),
    ("SA",   "Adelaide",         -34.93,  138.60,  1_408_669,  982),
    ("SA",   "Mount Gambier",    -37.83,  140.78,     30_432,   948),
    ("SA",   "Whyalla",          -33.03,  137.59,     22_056,   905),
    ("SA",   "Port Augusta",     -32.50,  137.77,     14_891,   880),
    ("WA",   "Perth",            -31.95,  115.86,  2_202_029,  1001),
    ("WA",   "Bunbury",          -33.33,  115.64,     78_147,   981),
    ("WA",   "Geraldton",        -28.78,  114.61,     42_701,   953),
    ("WA",   "Kalgoorlie",       -30.75,  121.47,     32_481,   944),
    ("WA",   "Albany",           -35.03,  117.88,     39_118,   971),
    ("WA",   "Karratha",         -20.74,  116.85,     25_362,   978),
    ("WA",   "Port Hedland",     -20.32,  118.57,     17_491,   952),
    ("TAS",  "Hobart",           -42.88,  147.33,    247_532,   964),
    ("TAS",  "Launceston",       -41.44,  147.14,    110_953,   953),
    ("TAS",  "Devonport",        -41.18,  146.36,     30_127,   940),
    ("TAS",  "Burnie",           -41.06,  145.91,     19_863,   921),
    ("NT",   "Darwin",           -12.46,  130.84,    148_564,   946),
    ("NT",   "Alice Springs",    -23.70,  133.88,     28_952,   893),
    ("NT",   "Palmerston",       -12.49,  130.97,     41_312,   942),
    ("ACT",  "Canberra North",   -35.28,  149.13,    219_432,  1058),
    ("ACT",  "Canberra South",   -35.42,  149.08,    219_431,  1052),
]


# ── Cache check ───────────────────────────────────────────────────────────────

def _cache_valid() -> bool:
    """True if local cache file exists and is under 7 days old."""
    if not os.path.exists(ABS_CACHE_PATH):
        return False
    age_hours = (
        datetime.now().timestamp() -
        os.path.getmtime(ABS_CACHE_PATH)
    ) / 3600
    return age_hours < ABS_CACHE_HOURS


# ── ABS API fetch ─────────────────────────────────────────────────────────────

def _try_abs_api(url: str, label: str) -> pd.DataFrame:
    """
    Try to fetch data from the ABS API.
    Returns an empty DataFrame if the request fails for any reason.
    The pipeline continues normally if this fails.
    """
    try:
        print(f"  ABS: trying to fetch {label}...")
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        print(f"  ABS: {label} fetched ({len(df):,} rows)")
        return df
    except Exception as e:
        print(f"  ABS: {label} unavailable ({type(e).__name__}) -- "
              f"using built-in data")
        return pd.DataFrame()


def _extract_population(df: pd.DataFrame) -> dict:
    """
    Extract SA2 name to population mapping from ABS ERP CSV.
    Returns empty dict if parsing fails.
    """
    if df.empty:
        return {}
    df.columns = df.columns.str.strip()

    # Find the value column
    val_col = next(
        (c for c in df.columns
         if "observation" in c.lower() or "value" in c.lower()),
        None
    )
    # Find the region name column
    name_col = next(
        (c for c in df.columns
         if "region" in c.lower() and "code" not in c.lower()),
        None
    )
    if not val_col or not name_col:
        return {}

    result = {}
    for _, row in df.iterrows():
        name = str(row[name_col]).strip()
        val  = pd.to_numeric(row[val_col], errors="coerce")
        if name and not pd.isna(val) and val > 0:
            result[name] = int(val)
    return result


def _extract_seifa(df: pd.DataFrame) -> dict:
    """
    Extract SA2 name to SEIFA IRSD score mapping from ABS SEIFA CSV.
    Returns empty dict if parsing fails.
    """
    if df.empty:
        return {}
    df.columns = df.columns.str.strip()

    val_col = next(
        (c for c in df.columns
         if "observation" in c.lower() or "value" in c.lower()
         or "score" in c.lower()),
        None
    )
    name_col = next(
        (c for c in df.columns
         if "region" in c.lower() and "code" not in c.lower()),
        None
    )
    if not val_col or not name_col:
        return {}

    result = {}
    for _, row in df.iterrows():
        name  = str(row[name_col]).strip()
        score = pd.to_numeric(row[val_col], errors="coerce")
        if name and not pd.isna(score):
            result[name] = float(score)
    return result


# ── Nearest centre matching ───────────────────────────────────────────────────

def _find_nearest_centre(
    station_lat: float,
    station_lon: float,
    centres_arr: np.ndarray,
) -> int:
    """
    Return the index of the nearest population centre
    using simple Euclidean distance on lat/lon.
    """
    dlat = centres_arr[:, 0] - station_lat
    dlon = centres_arr[:, 1] - station_lon
    dists = np.sqrt(dlat**2 + dlon**2)
    return int(np.argmin(dists))


# ── Main public function ──────────────────────────────────────────────────────

def fetch_abs_data(
    station_coords: dict,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch ABS population and SEIFA data matched to BOM stations.

    For each BOM station this returns:
    - nearby_population:  how many people live near this station
    - seifa_score:        area disadvantage (lower = more disadvantaged)
    - nearest_centre:     name of the matched population centre
    - distance_km:        how far the station is from that centre
    - population_risk:    simple 0-100 score combining population + climate

    Parameters
    ----------
    station_coords : {station_name: {'lat': float, 'lon': float}}
    force_refresh  : ignore existing cache and re-fetch

    Returns
    -------
    pd.DataFrame with one row per station
    """
    # Load cache if fresh
    if not force_refresh and _cache_valid():
        try:
            cached = pd.read_parquet(ABS_CACHE_PATH)
            print(f"  ABS: loaded from cache ({len(cached)} stations).")
            return cached
        except Exception:
            pass

    if not station_coords:
        print("  ABS: no station coordinates provided.")
        return pd.DataFrame()

    print("  ABS: building population lookup...")

    # Try to enrich built-in data with live ABS API values
    pop_from_api   = _extract_population(
        _try_abs_api(ABS_ERP_URL, "population by SA2")
    )
    seifa_from_api = _extract_seifa(
        _try_abs_api(ABS_SEIFA_URL, "SEIFA scores by SA2")
    )

    # Build the centres DataFrame from the built-in lookup
    # Enrich with API data where available
    centres = []
    for state, name, lat, lon, pop, seifa in POPULATION_CENTRES:
        # Try to get a more accurate value from the API
        api_pop   = pop_from_api.get(name, pop)
        api_seifa = seifa_from_api.get(name, seifa)
        centres.append({
            "name":       name,
            "state":      state,
            "lat":        lat,
            "lon":        lon,
            "population": api_pop,
            "seifa":      api_seifa,
        })

    df_centres = pd.DataFrame(centres)
    centres_arr = df_centres[["lat", "lon"]].values

    print(f"  ABS: {len(df_centres)} population centres loaded")

    # Match each BOM station to nearest centre
    rows = []
    for name, coords in station_coords.items():
        lat = coords.get("lat", coords.get("latitude", np.nan))
        lon = coords.get("lon", coords.get("longitude", np.nan))

        if pd.isna(lat) or pd.isna(lon):
            continue

        idx       = _find_nearest_centre(lat, lon, centres_arr)
        centre    = df_centres.iloc[idx]
        dist_km   = round(
            np.sqrt(
                (centre["lat"] - lat)**2 +
                (centre["lon"] - lon)**2
            ) * 111.0,
            1
        )

        rows.append({
            "station_name":    name,
            "nearest_centre":  centre["name"],
            "centre_state":    centre["state"],
            "distance_km":     dist_km,
            "population":      int(centre["population"]),
            "seifa_score":     float(centre["seifa"]),
        })

    if not rows:
        print("  ABS: no stations matched.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Compute simple 0-100 population exposure score
    pop  = df["population"].astype(float)
    p_min, p_max = pop.min(), pop.max()
    if p_max > p_min:
        df["population_exposure"] = (
            (pop - p_min) / (p_max - p_min) * 100
        ).round(1)
    else:
        df["population_exposure"] = 50.0

    # Compute simple disadvantage score (0-100, higher = more disadvantaged)
    seifa    = df["seifa_score"].astype(float)
    # Flip SEIFA: lower SEIFA means higher disadvantage
    s_min, s_max = seifa.min(), seifa.max()
    if s_max > s_min:
        df["disadvantage_score"] = (
            (s_max - seifa) / (s_max - s_min) * 100
        ).round(1)
    else:
        df["disadvantage_score"] = 50.0

    # Disadvantage label -- easy for non-technical users to read
    df["disadvantage_label"] = df["seifa_score"].apply(
        lambda s:
        "High disadvantage"    if s < 900  else
        "Above average"        if s < 970  else
        "Average"              if s < 1030 else
        "Below average"        if s < 1070 else
        "Low disadvantage"
    )

    # Population category label
    df["population_label"] = df["population"].apply(
        lambda p:
        "Major city"     if p > 1_000_000 else
        "Large city"     if p > 200_000   else
        "Medium city"    if p > 50_000    else
        "Small city"     if p > 10_000    else
        "Regional town"
    )

    # Save cache
    os.makedirs(os.path.dirname(ABS_CACHE_PATH), exist_ok=True)
    df.to_parquet(ABS_CACHE_PATH, index=False)

    print(f"  ABS: {len(df)} stations matched to population centres.")
    return df


def compute_population_risk(
    df_summary: pd.DataFrame,
    df_abs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine climate vulnerability with population exposure.

    Population Risk Score (0-100):
    - Takes the station's existing vulnerability index
    - Weights it by how many people live nearby
    - Stations near large cities with high climate stress score highest

    This means a council officer can sort by Population Risk Score
    and immediately see which stations need the most attention.

    Parameters
    ----------
    df_summary : station summary DataFrame (has vulnerability_index)
    df_abs     : output from fetch_abs_data()

    Returns
    -------
    df_summary with three new columns:
        population_risk_score  -- 0 to 100
        population_exposure    -- 0 to 100 (how many people nearby)
        disadvantage_score     -- 0 to 100 (how disadvantaged the area)
    """
    if df_abs.empty or df_summary.empty:
        return df_summary

    result = df_summary.merge(
        df_abs[[
            "station_name", "nearest_centre", "population",
            "seifa_score", "distance_km", "population_exposure",
            "disadvantage_score", "disadvantage_label",
            "population_label",
        ]],
        on="station_name",
        how="left",
    )

    # Get climate vulnerability (default 50 if missing)
    clim = pd.to_numeric(
        result.get("vulnerability_index", 50),
        errors="coerce"
    ).fillna(50)

    pop_exp = pd.to_numeric(
        result.get("population_exposure", 50),
        errors="coerce"
    ).fillna(50)

    # Population Risk Score:
    # 60% climate vulnerability + 40% population exposure
    # Simple, explainable formula
    result["population_risk_score"] = (
        clim * 0.6 + pop_exp * 0.4
    ).round(1)

    # Risk category label
    prs = result["population_risk_score"]
    p75 = prs.quantile(0.75)
    p90 = prs.quantile(0.90)
    result["population_risk_category"] = prs.apply(
        lambda v:
        "Critical"  if v >= p90 else
        "High"      if v >= p75 else
        "Moderate"  if v >= prs.median() else
        "Low"
    )

    n_enriched = result["population"].notna().sum()
    print(f"  ABS: population risk computed for {n_enriched} stations.")
    return result


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("ABS Integration Test")
    print("=" * 55)

    test_coords = {
        "Sydney Airport":    {"lat": -33.946, "lon": 151.177},
        "Melbourne Airport": {"lat": -37.669, "lon": 144.841},
        "Brisbane Airport":  {"lat": -27.384, "lon": 153.117},
        "Alice Springs":     {"lat": -23.796, "lon": 133.889},
        "Darwin Airport":    {"lat": -12.408, "lon": 130.872},
        "Thredbo":           {"lat": -36.500, "lon": 148.300},
    }

    df = fetch_abs_data(test_coords, force_refresh=True)
    cols = ["station_name", "nearest_centre", "distance_km",
            "population", "seifa_score", "population_label",
            "disadvantage_label", "population_exposure"]
    print("\nResult:")
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))