# ── AusClimate Configuration ──────────────────────────────────────────────────
# Central config for all data sources, paths, and settings.
# All other modules import from here.

from datetime import datetime

# ── BOM Configuration ─────────────────────────────────────────────────────────
# BOM state-level JSON feeds.
# Each URL returns ALL active stations in that state simultaneously.
# 7 requests covers the entire Australian observation network.
# No API key required. User-Agent header required (see ingestion/bom.py).

BOM_STATE_INDEX_PAGES = {
    "NSW": "http://www.bom.gov.au/nsw/observations/nswall.shtml",
    "VIC": "http://www.bom.gov.au/vic/observations/vicall.shtml",
    "QLD": "http://www.bom.gov.au/qld/observations/qldall.shtml",
    "SA":  "http://www.bom.gov.au/sa/observations/saall.shtml",
    "WA":  "http://www.bom.gov.au/wa/observations/waall.shtml",
    "NT":  "http://www.bom.gov.au/nt/observations/ntall.shtml",
    "TAS": "http://www.bom.gov.au/tas/observations/tasall.shtml",
    "ACT": "http://www.bom.gov.au/act/observations/canberra.shtml",
}

BOM_JSON_BASE        = "http://www.bom.gov.au/fwo"
BOM_MAX_WORKERS      = 20
BOM_TIMEOUT          = 10
BOM_STATION_CACHE_HOURS = 6

# BOM column mapping: raw BOM names -> clean AusClimate names
COLUMN_MAPPING = {
    "name":                  "station_name",
    "air_temp":              "temperature_c",
    "apparent_t":            "feels_like_c",
    "rel_hum":               "humidity_pct",
    "wind_spd_kmh":          "wind_speed_kmh",
    "gust_kmh":              "gust_kmh",
    "press_msl":             "pressure_hpa",
    "wind_dir":              "wind_direction",
    "rain_trace":            "rainfall_mm",
    "local_date_time_full":  "timestamp",
    "lat":                   "latitude",
    "lon":                   "longitude",
    "cloud":                 "cloud_cover",
    "vis_km":                "visibility_km",
    "dewpt":                 "dewpoint_c",
    "wmo":                   "wmo_id",
    "state":                 "state",
}

# Columns to cast to numeric after renaming
NUMERIC_COLS = [
    "temperature_c", "feels_like_c", "humidity_pct",
    "wind_speed_kmh", "gust_kmh", "pressure_hpa",
    "rainfall_mm", "latitude", "longitude",
    "visibility_km", "dewpoint_c",
]

# BOM filtering settings
BOM_REQUIRE_TEMPERATURE = True   # Drop rows with no temperature reading
BOM_MIN_LAT             = -45.0  # Australia south boundary
BOM_MAX_LAT             = -10.0  # Australia north boundary
BOM_MIN_LON             = 112.0  # Australia west boundary
BOM_MAX_LON             = 154.0  # Australia east boundary


# ── BARRA2 Configuration ──────────────────────────────────────────────────────
# BOM BARRA-R2 atmospheric reanalysis via NCI THREDDS.
# - Coverage: all of Australia, 1979 to near-present
# - Resolution: 12 km horizontal (0.11 degrees)
# - Access: NCI THREDDS NetCDF Subset Service (NCSS)
# - Performance: fetch full Australian grid once, extract all stations
#   using vectorized xarray selection — fast regardless of station count

BARRA2_BASE = (
    "https://thredds.nci.org.au/thredds/ncss/grid/ob53/output/reanalysis/"
    "AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1"
)

# BARRA2 variable names (NetCDF) -> clean names and units
# All conversions applied in ingestion/barra2.py
BARRA2_VARIABLES = {
    "tas":     {
        "clean": "barra2_temp_c",
        "desc":  "Mean temperature",
        "unit":  "°C",
        "convert": "kelvin_to_celsius"
    },
    "tasmax":  {
        "clean": "barra2_temp_max_c",
        "desc":  "Maximum temperature",
        "unit":  "°C",
        "convert": "kelvin_to_celsius"
    },
    "tasmin":  {
        "clean": "barra2_temp_min_c",
        "desc":  "Minimum temperature",
        "unit":  "°C",
        "convert": "kelvin_to_celsius"
    },
    "hurs":    {
        "clean": "barra2_humidity_pct",
        "desc":  "Relative humidity",
        "unit":  "%",
        "convert": None
    },
    "sfcWind": {
        "clean": "barra2_wind_kmh",
        "desc":  "Surface wind speed",
        "unit":  "km/h",
        "convert": "ms_to_kmh"
    },
    "pr":      {
        "clean": "barra2_precip_mm",
        "desc":  "Precipitation",
        "unit":  "mm",
        "convert": "kgm2s_to_mm"
    },
    "ps":      {
        "clean": "barra2_pressure_hpa",
        "desc":  "Surface pressure",
        "unit":  "hPa",
        "convert": "pa_to_hpa"
    },
}

# Australia bounding box for BARRA2 grid queries
AUSTRALIA_BOUNDS = {
    "north": -10.0,
    "south": -45.0,
    "east":  154.0,
    "west":  112.0,
}

# How many hours old BARRA2 cache can be before re-fetching
# Set to 24 because BARRA2 is historical context, not real-time
BARRA2_CACHE_HOURS = 24


# ── ERA5 Configuration ────────────────────────────────────────────────────────
# ECMWF ERA5 global reanalysis — supplementary fallback.
# Requires ~/.cdsapirc with valid Copernicus CDS API key.
# Coarser than BARRA2 (31km) but global coverage.
# Cached by date to avoid repeated slow CDS queue requests.

ERA5_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "total_precipitation",
]

ERA5_CACHE_HOURS = 24  # Re-fetch if cache older than this


# ── Storage Paths ─────────────────────────────────────────────────────────────

RAW_DATA_PATH          = "data/raw/bom_raw.parquet"
PROCESSED_DATA_PATH    = "data/processed/bom_latest.parquet"
BARRA2_CACHE_PATH      = "data/processed/barra2_baseline.parquet"
ERA5_CACHE_PATH        = "data/processed/era5_latest.parquet"
MERGED_DATA_PATH       = "data/processed/merged_latest.parquet"
STATION_SUMMARY_PATH   = "data/processed/station_summary.parquet"
STATION_LIST_PATH      = "data/processed/station_list.parquet"

# ── AEMO Configuration ────────────────────────────────────────────────────────
# Australian Energy Market Operator public dispatch data
# No API key required. Data updated every 5 minutes.
# Coverage: NEM regions only (WA and NT are separate markets)
# Licence: Creative Commons Attribution 4.0

# Old -- decommissioned April 2026
# AEMO_NEMWEB_BASE = "https://www.aemo.com.au/-/media/files/electricity/nem/data/nemweb/reports/current/dispatchis/"

# New -- confirmed working August 2026
AEMO_NEMWEB_BASE = "https://www.nemweb.com.au/Reports/Current/DispatchIS_Reports/"

# Fallback visualization API -- also updated
AEMO_VIS_API = (
    "https://visualisations.aemo.com.au/aemo/nemweb/index.html"
)

# NEM region to BOM state mapping
# WA and NT are not on the NEM
AEMO_REGION_TO_STATES = {
    "NSW1": ["NSW", "ACT"],
    "VIC1": ["VIC"],
    "QLD1": ["QLD"],
    "SA1":  ["SA"],
    "TAS1": ["TAS"],
}

# Human readable region names
AEMO_REGION_LABELS = {
    "NSW1": "New South Wales",
    "VIC1": "Victoria",
    "QLD1": "Queensland",
    "SA1":  "South Australia",
    "TAS1": "Tasmania",
}

# Storage
AEMO_CACHE_PATH    = "data/processed/aemo_latest.parquet"
AEMO_HISTORY_PATH  = "data/processed/aemo_history.parquet"
AEMO_CACHE_HOURS   = 0.25  # 15 minutes: AEMO updates every 5 min
AEMO_HISTORY_LIMIT = 2016  # Keep last 7 days of 5-min intervals