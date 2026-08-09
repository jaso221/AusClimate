# ── AusClimate Configuration ─────────────────────────────────────────────────

# BOM Station URLs
# Each station has a name and its live JSON feed URL
BOM_STATIONS = {
    "Sydney (Observatory Hill)": "http://www.bom.gov.au/fwo/IDN60901/IDN60901.94768.json",
    "Melbourne (Airport)":       "http://www.bom.gov.au/fwo/IDV60801/IDV60801.94866.json",
    "Brisbane (Airport)":        "http://www.bom.gov.au/fwo/IDQ60801/IDQ60801.94578.json",
    "Adelaide (Airport)":        "http://www.bom.gov.au/fwo/IDS60801/IDS60801.94672.json",
    "Perth (Airport)":           "http://www.bom.gov.au/fwo/IDW60801/IDW60801.94610.json",
    "Darwin (Airport)":          "http://www.bom.gov.au/fwo/IDD60801/IDD60801.94120.json",
    "Hobart (Airport)":          "http://www.bom.gov.au/fwo/IDT60801/IDT60801.94970.json",
    "Canberra (Airport)":        "http://www.bom.gov.au/fwo/IDN60903/IDN60903.94926.json",
    "Cairns (Airport)":          "http://www.bom.gov.au/fwo/IDQ60801/IDQ60801.94287.json",
    "Townsville (Airport)":      "http://www.bom.gov.au/fwo/IDQ60801/IDQ60801.94294.json",
    "Alice Springs (Airport)":   "http://www.bom.gov.au/fwo/IDA60801/IDA60801.94430.json",
    "Broken Hill (Airport)":     "http://www.bom.gov.au/fwo/IDN60901/IDN60901.95703.json",
    "Wagga Wagga (Airport)":     "http://www.bom.gov.au/fwo/IDN60901/IDN60901.72150.json",
    "Newcastle (Williamtown)":   "http://www.bom.gov.au/fwo/IDN60901/IDN60901.94776.json",
    "Gold Coast (Airport)":      "http://www.bom.gov.au/fwo/IDQ60801/IDQ60801.94595.json",
    "Launceston (Airport)":      "http://www.bom.gov.au/fwo/IDT60801/IDT60801.94968.json",
    "Broome (Airport)":          "http://www.bom.gov.au/fwo/IDW60801/IDW60801.94403.json",
}

# Column mapping from BOM raw names to clean names
COLUMN_MAPPING = {
    "air_temp":          "temperature_c",
    "apparent_t":        "feels_like_c",
    "rel_hum":           "humidity_pct",
    "wind_spd_kmh":      "wind_speed_kmh",
    "gust_kmh":          "gust_kmh",
    "press_msl":         "pressure_hpa",
    "wind_dir":          "wind_direction",
    "rain_trace":        "rainfall_mm",
    "local_date_time_full": "local_date_time_full",
    "name":              "station_name",
    "lat":               "latitude",
    "lon":               "longitude",
    "cloud":             "cloud_cover",
    "vis_km":            "visibility_km",
    "dewpt":             "dewpoint_c",
}

# Numeric columns to cast
NUMERIC_COLS = [
    "temperature_c",
    "feels_like_c",
    "humidity_pct",
    "wind_speed_kmh",
    "gust_kmh",
    "pressure_hpa",
    "rainfall_mm",
    "latitude",
    "longitude",
    "visibility_km",
    "dewpoint_c",
]

# Storage paths
RAW_DATA_PATH       = "data/raw/bom_raw.parquet"
PROCESSED_DATA_PATH = "data/processed/bom_latest.parquet"