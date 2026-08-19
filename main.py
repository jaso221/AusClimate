"""
AusClimate main pipeline.

Run order:
1. BOM live ingestion — 7 parallel requests, all of Australia (~5 sec)
2. BARRA2 historical baseline — single grid download per variable,
   vectorized extraction, cached for 24 hours
3. ERA5 supplementary — cached by date, skipped if not configured
4. Merge — BOM + BARRA2 + ERA5 into unified dataset
5. Feature engineering — station summaries and anomaly scores
6. Save all outputs to Parquet

Usage:
    python main.py            # Normal run
    python main.py --refresh  # Force refresh all caches
"""

import sys
from ingestion.bom import fetch_all_stations, get_station_coords
from ingestion.barra2 import fetch_barra2_all_stations
from ingestion.era5 import fetch_era5_for_stations
from pipeline.cleaner import clean_bom_data
from pipeline.feature_eng import compute_station_summary
from pipeline.storage import save_parquet, load_parquet
from pipeline.spatial_join import merge_bom_barra2, merge_era5
from config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH,
    MERGED_DATA_PATH, STATION_SUMMARY_PATH,
    STATION_LIST_PATH
)


def run_pipeline(force_refresh: bool = False) -> None:
    print("=" * 60)
    print("  AusClimate Data Pipeline")
    print("=" * 60)

    # ── Step 1: BOM live observations ─────────────────────────────
    # 7 parallel requests, covers all of Australia
    # Typical: 500-700 stations, completes in 3-6 seconds
    print("\n[1/5] BOM Live Ingestion")
    print("-" * 40)
    df_raw = fetch_all_stations()

    if df_raw.empty:
        print("Pipeline stopped: no BOM data fetched.")
        return

    save_parquet(df_raw, RAW_DATA_PATH)

    df_bom = clean_bom_data(df_raw)
    save_parquet(df_bom, PROCESSED_DATA_PATH)

    # Extract station coordinates for supplementary data fetches
    station_coords = get_station_coords(df_raw)

    # Save station list for reference
    import pandas as pd
    station_list = pd.DataFrame([
        {"station_name": k, "lat": v["lat"], "lon": v["lon"]}
        for k, v in station_coords.items()
    ])
    save_parquet(station_list, STATION_LIST_PATH)

    # ── Step 2: BARRA2 historical baseline ────────────────────────
    # Downloads full Australian grid once per variable
    # Vectorized extraction for all stations simultaneously
    # Cached for 24 hours — historical context does not change hourly
    print("\n[2/5] BARRA2 Historical Baseline")
    print("-" * 40)
    df_barra2 = fetch_barra2_all_stations(
        station_coords=station_coords,
        force_refresh=force_refresh
    )

    # ── Step 3: ERA5 supplementary ────────────────────────────────
    # Only runs if .cdsapirc is configured
    # Cached by date — skips if today's data already exists
    print("\n[3/5] ERA5 Supplementary Data")
    print("-" * 40)
    df_era5 = fetch_era5_for_stations(
        station_coords=station_coords,
        force_refresh=force_refresh
    )

    # ── Step 4: Merge all sources ─────────────────────────────────
    # BOM: current conditions
    # BARRA2: historical baseline + anomaly computation
    # ERA5: supplementary context where available
    print("\n[4/5] Merging Data Sources")
    print("-" * 40)
    df_merged = merge_bom_barra2(df_bom, df_barra2)
    df_merged = merge_era5(df_merged, df_era5)
    save_parquet(df_merged, MERGED_DATA_PATH)

    # ── Step 5: Feature engineering ───────────────────────────────
    # Computes station summaries, heat stress scores,
    # vulnerability index, and anomaly columns
    print("\n[5/5] Feature Engineering")
    print("-" * 40)
    df_summary = compute_station_summary(df_merged)
    save_parquet(df_summary, STATION_SUMMARY_PATH)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Pipeline Complete")
    print("=" * 60)
    print(f"  BOM stations     : {df_bom['station_name'].nunique()}")
    print(f"  BOM records      : {len(df_bom)}")
    print(f"  States covered   : {df_bom['state'].nunique() if 'state' in df_bom.columns else 'N/A'}")
    print(f"  BARRA2 stations  : {len(df_barra2)}")
    print(f"  ERA5 stations    : {len(df_era5)}")
    print(f"  Merged records   : {len(df_merged)}")
    print(f"  Station summaries: {len(df_summary)}")
    barra2_cols = [c for c in df_merged.columns if c.startswith("barra2_")]
    print(f"  BARRA2 columns   : {len(barra2_cols)}")
    anomaly_cols = [c for c in df_merged.columns if "anomaly" in c]
    print(f"  Anomaly columns  : {len(anomaly_cols)}")
    print("=" * 60)


if __name__ == "__main__":
    force = "--refresh" in sys.argv
    if force:
        print("Force refresh mode — ignoring all caches.")
    run_pipeline(force_refresh=force)