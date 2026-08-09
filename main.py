from ingestion.bom import fetch_all_stations
from pipeline.cleaner import clean_bom_data
from pipeline.feature_eng import compute_station_summary
from pipeline.storage import save_parquet
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH


def run_pipeline():
    print("=" * 50)
    print("AusClimate Pipeline Starting")
    print("=" * 50)

    # Step 1 — Ingest
    print("\n[1/4] Ingesting BOM data...")
    df_raw = fetch_all_stations()

    if df_raw.empty:
        print("No data ingested. Exiting.")
        return

    # Step 2 — Save raw
    print("\n[2/4] Saving raw data...")
    save_parquet(df_raw, RAW_DATA_PATH)

    # Step 3 — Clean
    print("\n[3/4] Cleaning data...")
    df_clean = clean_bom_data(df_raw)

    # Step 4 — Feature engineering and save
    print("\n[4/4] Computing features and saving...")
    df_features = compute_station_summary(df_clean)
    save_parquet(df_clean,    PROCESSED_DATA_PATH)

    print("\n" + "=" * 50)
    print("Pipeline complete.")
    print(f"  Records processed : {len(df_clean)}")
    print(f"  Stations          : {df_clean['station_name'].nunique()}")
    print("=" * 50)


if __name__ == "__main__":
    run_pipeline()