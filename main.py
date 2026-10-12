"""
AusClimate main pipeline — main.py

Run order:
1.  BOM live ingestion — 7 parallel state feeds, all of Australia
2.  BARRA2 historical baseline — full Australian grid, vectorized extraction
3.  ERA5 supplementary — cached by date, skipped if not configured
4.  Merge — BOM + BARRA2 + ERA5 into unified dataset
4b. ML Anomaly Detection — Isolation Forest Phase 1 and Phase 2
5.  Feature engineering — station summaries, stress scores, vulnerability index

Usage:
    python main.py              normal run, loads caches where valid
    python main.py --refresh    force refresh all caches
    python main.py --retrain    force retrain ML model
    python main.py --refresh --retrain   refresh everything
"""

import sys
import os

from ingestion.bom import fetch_all_stations, get_station_coords
from ingestion.barra2 import fetch_barra2_all_stations
from pipeline.cleaner import clean_bom_data
from pipeline.feature_eng import compute_station_summary
from pipeline.storage import save_parquet, load_parquet, append_to_history
from pipeline.spatial_join import merge_bom_barra2
from config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    BARRA2_CACHE_PATH,
    MERGED_DATA_PATH,
    STATION_SUMMARY_PATH,
    STATION_LIST_PATH,
    BOM_HISTORY_PATH,
)

import pandas as pd


def run_pipeline(force_refresh: bool = False,
                 force_retrain: bool = False) -> None:

    print("=" * 60)
    print("  AusClimate Data Pipeline")
    print("=" * 60)

    # ── Step 1: BOM live ingestion ─────────────────────────────────
    # 7 parallel state feeds covering all of Australia
    # Scrapes BOM observation index pages to discover all stations
    # then fetches each station's JSON in parallel
    # Typical: 700+ stations, completes in 10-30 seconds
    print("\n[1/6] BOM Live Ingestion")
    print("-" * 40)

    df_raw = fetch_all_stations(
        force_station_refresh=force_refresh
    )

    if df_raw is None or df_raw.empty:
        print("Pipeline stopped: no BOM data fetched.")
        print("Check your network connection and try again.")
        return

    save_parquet(df_raw, RAW_DATA_PATH)

    df_bom = clean_bom_data(df_raw)
    if df_bom.empty:
        print("Pipeline stopped: cleaning produced empty DataFrame.")
        return

    save_parquet(df_bom, PROCESSED_DATA_PATH)
    
    append_to_history(df_bom, BOM_HISTORY_PATH)

    # Extract station coordinates for supplementary data fetches
    station_coords = get_station_coords(df_raw)

    # Save station list as a lightweight reference file
    station_list = pd.DataFrame([
        {
            "station_name": k,
            "lat":          v["lat"],
            "lon":          v["lon"],
        }
        for k, v in station_coords.items()
    ])
    save_parquet(station_list, STATION_LIST_PATH)
    
    

    # ── Step 2: BARRA2 historical baseline ────────────────────────
    # Downloads full Australian climate grid once per variable
    # Vectorized xarray extraction for all stations simultaneously
    # Cached for 24 hours — historical context does not change hourly
    # Cache is validated for actual non-NaN data before being used
    print("\n[2/6] BARRA2 Historical Baseline")
    print("-" * 40)

    df_barra2 = fetch_barra2_all_stations(
        station_coords=station_coords,
        force_refresh=force_refresh,
    )


    # ── Step 4: Merge all data sources ────────────────────────────
    # BOM: current live conditions
    # BARRA2: historical baseline + anomaly computation
    #         (temp_anomaly_c, humidity_anomaly_pct, etc.)
    # ERA5: supplementary global context where available
    print("\n[4/6] Merging Data Sources")
    print("-" * 40)

    df_merged = merge_bom_barra2(df_bom, df_barra2)
    save_parquet(df_merged, MERGED_DATA_PATH)

    # ── Step 4b: ML Anomaly Detection ─────────────────────────────
    # Phase 1: Train (or load cached) Isolation Forest model
    #          across 5 climate variables simultaneously
    # Phase 2: Score every observation, assign severity,
    #          identify primary driver variable per anomaly
    # Phase 3 (future): SHAP attribution
    print("\n[4b/6] ML Anomaly Detection — Isolation Forest")
    print("-" * 40)

    try:
        from anomaly.detector import run_ml_pipeline
        df_merged = run_ml_pipeline(
            df_merged,
            force_retrain=force_retrain,
            force_shap=force_refresh, 
        )
        # Save again with ML columns added
        save_parquet(df_merged, MERGED_DATA_PATH)

    except ImportError:
        print("  scikit-learn or shap not installed.")
        print("  Run: pip install scikit-learn shap")
        print("  Continuing without ML results.")
    except Exception as e:
        print(f"  ML pipeline error: {e}")
        print("  Continuing without ML results.")
    
        # ── Step 4c: AEMO energy demand ────────────────────────────────
    # Fetches 5-minute electricity demand and spot price for
    # the five NEM regions, maps them to BOM states, and
    # computes temperature-demand correlation per region.
    print("\n[4c/6] AEMO Energy Demand")
    print("-" * 40)

    try:
        from ingestion.aemo import fetch_aemo_dispatch, merge_aemo_with_bom
        from pipeline.storage import save_parquet as sp

        df_aemo = fetch_aemo_dispatch()

        if not df_aemo.empty:
            sp(df_aemo, "data/processed/aemo_latest.parquet")

            # Merge AEMO with BOM temperature data
            df_aemo_bom = merge_aemo_with_bom(df_bom, df_aemo)
            if not df_aemo_bom.empty:
                sp(df_aemo_bom, "data/processed/aemo_bom_merged.parquet")
                print(f"  AEMO-BOM merge: {len(df_aemo_bom)} regions")

    except Exception as e:
        print(f"  AEMO error: {e}")
        print("  Continuing without AEMO data.")

    # ── Step 5: Feature engineering ───────────────────────────────
    # Computes per-station aggregate features:
    #   mean/max/min temperature, temperature range
    #   heat stress score, vulnerability index
    #   BARRA2 baseline columns and anomaly columns (if available)
    #   ML anomaly summary columns (if available)
    
    # ── Step 5: Feature engineering ───────────────────────────────
    print("\n[5/6] Feature Engineering")
    print("-" * 40)

    df_summary = compute_station_summary(df_merged)

    # ── Step 5b: K-Means clustering ────────────────────────────────
    print("\n[5b/6] K-Means Station Clustering")
    print("-" * 40)

    try:
        from anomaly.detector import run_kmeans_clustering
        df_summary = run_kmeans_clustering(
            df_summary,
            n_clusters=6,
            force_retrain=force_retrain,
        )
    except Exception as e:
        print(f"  K-Means error: {e}")
        print("  Continuing without cluster assignments.")
    
   
    # ── Step 5c: ABS Population Data ──────────────────────────────
    print("\n[5c/6] ABS Population and Disadvantage Data")
    print("-" * 40)

    try:
        from ingestion.abs_data import (
            fetch_abs_data,
            compute_population_risk,
        )

        # Build station coordinates from station list
        station_coords = {}
        if os.path.exists(STATION_LIST_PATH):
            stn_list = pd.read_parquet(STATION_LIST_PATH)
            lat_col  = "lat"      if "lat"      in stn_list.columns \
                       else "latitude"
            lon_col  = "lon"      if "lon"      in stn_list.columns \
                       else "longitude"
            for _, row in stn_list.iterrows():
                try:
                    station_coords[row["station_name"]] = {
                        "lat": float(row[lat_col]),
                        "lon": float(row[lon_col]),
                    }
                except Exception:
                    pass

        if station_coords:
            df_abs = fetch_abs_data(
                station_coords,
                force_refresh=force_refresh,
            )
            if not df_abs.empty:
                # Add population risk columns to the station summary
                df_summary = compute_population_risk(df_summary, df_abs)
                save_parquet(df_summary, STATION_SUMMARY_PATH)

                critical = int(
                    (df_summary.get(
                        "population_risk_category", pd.Series()
                    ) == "Critical").sum()
                ) if "population_risk_category" in df_summary.columns \
                    else 0
                print(f"  ABS: {critical} stations at critical population risk")
        else:
            print("  ABS: no station coordinates found.")

    except Exception as e:
        print(f"  ABS error: {e}")
        print("  Continuing without ABS data.")
        
    save_parquet(df_summary, STATION_SUMMARY_PATH)
    
    print("\n[6/6] Forecasting: Temperature, Humidity, Wind Speed")
    print("-" * 40)
    try:
        from forecasting.forecaster import run_forecasting, FORECAST_CACHE_HOURS
        from config import FORECAST_OUTPUT_PATH
        from datetime import datetime

        should_run = True
        if os.path.exists(FORECAST_OUTPUT_PATH) and not force_refresh:
            age_hours = (datetime.now() - datetime.fromtimestamp(
                os.path.getmtime(FORECAST_OUTPUT_PATH)
            )).total_seconds() / 3600
            if age_hours < FORECAST_CACHE_HOURS:
                should_run = False
                print(f"  Forecast cache is {age_hours:.1f}h old, skipping (refreshes every {FORECAST_CACHE_HOURS}h)")

        if should_run:
            df_forecast = run_forecasting()
            if not df_forecast.empty:
                print(f"  Forecasted {df_forecast['station_name'].nunique()} stations")
            else:
                print("  No forecasts produced yet -- needs more accumulated BOM history.")
    except Exception as e:
        print(f"  Forecasting error: {e}")
        print("  Continuing without forecast results.")   

    # ── Final summary ──────────────────────────────────────────────
    barra2_with_data = (
        df_barra2["barra2_temp_c"].notna().sum()
        if not df_barra2.empty and "barra2_temp_c" in df_barra2.columns
        else 0
    )
    merged_with_barra2 = (
        df_merged["barra2_temp_c"].notna().sum()
        if "barra2_temp_c" in df_merged.columns
        else 0
    )
    ml_flags = (
        int(df_merged["ml_anomaly_flag"].sum())
        if "ml_anomaly_flag" in df_merged.columns else 0
    )
    ml_severe = (
        int((df_merged["ml_severity"] == "Severe").sum())
        if "ml_severity" in df_merged.columns else 0
    )
    lof_flags = (
        int(df_merged["lof_anomaly_flag"].sum())
        if "lof_anomaly_flag" in df_merged.columns else 0
    )
    ensemble_flags = (
        int(df_merged["ensemble_anomaly"].sum())
        if "ensemble_anomaly" in df_merged.columns else 0
    )
    anomaly_cols = [
        c for c in df_merged.columns if "anomaly" in c.lower()
    ]
    barra2_cols = [
        c for c in df_merged.columns if c.startswith("barra2_")
    ]

    print("\n" + "=" * 60)
    print("  Pipeline Complete")
    print("=" * 60)
    print(f"  BOM stations         : "
          f"{df_bom['station_name'].nunique()}")
    print(f"  BOM records          : {len(df_bom):,}")
    print(f"  States covered       : "
          f"{df_bom['state'].nunique() if 'state' in df_bom.columns else 'N/A'}")
    print(f"  BARRA2 stations      : {len(df_barra2)}")
    print(f"  BARRA2 with data     : {barra2_with_data}")
    print(f"  Merged records       : {len(df_merged):,}")
    print(f"  Merged with BARRA2   : {merged_with_barra2:,}")
    print(f"  BARRA2 columns       : {len(barra2_cols)}")
    print(f"  Anomaly columns      : {len(anomaly_cols)}")
    print(f"  ML anomalies flagged : {ml_flags:,}")
    print(f"  ML severe flags      : {ml_severe}")
    print(f"  LOF anomalies flagged: {lof_flags:,}")   
    print(f"  Ensemble (both agree): {ensemble_flags:,}")
    print(f"  Station summaries    : {len(df_summary)}")
    n_clusters_found = df_summary["cluster_id"].nunique() \
        if "cluster_id" in df_summary.columns else 0
    print(f"  Climate clusters     : {n_clusters_found}")
    print("=" * 60)

    # Warn if nothing useful came through
    if merged_with_barra2 == 0:
        print("\n⚠️  BARRA2 context is missing from merged data.")
        print("  Run: del data\\processed\\barra2_baseline.parquet")
        print("  Then: python main.py")

    if ml_flags == 0 and "ml_anomaly_flag" in df_merged.columns:
        print("\n⚠️  ML detection ran but found zero anomalies.")
        print("  This is unusual. Try: python main.py --retrain")


if __name__ == "__main__":
    force_refresh = "--refresh" in sys.argv
    force_retrain = "--retrain" in sys.argv

    if force_refresh:
        print("Force refresh mode — all caches will be ignored.")
    if force_retrain:
        print("Force retrain mode — ML model will be rebuilt.")

    run_pipeline(
        force_refresh=force_refresh,
        force_retrain=force_retrain,
    )
    aemo_demand = 0
    aemo_path   = "data/processed/aemo_latest.parquet"
    if os.path.exists(aemo_path):
        try:
            df_a = pd.read_parquet(aemo_path)
            if "total_demand_mw" in df_a.columns:
                aemo_demand = df_a["total_demand_mw"].sum()
        except Exception:
            pass
    print(f"  AEMO total NEM demand : {aemo_demand:,.0f} MW")
    
    abs_matched = 0
    abs_path    = "data/processed/abs_station_matched.parquet"
    if os.path.exists(abs_path):
        try:
            df_a = pd.read_parquet(abs_path)
            abs_matched = df_a["matched"].sum() \
                if "matched" in df_a.columns else len(df_a)
        except Exception:
            pass
    print(f"  ABS stations matched  : {abs_matched}")