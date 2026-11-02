"""
ingestion/aemo.py

AEMO (Australian Energy Market Operator) ingestion module.

The DISPATCHIS ZIP file contains multiple interleaved tables:
- DISPATCHREGIONSUM: demand (TOTALDEMAND) per region
- DISPATCHPRICE:     spot price (RRP) per region

Both are parsed from the same ZIP and joined on REGIONID.

Confirmed schema from AEMO official data model:
https://nemweb.com.au/Reports/Current/MMSDataModelReport/

Base URL confirmed August 2026: nemweb.com.au/Reports/Current/DispatchIS_Reports/
No API key required. CC-BY 4.0.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import zipfile
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from bs4 import BeautifulSoup

from config import (
    AEMO_NEMWEB_BASE,
    AEMO_REGION_TO_STATES,
    AEMO_REGION_LABELS,
    AEMO_CACHE_PATH,
    AEMO_HISTORY_PATH,
    AEMO_CACHE_HOURS,
    AEMO_HISTORY_LIMIT,
)

HEADERS = {
    "User-Agent": "AusClimate/1.0 UTS Capstone Research Project"
}
TIMEOUT = 30


# ── Cache management ──────────────────────────────────────────────────────────

def _cache_is_valid() -> bool:
    """True if cache exists, is younger than AEMO_CACHE_HOURS, and has real data."""
    if not os.path.exists(AEMO_CACHE_PATH):
        return False
    age_h = (
        datetime.now().timestamp() -
        os.path.getmtime(AEMO_CACHE_PATH)
    ) / 3600.0
    if age_h >= AEMO_CACHE_HOURS:
        return False
    try:
        df = pd.read_parquet(AEMO_CACHE_PATH)
        if "data_available" in df.columns:
            return bool(df["data_available"].any())
        return not df.empty
    except Exception:
        return False


# ── Step 1: Find latest dispatch ZIP URL ──────────────────────────────────────

def _get_latest_dispatch_url() -> str:
    """
    Scrape nemweb.com.au/Reports/Current/DispatchIS_Reports/ for
    the most recent dispatch ZIP file.
    Files named PUBLIC_DISPATCHIS_YYYYMMDDHHmm_<seq>.zip
    Max by filename = latest because timestamp is embedded.
    """
    try:
        r = requests.get(AEMO_NEMWEB_BASE, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()

        soup  = BeautifulSoup(r.text, "html.parser")
        links = soup.find_all("a", href=True)

        zips = [
            link["href"] for link in links
            if "PUBLIC_DISPATCHIS" in link["href"].upper()
            and link["href"].upper().endswith(".ZIP")
        ]

        if not zips:
            print("  AEMO: no dispatch ZIPs found in directory listing.")
            return ""

        latest = sorted(zips)[-1]

        if latest.startswith("http"):
            return latest
        elif latest.startswith("/"):
            return "https://www.nemweb.com.au" + latest
        else:
            return AEMO_NEMWEB_BASE + latest

    except Exception as e:
        print(f"  AEMO: directory scrape failed: {e}")
        return ""


# ── Step 2: Parse ZIP extracting BOTH required tables ─────────────────────────

def _parse_table_from_lines(lines: list,
                              section: str,
                              table: str) -> pd.DataFrame:
    """
    Generic parser for one named table from AEMO's C/I/D row format.

    Each table section in the CSV looks like:
    I,DISPATCH,REGIONSUM,5,COL1,COL2,...
    D,DISPATCH,REGIONSUM,5,val1,val2,...
    D,DISPATCH,REGIONSUM,5,val1,val2,...

    section: e.g. "DISPATCH"
    table:   e.g. "REGIONSUM" or "PRICE"
    """
    col_names = None
    rows      = []

    for line in lines:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue

        row_type    = parts[0].upper()
        row_section = parts[1].upper() if len(parts) > 1 else ""
        row_table   = parts[2].upper() if len(parts) > 2 else ""

        if row_section != section.upper():
            continue
        if table.upper() not in row_table:
            continue

        if row_type == "I":
            col_names = [p.upper() for p in parts]
        elif row_type == "D" and col_names is not None:
            while len(parts) < len(col_names):
                parts.append("")
            rows.append(parts[:len(col_names)])

    if not rows or col_names is None:
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=col_names)


def _parse_dispatch_zip(zip_bytes: bytes) -> pd.DataFrame:
    """
    Parse the AEMO dispatch ZIP file.

    Extracts and joins:
    - DISPATCHREGIONSUM table: REGIONID, SETTLEMENTDATE, TOTALDEMAND
    - DISPATCHPRICE table:     REGIONID, SETTLEMENTDATE, RRP

    RRP is confirmed to be in DISPATCHPRICE not DISPATCHREGIONSUM.
    Source: AEMO official data model (nemweb.com.au/Reports/Current/MMSDataModelReport/)
    """
    try:
        z        = zipfile.ZipFile(io.BytesIO(zip_bytes))
        csv_name = z.namelist()[0]
        content  = z.open(csv_name).read().decode("utf-8", errors="ignore")
        lines    = content.splitlines()

        # Parse demand table
        df_demand = _parse_table_from_lines(lines, "DISPATCH", "REGIONSUM")

        # Parse price table
        df_price  = _parse_table_from_lines(lines, "DISPATCH", "PRICE")

        if df_demand.empty:
            print("  AEMO: DISPATCHREGIONSUM table not found in ZIP.")
            return pd.DataFrame()

        print(f"  AEMO: DISPATCHREGIONSUM columns: {list(df_demand.columns)}")

        if not df_price.empty:
            print(f"  AEMO: DISPATCHPRICE columns: {list(df_price.columns)}")

        # Keep only the latest interval per region from demand
        region_col = next(
            (c for c in df_demand.columns if c == "REGIONID"), None
        )
        time_col = next(
            (c for c in df_demand.columns if "SETTLEMENT" in c), None
        )
        demand_col = next(
            (c for c in df_demand.columns
             if c in ("TOTALDEMAND", "CLEAREDSUPPLY")), None
        )

        if region_col is None:
            print("  AEMO: cannot find REGIONID in demand table.")
            return pd.DataFrame()

        keep_demand = [region_col]
        if time_col:
            keep_demand.append(time_col)
        if demand_col:
            keep_demand.append(demand_col)

        df_out = df_demand[keep_demand].copy()
        df_out.rename(columns={
            region_col: "REGIONID",
            time_col:   "SETTLEMENTDATE" if time_col else None,
            demand_col: "TOTALDEMAND"    if demand_col else None,
        }, inplace=True)
        df_out = {k: v for k, v in {
            region_col: "REGIONID",
            time_col:   "SETTLEMENTDATE",
            demand_col: "TOTALDEMAND",
        }.items() if k}
        df_out = df_demand[keep_demand].copy()
        rename_map = {}
        if region_col:
            rename_map[region_col] = "REGIONID"
        if time_col:
            rename_map[time_col] = "SETTLEMENTDATE"
        if demand_col:
            rename_map[demand_col] = "TOTALDEMAND"
        df_out.rename(columns=rename_map, inplace=True)

        # Merge price (RRP) from DISPATCHPRICE table
        if not df_price.empty:
            price_region = next(
                (c for c in df_price.columns if c == "REGIONID"), None
            )
            price_col = next(
                (c for c in df_price.columns if c == "RRP"), None
            )
            price_time = next(
                (c for c in df_price.columns if "SETTLEMENT" in c), None
            )

            if price_col and price_region:
                price_keep = [price_region]
                if price_time:
                    price_keep.append(price_time)
                price_keep.append(price_col)

                df_price_clean = df_price[price_keep].copy()
                price_rename   = {price_region: "REGIONID", price_col: "RRP"}
                if price_time:
                    price_rename[price_time] = "SETTLEMENTDATE"
                df_price_clean.rename(columns=price_rename, inplace=True)

                join_cols = ["REGIONID"]
                if "SETTLEMENTDATE" in df_out.columns and "SETTLEMENTDATE" in df_price_clean.columns:
                    join_cols.append("SETTLEMENTDATE")

                df_out = df_out.merge(
                    df_price_clean,
                    on=join_cols,
                    how="left"
                )
                print(f"  AEMO: RRP joined from DISPATCHPRICE table.")
            else:
                print(f"  AEMO: RRP column not found in DISPATCHPRICE.")

        return df_out

    except Exception as e:
        print(f"  AEMO: ZIP parse failed: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


# ── Step 3: Fetch from nemweb.com.au ─────────────────────────────────────────

def _fetch_via_nemweb() -> pd.DataFrame:
    """Download latest dispatch ZIP and parse both demand and price tables."""
    print("  AEMO: fetching from nemweb.com.au...")

    url = _get_latest_dispatch_url()
    if not url:
        return pd.DataFrame()

    filename = os.path.basename(url.split("?")[0])
    print(f"  AEMO: downloading {filename}...")

    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()

        if len(r.content) < 100:
            print("  AEMO: response too small, not a valid ZIP.")
            return pd.DataFrame()

        return _parse_dispatch_zip(r.content)

    except Exception as e:
        print(f"  AEMO: download failed: {e}")
        return pd.DataFrame()


# ── Step 4: Clean and standardise ────────────────────────────────────────────

def _clean_dispatch_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise the parsed AEMO DataFrame.
    Selects NEM regions, casts types, parses timestamps, adds labels.
    """
    if df.empty:
        return df

    df.columns = df.columns.str.strip().str.upper()

    # Map parsed column names to our standard names
    rename_map = {}

    region_col = next((c for c in df.columns if c == "REGIONID"), None)
    time_col   = next((c for c in df.columns if "SETTLEMENT" in c), None)
    demand_col = next(
        (c for c in df.columns if c in ("TOTALDEMAND", "CLEAREDSUPPLY")), None
    )
    price_col  = next((c for c in df.columns if c == "RRP"), None)

    if region_col:
        rename_map[region_col] = "region"
    if time_col:
        rename_map[time_col] = "settlement_date"
    if demand_col:
        rename_map[demand_col] = "total_demand_mw"
    if price_col:
        rename_map[price_col] = "rrp_per_mwh"

    if not region_col:
        print(f"  AEMO: no REGIONID column. Available: {list(df.columns)}")
        return pd.DataFrame()

    keep = list(rename_map.keys())
    df   = df[keep].copy()
    df.rename(columns=rename_map, inplace=True)

    # Filter to NEM regions
    df = df[df["region"].isin(AEMO_REGION_TO_STATES.keys())]
    if df.empty:
        print("  AEMO: no NEM region rows found.")
        return pd.DataFrame()

    # Cast numeric columns
    for col in ["total_demand_mw", "rrp_per_mwh"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse timestamp
    if "settlement_date" in df.columns:
        df["settlement_date"] = pd.to_datetime(
            df["settlement_date"],
            format="%Y/%m/%d %H:%M:%S",
            errors="coerce"
        )

    # Keep only the latest interval per region
    if "settlement_date" in df.columns and df["settlement_date"].notna().any():
        latest = df["settlement_date"].max()
        df     = df[df["settlement_date"] == latest]

    df = df.drop_duplicates(subset=["region"]).reset_index(drop=True)

    # Add labels and metadata
    df["region_name"]    = df["region"].map(AEMO_REGION_LABELS)
    df["fetch_time"]     = datetime.now()
    df["data_available"] = True

    return df


# ── Placeholder ───────────────────────────────────────────────────────────────

def _build_placeholder() -> pd.DataFrame:
    """Placeholder with NaN values so dashboard still renders when AEMO is down."""
    rows = []
    for region, states in AEMO_REGION_TO_STATES.items():
        rows.append({
            "region":          region,
            "region_name":     AEMO_REGION_LABELS[region],
            "total_demand_mw": np.nan,
            "rrp_per_mwh":     np.nan,
            "settlement_date": pd.NaT,
            "fetch_time":      datetime.now(),
            "data_available":  False,
        })
    return pd.DataFrame(rows)


# ── History ───────────────────────────────────────────────────────────────────

def _append_to_history(df: pd.DataFrame) -> None:
    """Append real readings to rolling history. Skips placeholders."""
    if df.empty:
        return
    if "data_available" in df.columns and not df["data_available"].any():
        return

    os.makedirs(os.path.dirname(AEMO_HISTORY_PATH), exist_ok=True)

    if os.path.exists(AEMO_HISTORY_PATH):
        try:
            existing = pd.read_parquet(AEMO_HISTORY_PATH)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.tail(AEMO_HISTORY_LIMIT)
        except Exception:
            combined = df
    else:
        combined = df

    combined.to_parquet(AEMO_HISTORY_PATH, index=False)


# ── BOM merge ─────────────────────────────────────────────────────────────────

def merge_aemo_with_bom(df_bom: pd.DataFrame,
                         df_aemo: pd.DataFrame) -> pd.DataFrame:
    """
    Join AEMO regional demand/price with BOM station temperature averages.
    Produces the temperature-demand correlation dataset for the Energy tab.
    """
    if df_bom.empty or df_aemo.empty:
        return pd.DataFrame()

    if not {"state", "temperature_c"}.issubset(df_bom.columns):
        return pd.DataFrame()

    df_bom = df_bom.copy()
    df_bom["temperature_c"] = pd.to_numeric(
        df_bom["temperature_c"], errors="coerce"
    )

    state_temps = (
        df_bom.groupby("state")["temperature_c"]
        .agg(avg_temp_c="mean", max_temp_c="max",
             min_temp_c="min", station_count="count")
        .reset_index()
    )

    merged_rows = []
    for _, aemo_row in df_aemo.iterrows():
        region = aemo_row.get("region", "")
        states = AEMO_REGION_TO_STATES.get(region, [])
        row    = aemo_row.to_dict()

        region_temps = state_temps[state_temps["state"].isin(states)]

        if not region_temps.empty:
            row["avg_temp_c"]    = round(float(region_temps["avg_temp_c"].mean()), 1)
            row["max_temp_c"]    = round(float(region_temps["max_temp_c"].max()), 1)
            row["min_temp_c"]    = round(float(region_temps["min_temp_c"].min()), 1)
            row["station_count"] = int(region_temps["station_count"].sum())
        else:
            row["avg_temp_c"]    = np.nan
            row["max_temp_c"]    = np.nan
            row["min_temp_c"]    = np.nan
            row["station_count"] = 0

        row["states"] = ", ".join(states)
        merged_rows.append(row)

    if not merged_rows:
        return pd.DataFrame()

    result = pd.DataFrame(merged_rows)
    for col in ["avg_temp_c", "max_temp_c", "min_temp_c",
                "total_demand_mw", "rrp_per_mwh"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").round(1)

    return result


# ── Main public function ──────────────────────────────────────────────────────

def fetch_aemo_dispatch(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch current AEMO dispatch data for all NEM regions.
    Returns DataFrame with region, total_demand_mw, rrp_per_mwh, settlement_date.
    Falls back to placeholder if AEMO is unreachable.
    """
    if not force_refresh and _cache_is_valid():
        try:
            df = pd.read_parquet(AEMO_CACHE_PATH)
            print(f"  AEMO: loaded from cache (real data).")
            return df
        except Exception:
            pass

    print("  AEMO: fetching current dispatch data...")

    df_raw = _fetch_via_nemweb()
    df     = pd.DataFrame()

    if not df_raw.empty:
        df = _clean_dispatch_data(df_raw)

    if df.empty:
        print("  AEMO: using placeholder.")
        df = _build_placeholder()
    else:
        total = df["total_demand_mw"].sum() \
            if "total_demand_mw" in df.columns else 0
        has_price = "rrp_per_mwh" in df.columns and df["rrp_per_mwh"].notna().any()
        print(f"  AEMO: {len(df)} regions | "
              f"Total NEM demand: {total:,.0f} MW | "
              f"RRP available: {has_price}")

    os.makedirs(os.path.dirname(AEMO_CACHE_PATH), exist_ok=True)
    df.to_parquet(AEMO_CACHE_PATH, index=False)
    _append_to_history(df)

    return df


def get_aemo_history() -> pd.DataFrame:
    """Load rolling AEMO history."""
    if not os.path.exists(AEMO_HISTORY_PATH):
        return pd.DataFrame()
    try:
        return pd.read_parquet(AEMO_HISTORY_PATH)
    except Exception:
        return pd.DataFrame()


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("AEMO Connectivity Test")
    print("=" * 55)

    print("\nStep 1: Finding latest ZIP URL...")
    url = _get_latest_dispatch_url()
    if url:
        print(f"  URL: {url}")
    else:
        print("  No URL found.")

    print("\nStep 2: Fetching and parsing...")
    df = fetch_aemo_dispatch(force_refresh=True)

    print("\nResult:")
    show = [c for c in [
        "region", "region_name", "total_demand_mw",
        "rrp_per_mwh", "settlement_date", "data_available"
    ] if c in df.columns]
    print(df[show].to_string(index=False))