"""
Storage module.
Handles reading and writing Parquet files.
Parquet is 3-5x smaller than CSV and reads faster for column queries.
"""

import pandas as pd
import os
import tempfile


def save_parquet(df: pd.DataFrame, path: str) -> None:
    if df.empty:
        print(f"  Nothing to save at {path} -- DataFrame is empty.")
        return
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    # Write to a temp file in the same folder, then rename it over the
    # real file. The rename is atomic, so a reader never sees a half
    # written file, and a crash mid write never corrupts the real one.
    fd, tmp_path = tempfile.mkstemp(suffix=".parquet", dir=folder)
    os.close(fd)
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved: {path} ({len(df)} rows, {size_kb:.1f} KB)")


def load_parquet(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    print(f"  Loaded: {path} ({len(df)} rows)")
    return df


def append_parquet(new_df: pd.DataFrame, path: str,
                   max_rows: int = 50000) -> None:
    """
    Append new rows to existing Parquet file.
    Caps at max_rows to prevent files growing unbounded.
    """
    if new_df.empty:
        return
    existing = load_parquet(path)
    if existing.empty:
        save_parquet(new_df, path)
    else:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.tail(max_rows)
        save_parquet(combined, path)


def append_to_history(latest_df: pd.DataFrame, history_path: str):
    """Append the latest BOM snapshot to a rolling 90-day history file."""
    if latest_df.empty:
        return

    if os.path.exists(history_path):
        try:
            existing = pd.read_parquet(history_path)
            combined = pd.concat([existing, latest_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["station_name", "timestamp"])
        except Exception:
            combined = latest_df
    else:
        combined = latest_df

    combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
    combined = combined[combined["timestamp"] >= cutoff]

    save_parquet(combined, history_path)