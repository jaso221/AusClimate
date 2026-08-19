"""
Storage module.
Handles reading and writing Parquet files.
Parquet is 3-5x smaller than CSV and reads faster for column queries.
"""

import pandas as pd
import os


def save_parquet(df: pd.DataFrame, path: str) -> None:
    if df.empty:
        print(f"  Nothing to save at {path} — DataFrame is empty.")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
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
        # Keep only the most recent max_rows records
        combined = combined.tail(max_rows)
        save_parquet(combined, path)