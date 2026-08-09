import pandas as pd
import os


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """
    Save a DataFrame to Parquet format.
    Creates the directory if it does not exist.
    """
    if df.empty:
        print(f"  Nothing to save — DataFrame is empty.")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved {len(df)} records to {path} ({size_kb:.1f} KB)")


def load_parquet(path: str) -> pd.DataFrame:
    """
    Load a Parquet file into a DataFrame.
    Returns empty DataFrame if file does not exist.
    """
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df)} records from {path}")
    return df


def append_parquet(new_df: pd.DataFrame, path: str) -> None:
    """
    Append new rows to an existing Parquet file.
    Creates the file if it does not exist.
    """
    if new_df.empty:
        return

    existing = load_parquet(path)

    if existing.empty:
        save_parquet(new_df, path)
    else:
        combined = pd.concat([existing, new_df], ignore_index=True)
        save_parquet(combined, path)