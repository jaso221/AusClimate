"""
Cleaning module.
Standardises raw BOM data into a consistent schema.
Handles the BARRA2-augmented data too.
"""

import pandas as pd
from config import COLUMN_MAPPING, NUMERIC_COLS


def clean_bom_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw BOM observation DataFrame.
    - Rename columns to clean names
    - Cast numeric columns
    - Parse timestamps
    - Drop rows missing temperature
    """
    if df.empty:
        return df

    # Keep only columns we have mappings for, plus state
    cols_available = {
        k: v for k, v in COLUMN_MAPPING.items()
        if k in df.columns
    }

    df = df[list(cols_available.keys())].copy()
    df.rename(columns=cols_available, inplace=True)

    # Cast numeric columns
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], format="%Y%m%d%H%M%S", errors="coerce"
        )

    # Drop rows with no temperature
    if "temperature_c" in df.columns:
        before = len(df)
        df     = df.dropna(subset=["temperature_c"])
        if len(df) < before:
            print(f"  Cleaner: dropped {before - len(df)} "
                  f"rows with no temperature.")

    df = df.reset_index(drop=True)
    print(f"Cleaned: {len(df)} records, {len(df.columns)} columns.")
    return df