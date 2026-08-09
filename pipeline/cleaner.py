import pandas as pd
from config import COLUMN_MAPPING, NUMERIC_COLS


def clean_bom_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardise raw BOM observation data.
    - Renames columns to consistent names
    - Casts numeric columns
    - Drops rows with no temperature reading
    """
    if df.empty:
        return df

    # Keep only columns that exist in the raw data
    cols_to_keep = {k: v for k, v in COLUMN_MAPPING.items() if k in df.columns}
    df = df[list(cols_to_keep.keys())].copy()
    df.rename(columns=cols_to_keep, inplace=True)

    # Cast numeric columns
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with no temperature
    if "temperature_c" in df.columns:
        df = df.dropna(subset=["temperature_c"])

    # Parse datetime if present
    if "local_date_time_full" in df.columns:
        df["local_date_time_full"] = pd.to_datetime(
            df["local_date_time_full"], format="%Y%m%d%H%M%S", errors="coerce"
        )

    df.reset_index(drop=True, inplace=True)
    print(f"Cleaned data: {len(df)} records, {df.columns.tolist()}")
    return df