"""
anomaly/detector.py

Isolation Forest anomaly detection for AusClimate.

Phase 1 — Training:
    Trains an Isolation Forest model on cleaned multi-variable
    BOM observation data. Learns what normal looks like across
    all five primary climate variables simultaneously.

Phase 2 — Detection:
    Scores every observation. Returns anomaly scores and binary
    flags. Adds per-observation feature importance via a simple
    deviation-based attribution (SHAP comes in Phase 3).

No labelled data required — fully unsupervised.
Works on BOM-only data. BARRA2 anomaly columns are complementary
(historical context) not a replacement for this ML layer.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────

# Variables used for anomaly detection
# These five together capture compound climate stress
DETECTION_FEATURES = [
    "temperature_c",
    "feels_like_c",
    "humidity_pct",
    "wind_speed_kmh",
    "pressure_hpa",
]

# Contamination: expected proportion of anomalous observations
# 0.05 means we expect ~5% of readings to be anomalous
# Adjust up if too few flags, down if too many
CONTAMINATION = 0.05

# Model cache path
MODEL_CACHE_PATH   = "data/processed/isolation_forest_model.pkl"
SCALER_CACHE_PATH  = "data/processed/isolation_forest_scaler.pkl"
ML_RESULTS_PATH    = "data/processed/ml_anomaly_results.parquet"

# Severity thresholds on the anomaly score
# Isolation Forest returns scores where more negative = more anomalous
SCORE_MODERATE = -0.05   # Moderate anomaly
SCORE_SEVERE   = -0.10   # Severe anomaly


# ── Phase 1: Training ─────────────────────────────────────────────────────────

def train_isolation_forest(df: pd.DataFrame,
                            force_retrain: bool = False
                            ) -> tuple:
    """
    Phase 1: Train Isolation Forest on BOM observation data.

    How it works:
    - Isolation Forest randomly partitions the feature space.
    - Anomalous observations get isolated in fewer partitions
      because they are rare and different from the bulk of data.
    - Normal observations cluster together and require many more
      partitions to separate from their neighbours.
    - No labels needed — the algorithm finds anomalies purely
      from the structure of the data itself.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned BOM DataFrame with all DETECTION_FEATURES present.
    force_retrain : bool
        If True, ignore cached model and retrain from scratch.

    Returns
    -------
    model : IsolationForest
        Trained model.
    scaler : StandardScaler
        Fitted scaler (features are normalised before training).
    X_clean : pd.DataFrame
        The feature matrix used for training (rows with no NaN).
    """

    # Load cached model if available and not forcing retrain
    if (not force_retrain and
            os.path.exists(MODEL_CACHE_PATH) and
            os.path.exists(SCALER_CACHE_PATH)):
        try:
            with open(MODEL_CACHE_PATH, "rb") as f:
                model = pickle.load(f)
            with open(SCALER_CACHE_PATH, "rb") as f:
                scaler = pickle.load(f)
            print("  ML: loaded cached Isolation Forest model.")

            # Still need clean features for scoring
            available = [c for c in DETECTION_FEATURES if c in df.columns]
            X = df[available].copy()
            for col in available:
                X[col] = pd.to_numeric(X[col], errors="coerce")
            X_clean = X.dropna()
            return model, scaler, X_clean
        except Exception as e:
            print(f"  ML: cache load failed ({e}) — retraining.")

    print("  ML: training Isolation Forest...")
    print(f"  Features: {DETECTION_FEATURES}")

    # Select and clean feature columns
    available = [c for c in DETECTION_FEATURES if c in df.columns]
    missing   = [c for c in DETECTION_FEATURES if c not in df.columns]

    if missing:
        print(f"  ML: missing features {missing} — using {available}")

    if len(available) < 2:
        print("  ML: need at least 2 features. Skipping.")
        return None, None, pd.DataFrame()

    X = df[available].copy()
    for col in available:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X_clean = X.dropna()

    if len(X_clean) < 100:
        print(f"  ML: only {len(X_clean)} clean rows — too few to train.")
        return None, None, X_clean

    print(f"  ML: training on {len(X_clean):,} observations, "
          f"{len(available)} features...")

    # Normalise features before training
    # Isolation Forest is sensitive to feature scales
    # StandardScaler puts each feature on mean=0, std=1
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # Train the model
    # n_estimators=100: number of isolation trees (more = more stable)
    # max_samples='auto': subsample size per tree
    # random_state=42: reproducibility
    model = IsolationForest(
        n_estimators=100,
        max_samples="auto",
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1,         # Use all CPU cores
        warm_start=False,
    )
    model.fit(X_scaled)

    # Save model and scaler to cache
    os.makedirs(os.path.dirname(MODEL_CACHE_PATH), exist_ok=True)
    with open(MODEL_CACHE_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_CACHE_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"  ML: training complete. Model saved.")
    print(f"  Contamination: {CONTAMINATION} "
          f"(~{int(CONTAMINATION * len(X_clean)):,} anomalies expected)")

    return model, scaler, X_clean


# ── Phase 2: Detection ────────────────────────────────────────────────────────

def run_detection(df: pd.DataFrame,
                  model: IsolationForest,
                  scaler: StandardScaler) -> pd.DataFrame:
    """
    Phase 2: Score every observation using the trained model.

    Isolation Forest outputs:
    - prediction: 1 = normal, -1 = anomaly
    - score: the raw anomaly score (more negative = more anomalous)

    We also compute a simple per-feature deviation score so the
    dashboard can show which variable drove each anomaly flag.
    This is a lightweight attribution method. Full SHAP comes later.

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned BOM DataFrame.
    model : IsolationForest
        Trained model from Phase 1.
    scaler : StandardScaler
        Fitted scaler from Phase 1.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with ML columns added.
    """
    if model is None or scaler is None:
        print("  ML: no model available — skipping detection.")
        return df

    available = [c for c in DETECTION_FEATURES if c in df.columns]

    result  = df.copy()
    X       = result[available].copy()
    for col in available:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Track which rows have complete data
    complete_mask = X.notna().all(axis=1)
    X_complete    = X[complete_mask]

    # Initialise ML columns with defaults
    result["ml_anomaly_flag"]  = False
    result["ml_anomaly_score"] = np.nan
    result["ml_severity"]      = "Unknown"
    result["ml_primary_driver"] = ""

    if X_complete.empty:
        print("  ML: no complete rows to score.")
        return result

    print(f"  ML: scoring {len(X_complete):,} complete observations...")

    # Scale features using the same scaler used in training
    X_scaled = scaler.transform(X_complete)

    # Get predictions and scores
    # predict: 1=normal, -1=anomaly
    # score_samples: raw anomaly score (more negative = more anomalous)
    predictions    = model.predict(X_scaled)
    anomaly_scores = model.score_samples(X_scaled)

    # Write results back to the rows that had complete data
    result.loc[complete_mask, "ml_anomaly_flag"] = (
        predictions == -1
    )
    result.loc[complete_mask, "ml_anomaly_score"] = (
        anomaly_scores.round(4)
    )

    # Assign severity based on score thresholds
    def _severity(score, flag):
        if not flag:
            return "Normal"
        if score <= SCORE_SEVERE:
            return "Severe"
        if score <= SCORE_MODERATE:
            return "Moderate"
        return "Mild"

    result["ml_severity"] = result.apply(
        lambda row: _severity(
            row["ml_anomaly_score"],
            row["ml_anomaly_flag"]
        ),
        axis=1
    )

    # ── Lightweight feature attribution ──────────────────────────
    # For each anomalous observation, identify which feature
    # deviates most from the training set mean.
    # This is not SHAP — it is a simple deviation ranking.
    # Full SHAP attribution is Phase 3.

    feature_means = {col: X_complete[col].mean() for col in available}
    feature_stds  = {col: max(X_complete[col].std(), 0.001)
                     for col in available}

    anomaly_idx = result[result["ml_anomaly_flag"]].index

    if not anomaly_idx.empty:
        anomaly_rows   = result.loc[anomaly_idx, available].copy()
        for col in available:
            anomaly_rows[col] = pd.to_numeric(
                anomaly_rows[col], errors="coerce"
            )

        # Z-score of each feature relative to training distribution
        deviations = pd.DataFrame(index=anomaly_rows.index)
        for col in available:
            deviations[col] = (
                (anomaly_rows[col] - feature_means[col]).abs()
                / feature_stds[col]
            )

        # Primary driver = feature with highest absolute z-score
        primary_drivers = deviations.idxmax(axis=1)

        # Human-readable labels
        label_map = {
            "temperature_c":  "Temperature",
            "feels_like_c":   "Apparent Temperature",
            "humidity_pct":   "Humidity",
            "wind_speed_kmh": "Wind Speed",
            "pressure_hpa":   "Pressure",
        }
        result.loc[anomaly_idx, "ml_primary_driver"] = (
            primary_drivers.map(label_map)
        )

    # ── Summary stats ─────────────────────────────────────────────
    n_total    = len(X_complete)
    n_anomaly  = int(result["ml_anomaly_flag"].sum())
    n_severe   = len(result[result["ml_severity"] == "Severe"])
    n_moderate = len(result[result["ml_severity"] == "Moderate"])
    n_mild     = len(result[result["ml_severity"] == "Mild"])

    print(f"\n  ML Detection Results:")
    print(f"  Observations scored : {n_total:,}")
    print(f"  Anomalies flagged   : {n_anomaly:,} "
          f"({n_anomaly/n_total*100:.1f}%)")
    print(f"  Severity breakdown  : "
          f"Severe={n_severe}, "
          f"Moderate={n_moderate}, "
          f"Mild={n_mild}")

    if n_anomaly > 0:
        driver_counts = (
            result[result["ml_anomaly_flag"]]["ml_primary_driver"]
            .value_counts()
        )
        print("  Primary drivers:")
        for driver, count in driver_counts.items():
            print(f"    {driver}: {count} anomalies")

    return result


# ── Station-level summary ─────────────────────────────────────────────────────

def compute_station_anomaly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ML anomaly results to station level.
    Returns one row per station with anomaly statistics.
    """
    if "ml_anomaly_flag" not in df.columns:
        return pd.DataFrame()

    agg = (
        df.groupby("station_name")
        .agg(
            total_observations  = ("ml_anomaly_flag", "count"),
            anomaly_count       = ("ml_anomaly_flag", "sum"),
            avg_anomaly_score   = ("ml_anomaly_score", "mean"),
            min_anomaly_score   = ("ml_anomaly_score", "min"),
            severe_count        = ("ml_severity",
                                   lambda x: (x == "Severe").sum()),
            moderate_count      = ("ml_severity",
                                   lambda x: (x == "Moderate").sum()),
            primary_driver      = ("ml_primary_driver",
                                   lambda x: x.mode()[0]
                                   if not x.mode().empty else ""),
            state               = ("state", "first"),
            latitude            = ("latitude", "first"),
            longitude           = ("longitude", "first"),
        )
        .reset_index()
    )

    agg["anomaly_rate_pct"] = (
        agg["anomaly_count"] / agg["total_observations"] * 100
    ).round(1)

    agg["avg_anomaly_score"] = agg["avg_anomaly_score"].round(4)
    agg["min_anomaly_score"] = agg["min_anomaly_score"].round(4)

    # Risk classification per station
    agg["station_risk"] = agg.apply(
        lambda row:
        "High"   if row["severe_count"] > 0 else
        "Medium" if row["moderate_count"] > 2 else
        "Low"    if row["anomaly_count"] > 0 else
        "Normal",
        axis=1
    )

    return agg.sort_values("min_anomaly_score")


# ── Save results ──────────────────────────────────────────────────────────────

def save_ml_results(df: pd.DataFrame) -> None:
    """Save ML-annotated DataFrame to Parquet."""
    os.makedirs(os.path.dirname(ML_RESULTS_PATH), exist_ok=True)
    df.to_parquet(ML_RESULTS_PATH, index=False)
    size_kb = os.path.getsize(ML_RESULTS_PATH) / 1024
    print(f"  ML results saved: {ML_RESULTS_PATH} "
          f"({len(df)} rows, {size_kb:.1f} KB)")


# ── Main public function ──────────────────────────────────────────────────────

def run_ml_pipeline(df: pd.DataFrame,
                    force_retrain: bool = False) -> pd.DataFrame:
    """
    Run the full ML anomaly detection pipeline.

    Phase 1: Train (or load cached) Isolation Forest
    Phase 2: Score all observations, add ML columns

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned and merged BOM DataFrame.
    force_retrain : bool
        Force model retraining even if cache exists.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with ML columns added:
        - ml_anomaly_flag   : bool
        - ml_anomaly_score  : float (more negative = more anomalous)
        - ml_severity       : str ('Normal', 'Mild', 'Moderate', 'Severe')
        - ml_primary_driver : str (feature that drove the anomaly)
    """
    print("\nML Anomaly Detection Pipeline")
    print("-" * 40)

    if df.empty:
        print("  Empty DataFrame — skipping ML.")
        return df

    # Check minimum required features
    available = [c for c in DETECTION_FEATURES if c in df.columns]
    if len(available) < 2:
        print(f"  Only {len(available)} features available. "
              f"Need at least 2. Skipping.")
        return df

    # Phase 1: Train
    model, scaler, X_clean = train_isolation_forest(
        df, force_retrain=force_retrain
    )

    if model is None:
        print("  Phase 1 failed — skipping Phase 2.")
        return df

    # Phase 2: Detect
    df_with_ml = run_detection(df, model, scaler)

    # Save results
    save_ml_results(df_with_ml)

    return df_with_ml


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Test the ML pipeline standalone.
    python anomaly/detector.py
    """
    from pipeline.storage import load_parquet
    from config import MERGED_DATA_PATH

    print("=" * 55)
    print("AusClimate — ML Anomaly Detection Test")
    print("=" * 55)

    df = load_parquet(MERGED_DATA_PATH)
    if df.empty:
        print("No merged data found. Run python main.py first.")
    else:
        df_ml = run_ml_pipeline(df, force_retrain=True)
        print(f"\nSample anomalies:")
        sample = df_ml[df_ml["ml_anomaly_flag"]].head(5)[
            ["station_name", "temperature_c", "humidity_pct",
             "ml_anomaly_score", "ml_severity", "ml_primary_driver"]
        ]
        print(sample.to_string())