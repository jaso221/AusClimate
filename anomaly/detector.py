"""
anomaly/detector.py

ML anomaly detection pipeline for AusClimate.

Phase 1 -- Training:
    Trains an Isolation Forest model on multi-variable BOM data.
    Learns what normal looks like across 5 climate variables.

Phase 2 -- Detection:
    Scores every observation. Returns anomaly flags, severity levels,
    and a lightweight primary driver attribution.

Phase 3 -- SHAP Explainability:
    Computes precise Shapley values for every anomalous observation.
    Each anomaly gets per-feature attribution scores and a plain
    English explanation showing which variables drove the flag and by
    how much. SHAP values are mathematically additive: they sum to
    the total anomaly score.

No labelled data required. Fully unsupervised.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ── Constants ─────────────────────────────────────────────────────────────────

DETECTION_FEATURES = [
    "temperature_c",
    "feels_like_c",
    "humidity_pct",
    "wind_speed_kmh",
    "pressure_hpa",
]

# Human-readable labels for dashboard display
FEATURE_LABELS = {
    "temperature_c":  "Temperature",
    "feels_like_c":   "Apparent Temp",
    "humidity_pct":   "Humidity",
    "wind_speed_kmh": "Wind Speed",
    "pressure_hpa":   "Pressure",
}

FEATURE_UNITS = {
    "temperature_c":  "C",
    "feels_like_c":   "C",
    "humidity_pct":   "%",
    "wind_speed_kmh": "km/h",
    "pressure_hpa":   "hPa",
}

CONTAMINATION      = 0.05
SCORE_MODERATE     = -0.05
SCORE_SEVERE       = -0.10

# File paths
MODEL_CACHE_PATH   = "data/processed/isolation_forest_model.pkl"
SCALER_CACHE_PATH  = "data/processed/isolation_forest_scaler.pkl"
ML_RESULTS_PATH    = "data/processed/ml_anomaly_results.parquet"
SHAP_RESULTS_PATH  = "data/processed/shap_results.parquet"
SHAP_GLOBAL_PATH   = "data/processed/shap_global_importance.parquet"

# How many normal observations to use for SHAP background
# (computing SHAP for 108K rows would be slow; anomalies + sample is fine)
SHAP_NORMAL_SAMPLE = 500


# ── Phase 1: Training ─────────────────────────────────────────────────────────

def train_isolation_forest(df: pd.DataFrame,
                            force_retrain: bool = False
                            ) -> tuple:
    """
    Phase 1: Train (or load cached) Isolation Forest model.

    How Isolation Forest works:
    Random trees partition the feature space recursively. Observations
    that require few partitions to isolate are anomalous because they
    sit far from the bulk of the data. Normal observations cluster
    together and need many partitions to separate from neighbours.

    Returns (model, scaler, X_clean) or (None, None, empty) on failure.
    """
    if (not force_retrain
            and os.path.exists(MODEL_CACHE_PATH)
            and os.path.exists(SCALER_CACHE_PATH)):
        try:
            with open(MODEL_CACHE_PATH, "rb") as f:
                model = pickle.load(f)
            with open(SCALER_CACHE_PATH, "rb") as f:
                scaler = pickle.load(f)
            print("  ML: loaded cached Isolation Forest model.")
            available = [c for c in DETECTION_FEATURES if c in df.columns]
            X = df[available].copy()
            for col in available:
                X[col] = pd.to_numeric(X[col], errors="coerce")
            return model, scaler, X.dropna()
        except Exception as e:
            print(f"  ML: cache load failed ({e}), retraining.")

    print("  ML: training Isolation Forest...")
    available = [c for c in DETECTION_FEATURES if c in df.columns]
    missing   = [c for c in DETECTION_FEATURES if c not in df.columns]

    if missing:
        print(f"  ML: missing features {missing}, using {available}")
    if len(available) < 2:
        print("  ML: need at least 2 features. Skipping.")
        return None, None, pd.DataFrame()

    X = df[available].copy()
    for col in available:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X_clean = X.dropna()

    if len(X_clean) < 100:
        print(f"  ML: only {len(X_clean)} clean rows, too few to train.")
        return None, None, X_clean

    print(f"  ML: training on {len(X_clean):,} observations, "
          f"{len(available)} features...")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    model = IsolationForest(
        n_estimators=100,
        max_samples="auto",
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    os.makedirs(os.path.dirname(MODEL_CACHE_PATH), exist_ok=True)
    with open(MODEL_CACHE_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_CACHE_PATH, "wb") as f:
        pickle.dump(scaler, f)

    expected = int(CONTAMINATION * len(X_clean))
    print(f"  ML: training complete.")
    print(f"  Contamination: {CONTAMINATION} (~{expected:,} anomalies expected)")

    return model, scaler, X_clean


# ── Phase 2: Detection ────────────────────────────────────────────────────────

def run_detection(df: pd.DataFrame,
                  model: IsolationForest,
                  scaler: StandardScaler) -> pd.DataFrame:
    """
    Phase 2: Score every observation.

    Adds columns:
    - ml_anomaly_flag:   True if Isolation Forest flags as anomalous
    - ml_anomaly_score:  Raw score (more negative = more anomalous)
    - ml_severity:       Normal / Mild / Moderate / Severe
    - ml_primary_driver: Feature with highest z-score deviation (Phase 2 proxy)
    """
    if model is None or scaler is None:
        print("  ML: no model available, skipping detection.")
        return df

    available     = [c for c in DETECTION_FEATURES if c in df.columns]
    result        = df.copy()
    X             = result[available].copy()
    for col in available:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    complete_mask = X.notna().all(axis=1)
    X_complete    = X[complete_mask]

    result["ml_anomaly_flag"]   = False
    result["ml_anomaly_score"]  = np.nan
    result["ml_severity"]       = "Unknown"
    result["ml_primary_driver"] = ""

    if X_complete.empty:
        print("  ML: no complete rows to score.")
        return result

    print(f"  ML: scoring {len(X_complete):,} observations...")

    X_scaled       = scaler.transform(X_complete)
    predictions    = model.predict(X_scaled)
    anomaly_scores = model.score_samples(X_scaled)

    result.loc[complete_mask, "ml_anomaly_flag"]  = (predictions == -1)
    result.loc[complete_mask, "ml_anomaly_score"] = anomaly_scores.round(4)

    def _severity(score, flag):
        if not flag:
            return "Normal"
        if score <= SCORE_SEVERE:
            return "Severe"
        if score <= SCORE_MODERATE:
            return "Moderate"
        return "Mild"

    result["ml_severity"] = result.apply(
        lambda r: _severity(r["ml_anomaly_score"], r["ml_anomaly_flag"]),
        axis=1
    )

    # Phase 2 lightweight driver (z-score based proxy)
    feature_means = {col: X_complete[col].mean() for col in available}
    feature_stds  = {col: max(X_complete[col].std(), 0.001)
                     for col in available}

    anomaly_idx = result[result["ml_anomaly_flag"]].index
    if not anomaly_idx.empty:
        anom_rows  = result.loc[anomaly_idx, available].copy()
        for col in available:
            anom_rows[col] = pd.to_numeric(anom_rows[col], errors="coerce")
        deviations = pd.DataFrame(index=anom_rows.index)
        for col in available:
            deviations[col] = (
                (anom_rows[col] - feature_means[col]).abs()
                / feature_stds[col]
            )
        primary = deviations.idxmax(axis=1)
        result.loc[anomaly_idx, "ml_primary_driver"] = (
            primary.map(FEATURE_LABELS)
        )

    n_total   = len(X_complete)
    n_anomaly = int(result["ml_anomaly_flag"].sum())
    n_severe  = int((result["ml_severity"] == "Severe").sum())
    n_mod     = int((result["ml_severity"] == "Moderate").sum())

    print(f"\n  ML Detection Results:")
    print(f"  Scored        : {n_total:,}")
    print(f"  Anomalies     : {n_anomaly:,} ({n_anomaly/n_total*100:.1f}%)")
    print(f"  Severe        : {n_severe}")
    print(f"  Moderate      : {n_mod}")

    return result


# ── Phase 3: SHAP Explainability ──────────────────────────────────────────────

def _build_shap_explanation(shap_row: np.ndarray,
                              feature_vals: pd.Series,
                              available: list,
                              feature_means: dict,
                              feature_stds:  dict) -> str:
    """
    Build a plain English explanation from SHAP values for one observation.

    SHAP values are additive contributions to the anomaly score:
    - More negative = pushed score toward anomalous
    - More positive = pushed score toward normal

    For anomaly detection we focus on the most negative contributions.
    """
    contributions = []
    for i, feat in enumerate(available):
        shap_val   = float(shap_row[i])
        actual_val = float(feature_vals[feat]) \
            if feat in feature_vals.index else np.nan
        z_score    = (
            (actual_val - feature_means[feat]) / feature_stds[feat]
            if not np.isnan(actual_val) else 0
        )
        label = FEATURE_LABELS.get(feat, feat)
        unit  = FEATURE_UNITS.get(feat, "")
        contributions.append((feat, label, shap_val, actual_val,
                               z_score, unit))

    # Sort by contribution magnitude (most anomalous first)
    contributions.sort(key=lambda x: x[2])

    parts = []
    for feat, label, shap_val, actual, z, unit in contributions[:3]:
        if np.isnan(actual):
            continue
        direction = "anomalously high" if z > 0 else "anomalously low"
        parts.append(
            f"{label}: {actual:.1f}{unit} "
            f"({direction}, SHAP: {shap_val:+.4f})"
        )

    if not parts:
        return "No explanation available."
    return " | ".join(parts)


def run_shap_analysis(df: pd.DataFrame,
                       model: IsolationForest,
                       scaler: StandardScaler,
                       X_clean: pd.DataFrame,
                       force_recompute: bool = False) -> pd.DataFrame:
    """
    Phase 3: SHAP explainability for all anomalous observations.

    How SHAP works with Isolation Forest:
    - TreeExplainer decomposes each tree in the forest
    - For each observation, it computes the marginal contribution
      of each feature to the final anomaly score
    - Values are additive: they sum to the difference between
      the individual score and the expected background score
    - More negative SHAP = that feature pushed toward anomalous
    - More positive SHAP = that feature pushed toward normal

    Performance strategy:
    - Compute SHAP only for anomalous observations (not all 108K rows)
    - Use a small sample of normal rows as background reference
    - Saves SHAP values to Parquet for dashboard to load without recompute

    Adds columns to df:
    - shap_temperature_c, shap_feels_like_c, etc. (one per feature)
    - shap_top_driver     (feature with most negative SHAP = biggest contributor)
    - shap_top_driver_val (actual value of that feature)
    - shap_explanation    (plain English description)
    - shap_computed       (True if SHAP was successfully computed)
    """
    # Check if SHAP already computed and saved
    if (not force_recompute
            and os.path.exists(SHAP_RESULTS_PATH)
            and "ml_anomaly_flag" in df.columns):
        try:
            shap_df = pd.read_parquet(SHAP_RESULTS_PATH)
            shap_cols = [c for c in shap_df.columns
                         if c.startswith("shap_")]
            if shap_cols:
                print(f"  SHAP: loaded {len(shap_cols)} cached columns.")
                # Merge SHAP results back on index
                for col in shap_cols:
                    if col in shap_df.columns:
                        df[col] = shap_df[col].reindex(df.index)
                return df
        except Exception as e:
            print(f"  SHAP: cache load failed ({e}), recomputing.")

    # Check dependencies
    try:
        import shap as shap_lib
    except ImportError:
        print("  SHAP: shap not installed. Run: pip install shap")
        print("  Skipping Phase 3.")
        return df

    if model is None or scaler is None:
        print("  SHAP: no model available. Skipping Phase 3.")
        return df

    if "ml_anomaly_flag" not in df.columns:
        print("  SHAP: Phase 2 not complete. Skipping Phase 3.")
        return df

    available = [c for c in DETECTION_FEATURES if c in df.columns]
    if not available:
        return df

    print("  SHAP: computing explainability (Phase 3)...")

    # Get clean feature matrix
    X = df[available].copy()
    for col in available:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    complete_mask = X.notna().all(axis=1)

    # Separate anomalous and normal rows
    anomaly_mask  = df["ml_anomaly_flag"] == True
    anomaly_index = df[anomaly_mask & complete_mask].index
    normal_index  = df[~anomaly_mask & complete_mask].index

    if len(anomaly_index) == 0:
        print("  SHAP: no anomalies found to explain.")
        return df

    print(f"  SHAP: explaining {len(anomaly_index):,} anomalous observations...")

    # Sample normal observations as SHAP background
    n_background = min(SHAP_NORMAL_SAMPLE, len(normal_index))
    bg_index     = normal_index[:n_background] \
        if len(normal_index) > 0 else anomaly_index[:50]

    X_anomaly = scaler.transform(X.loc[anomaly_index])
    X_bg      = scaler.transform(X.loc[bg_index])

    # Initialise SHAP explainer using background dataset
    # TreeExplainer decomposes all trees in the Isolation Forest
    try:
        explainer   = shap_lib.TreeExplainer(
            model,
            data=X_bg,
            feature_perturbation="interventional",
        )
        shap_values = explainer.shap_values(X_anomaly)

    except Exception as e:
        print(f"  SHAP: TreeExplainer failed ({e}).")
        print("  SHAP: trying without background data...")
        try:
            explainer   = shap_lib.TreeExplainer(model)
            shap_values = explainer.shap_values(X_anomaly)
        except Exception as e2:
            print(f"  SHAP: failed entirely: {e2}")
            print("  Skipping Phase 3.")
            return df

    # shap_values shape: (n_anomalies, n_features)
    shap_array = np.array(shap_values)
    if shap_array.ndim == 3:
        # Some versions return 3D -- take the positive class
        shap_array = shap_array[:, :, 1] \
            if shap_array.shape[2] == 2 else shap_array[:, :, 0]

    print(f"  SHAP: computed values shape {shap_array.shape}")

    # Build SHAP column names
    shap_col_names = [f"shap_{feat}" for feat in available]

    # Initialise SHAP columns with NaN for all rows
    for col in shap_col_names:
        df[col] = np.nan
    df["shap_top_driver"]     = ""
    df["shap_top_driver_val"] = np.nan
    df["shap_explanation"]    = ""
    df["shap_computed"]       = False

    # Fill in SHAP values for anomalous rows
    feature_means = {col: X.loc[complete_mask, col].mean()
                     for col in available}
    feature_stds  = {col: max(X.loc[complete_mask, col].std(), 0.001)
                     for col in available}

    for i, idx in enumerate(anomaly_index):
        if i >= len(shap_array):
            break

        shap_row = shap_array[i]

        # Store SHAP values
        for j, feat in enumerate(available):
            df.at[idx, f"shap_{feat}"] = round(float(shap_row[j]), 6)

        # Identify top driver (most negative SHAP = biggest anomaly contributor)
        top_j    = int(np.argmin(shap_row))
        top_feat = available[top_j]
        top_val  = float(X.at[idx, top_feat]) \
            if idx in X.index else np.nan

        df.at[idx, "shap_top_driver"]     = FEATURE_LABELS.get(top_feat, top_feat)
        df.at[idx, "shap_top_driver_val"] = round(top_val, 2)

        # Build plain English explanation
        feature_vals = X.loc[idx] if idx in X.index else pd.Series()
        explanation  = _build_shap_explanation(
            shap_row, feature_vals, available,
            feature_means, feature_stds
        )
        df.at[idx, "shap_explanation"] = explanation
        df.at[idx, "shap_computed"]    = True

    # Compute global feature importance (mean absolute SHAP per feature)
    if len(shap_array) > 0:
        global_importance = pd.DataFrame({
            "feature":    available,
            "label":      [FEATURE_LABELS.get(f, f) for f in available],
            "unit":       [FEATURE_UNITS.get(f, "") for f in available],
            "mean_abs_shap": np.abs(shap_array).mean(axis=0).round(6),
            "mean_shap":     shap_array.mean(axis=0).round(6),
            "n_anomalies":   len(anomaly_index),
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        os.makedirs(os.path.dirname(SHAP_GLOBAL_PATH), exist_ok=True)
        global_importance.to_parquet(SHAP_GLOBAL_PATH, index=False)
        print(f"  SHAP: global importance saved.")
        print("  SHAP: Feature importance ranking:")
        for _, row in global_importance.iterrows():
            bar = "█" * int(row["mean_abs_shap"] * 500)
            print(f"    {row['label']:<18} {row['mean_abs_shap']:.5f}  {bar}")

    # Save SHAP results
    shap_result_cols = (
        shap_col_names +
        ["shap_top_driver", "shap_top_driver_val",
         "shap_explanation", "shap_computed"]
    )
    shap_save = df[shap_result_cols].copy()
    os.makedirs(os.path.dirname(SHAP_RESULTS_PATH), exist_ok=True)
    shap_save.to_parquet(SHAP_RESULTS_PATH, index=False)

    n_explained = int(df["shap_computed"].sum())
    print(f"\n  SHAP Phase 3 Complete:")
    print(f"  Anomalies explained : {n_explained:,}")
    print(f"  SHAP cols added     : {len(shap_col_names)}")
    print(f"  Results saved to    : {SHAP_RESULTS_PATH}")

    return df


# ── Station summary ───────────────────────────────────────────────────────────

def compute_station_anomaly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ML and SHAP results to station level."""
    if "ml_anomaly_flag" not in df.columns:
        return pd.DataFrame()

    agg_dict = dict(
        total_observations  = ("ml_anomaly_flag", "count"),
        anomaly_count       = ("ml_anomaly_flag", "sum"),
        avg_anomaly_score   = ("ml_anomaly_score", "mean"),
        min_anomaly_score   = ("ml_anomaly_score", "min"),
        severe_count        = ("ml_severity",
                               lambda x: (x == "Severe").sum()),
        moderate_count      = ("ml_severity",
                               lambda x: (x == "Moderate").sum()),
        primary_driver      = ("ml_primary_driver",
                               lambda x: x.mode().iloc[0]
                               if not x.mode().empty else ""),
        state               = ("state", "first"),
        latitude            = ("latitude", "first"),
        longitude           = ("longitude", "first"),
    )

    # Add SHAP top driver if available
    if "shap_top_driver" in df.columns:
        agg_dict["shap_top_driver"] = (
            "shap_top_driver",
            lambda x: x[x != ""].mode().iloc[0]
            if not x[x != ""].mode().empty else ""
        )
    if "shap_explanation" in df.columns:
        agg_dict["shap_example"] = (
            "shap_explanation",
            lambda x: x[x != ""].iloc[0] if (x != "").any() else ""
        )

    summary = (
        df.groupby("station_name")
        .agg(**agg_dict)
        .reset_index()
    )

    summary["anomaly_rate_pct"] = (
        summary["anomaly_count"] / summary["total_observations"] * 100
    ).round(1)
    summary["avg_anomaly_score"] = summary["avg_anomaly_score"].round(4)
    summary["min_anomaly_score"] = summary["min_anomaly_score"].round(4)

    summary["station_risk"] = summary.apply(
        lambda r:
        "High"   if r["severe_count"] > 0   else
        "Medium" if r["moderate_count"] > 2  else
        "Low"    if r["anomaly_count"] > 0   else
        "Normal",
        axis=1
    )

    return summary.sort_values("min_anomaly_score")


def save_ml_results(df: pd.DataFrame) -> None:
    """Save ML and SHAP annotated DataFrame to Parquet."""
    os.makedirs(os.path.dirname(ML_RESULTS_PATH), exist_ok=True)
    df.to_parquet(ML_RESULTS_PATH, index=False)
    size_kb = os.path.getsize(ML_RESULTS_PATH) / 1024
    print(f"  ML results saved: {ML_RESULTS_PATH} "
          f"({len(df)} rows, {size_kb:.1f} KB)")


# ── Main pipeline function ────────────────────────────────────────────────────

def run_ml_pipeline(df: pd.DataFrame,
                    force_retrain:  bool = False,
                    force_shap:     bool = False) -> pd.DataFrame:
    """
    Run the complete ML pipeline: Phase 1, 2, and 3.

    Phase 1: Train or load Isolation Forest
    Phase 2: Score all observations, assign severity and driver
    Phase 3: SHAP attribution for all anomalous observations

    Parameters
    ----------
    df            : cleaned and merged BOM DataFrame
    force_retrain : retrain the model even if cache exists
    force_shap    : recompute SHAP even if cache exists

    Returns
    -------
    DataFrame with ML and SHAP columns added.
    """
    print("\nML Anomaly Detection Pipeline")
    print("-" * 40)

    if df.empty:
        print("  Empty DataFrame, skipping ML.")
        return df

    available = [c for c in DETECTION_FEATURES if c in df.columns]
    if len(available) < 2:
        print(f"  Only {len(available)} features available. Need 2+. Skipping.")
        return df

    # Phase 1
    model, scaler, X_clean = train_isolation_forest(
        df, force_retrain=force_retrain
    )
    if model is None:
        print("  Phase 1 failed, skipping.")
        return df

    # Phase 2
    df = run_detection(df, model, scaler)

    # Phase 3
    df = run_shap_analysis(
        df, model, scaler, X_clean,
        force_recompute=force_shap
    )

    save_ml_results(df)
    return df


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pipeline.storage import load_parquet
    from config import MERGED_DATA_PATH

    print("=" * 55)
    print("AusClimate ML Pipeline Test (Phases 1, 2, 3)")
    print("=" * 55)

    df = load_parquet(MERGED_DATA_PATH)
    if df.empty:
        print("No merged data. Run python main.py first.")
    else:
        df_ml = run_ml_pipeline(df, force_retrain=True, force_shap=True)

        print("\nTop 5 most anomalous observations:")
        cols = [c for c in [
            "station_name", "temperature_c", "humidity_pct",
            "ml_anomaly_score", "ml_severity",
            "shap_top_driver", "shap_explanation"
        ] if c in df_ml.columns]
        top5 = (
            df_ml[df_ml["ml_anomaly_flag"] == True]
            .nsmallest(5, "ml_anomaly_score")[cols]
        )
        for _, row in top5.iterrows():
            print(f"\n  Station: {row.get('station_name', '?')}")
            print(f"  Score  : {row.get('ml_anomaly_score', '?')}")
            print(f"  Driver : {row.get('shap_top_driver', '?')}")
            print(f"  Explain: {row.get('shap_explanation', '?')}")