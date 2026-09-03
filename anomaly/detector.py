"""
anomaly/detector.py

ML anomaly detection pipeline for AusClimate.

Phase 1 -- Training:
    Trains an Isolation Forest model on multi-variable BOM data.
    Learns what normal looks like across 5 climate variables.

Phase 2 -- IF Detection:
    Scores every observation with IF. Assigns severity and driver.

Phase 3 -- SHAP Explainability:
    Computes precise Shapley values for every anomalous observation.
    Each anomaly gets per-feature attribution scores and a plain
    English explanation showing which variables drove the flag and by
    how much. SHAP values are mathematically additive: they sum to
    the total anomaly score.

Phase 4 -- LOF Detection:
    Local Outlier Factor as independent second model.
    Detects anomalies based on local density rather than isolation.
    Catches different anomaly patterns than IF.

Phase 5 -- Ensemble Voting:
    Flags observations where BOTH IF and LOF agree.
    Consensus anomalies have much lower false positive rate.

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
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# ── Constants ─────────────────────────────────────────────────────────────────

DETECTION_FEATURES = [
    "temperature_c",
    "feels_like_c",
    "humidity_pct",
    "wind_speed_kmh",
    "pressure_hpa",
]

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

# Model settings
CONTAMINATION      = 0.05
IF_N_ESTIMATORS    = 100
LOF_N_NEIGHBORS    = 20

# Anomaly score thresholds
SCORE_MODERATE     = -0.05
SCORE_SEVERE       = -0.10
LOF_SCORE_MODERATE = -1.5
LOF_SCORE_SEVERE   = -2.0

# File paths
MODEL_CACHE_PATH   = "data/processed/isolation_forest_model.pkl"
SCALER_CACHE_PATH  = "data/processed/isolation_forest_scaler.pkl"
LOF_CACHE_PATH     = "data/processed/lof_model.pkl"
ML_RESULTS_PATH    = "data/processed/ml_anomaly_results.parquet"
SHAP_RESULTS_PATH  = "data/processed/shap_results.parquet"
SHAP_GLOBAL_PATH   = "data/processed/shap_global_importance.parquet"

SHAP_NORMAL_SAMPLE = 500


# ── Helper: clean feature matrix ─────────────────────────────────────────────

def _get_clean_features(df: pd.DataFrame) -> tuple:
    """
    Extract and clean the feature matrix from a DataFrame.
    Returns (available_features, X_complete, complete_mask).
    """
    available     = [c for c in DETECTION_FEATURES if c in df.columns]
    X             = df[available].copy()
    for col in available:
        X[col]    = pd.to_numeric(X[col], errors="coerce")
    complete_mask = X.notna().all(axis=1)
    return available, X, complete_mask


# ── Phase 1: Isolation Forest Training ───────────────────────────────────────

def train_isolation_forest(df: pd.DataFrame,
                            force_retrain: bool = False) -> tuple:
    """
    Phase 1: Train or load cached Isolation Forest model.

    Isolation Forest works by randomly partitioning the feature space.
    Anomalous observations get isolated quickly because they sit apart
    from the bulk of the data. Normal observations cluster together
    and require many more partitions to separate.

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
            print("  IF: loaded cached model.")
            _, X, complete_mask = _get_clean_features(df)
            return model, scaler, X[complete_mask]
        except Exception as e:
            print(f"  IF: cache load failed ({e}), retraining.")

    print("  IF: training Isolation Forest...")
    available, X, complete_mask = _get_clean_features(df)
    X_clean = X[complete_mask]

    missing = [c for c in DETECTION_FEATURES if c not in df.columns]
    if missing:
        print(f"  IF: missing features {missing}, using {available}")
    if len(available) < 2:
        print("  IF: need at least 2 features. Skipping.")
        return None, None, pd.DataFrame()
    if len(X_clean) < 100:
        print(f"  IF: only {len(X_clean)} rows, too few to train.")
        return None, None, X_clean

    print(f"  IF: training on {len(X_clean):,} observations, "
          f"{len(available)} features...")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    model = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
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
    print(f"  IF: training complete. "
          f"~{expected:,} anomalies expected ({CONTAMINATION*100:.0f}%)")

    return model, scaler, X_clean


# ── Phase 2: IF Detection ─────────────────────────────────────────────────────

def run_detection(df: pd.DataFrame,
                  model: IsolationForest,
                  scaler: StandardScaler) -> pd.DataFrame:
    """
    Phase 2: Score every observation with Isolation Forest.

    Adds columns:
    - ml_anomaly_flag:   True if IF flags as anomalous
    - ml_anomaly_score:  Raw score (more negative = more anomalous)
    - ml_severity:       Normal / Mild / Moderate / Severe
    - ml_primary_driver: Feature with highest z-score deviation
    """
    if model is None or scaler is None:
        print("  IF: no model, skipping detection.")
        return df

    available, X, complete_mask = _get_clean_features(df)
    X_complete = X[complete_mask]

    result = df.copy()
    result["ml_anomaly_flag"]   = False
    result["ml_anomaly_score"]  = np.nan
    result["ml_severity"]       = "Unknown"
    result["ml_primary_driver"] = ""

    if X_complete.empty:
        return result

    print(f"  IF: scoring {len(X_complete):,} observations...")

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

    # Phase 2 lightweight driver attribution
    feature_means = {col: X_complete[col].mean() for col in available}
    feature_stds  = {col: max(X_complete[col].std(), 0.001)
                     for col in available}
    anomaly_idx   = result[result["ml_anomaly_flag"]].index

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

    print(f"  IF results: {n_anomaly:,}/{n_total:,} anomalies "
          f"({n_anomaly/n_total*100:.1f}%) | "
          f"Severe: {n_severe} | Moderate: {n_mod}")

    return result


# ── Phase 3: SHAP ─────────────────────────────────────────────────────────────

def _build_shap_explanation(shap_row: np.ndarray,
                              feature_vals: pd.Series,
                              available: list,
                              feature_means: dict,
                              feature_stds:  dict) -> str:
    """Build plain English explanation from SHAP values."""
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
        contributions.append((feat, label, shap_val, actual_val, z_score, unit))

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
    return " | ".join(parts) if parts else "No explanation available."


def run_shap_analysis(df: pd.DataFrame,
                       model: IsolationForest,
                       scaler: StandardScaler,
                       X_clean: pd.DataFrame,
                       force_recompute: bool = False) -> pd.DataFrame:
    """
    Phase 3: SHAP explainability for all IF anomalous observations.

    Computes Shapley values showing exactly how much each variable
    contributed to each anomaly score. Values are additive.
    More negative SHAP = that variable pushed toward anomalous.
    """
    if (not force_recompute and os.path.exists(SHAP_RESULTS_PATH)
            and "ml_anomaly_flag" in df.columns):
        try:
            shap_df   = pd.read_parquet(SHAP_RESULTS_PATH)
            shap_cols = [c for c in shap_df.columns if c.startswith("shap_")]
            if shap_cols:
                print(f"  SHAP: loaded {len(shap_cols)} cached columns.")
                for col in shap_cols:
                    df[col] = shap_df[col].reindex(df.index)
                return df
        except Exception as e:
            print(f"  SHAP: cache load failed ({e}), recomputing.")

    try:
        import shap as shap_lib
    except ImportError:
        print("  SHAP: not installed. Run: pip install shap")
        return df

    if model is None or scaler is None:
        print("  SHAP: no model, skipping.")
        return df
    if "ml_anomaly_flag" not in df.columns:
        print("  SHAP: Phase 2 not complete, skipping.")
        return df

    available, X, complete_mask = _get_clean_features(df)
    if not available:
        return df

    anomaly_mask  = df["ml_anomaly_flag"] == True
    anomaly_index = df[anomaly_mask & complete_mask].index
    normal_index  = df[~anomaly_mask & complete_mask].index

    if len(anomaly_index) == 0:
        print("  SHAP: no anomalies to explain.")
        return df

    print(f"  SHAP: explaining {len(anomaly_index):,} anomalies...")

    n_bg    = min(SHAP_NORMAL_SAMPLE, len(normal_index))
    bg_idx  = normal_index[:n_bg] if len(normal_index) > 0 \
        else anomaly_index[:50]

    X_anomaly = scaler.transform(X.loc[anomaly_index])
    X_bg      = scaler.transform(X.loc[bg_idx])

    try:
        explainer   = shap_lib.TreeExplainer(
            model, data=X_bg,
            feature_perturbation="interventional",
        )
        shap_values = explainer.shap_values(X_anomaly)
    except Exception as e:
        print(f"  SHAP: TreeExplainer failed ({e}), trying without background...")
        try:
            explainer   = shap_lib.TreeExplainer(model)
            shap_values = explainer.shap_values(X_anomaly)
        except Exception as e2:
            print(f"  SHAP: failed entirely: {e2}")
            return df

    shap_array = np.array(shap_values)
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 1] \
            if shap_array.shape[2] == 2 else shap_array[:, :, 0]

    shap_col_names = [f"shap_{feat}" for feat in available]

    for col in shap_col_names:
        df[col] = np.nan
    df["shap_top_driver"]     = ""
    df["shap_top_driver_val"] = np.nan
    df["shap_explanation"]    = ""
    df["shap_computed"]       = False

    feature_means = {col: X[complete_mask][col].mean() for col in available}
    feature_stds  = {col: max(X[complete_mask][col].std(), 0.001)
                     for col in available}

    for i, idx in enumerate(anomaly_index):
        if i >= len(shap_array):
            break
        shap_row = shap_array[i]
        for j, feat in enumerate(available):
            df.at[idx, f"shap_{feat}"] = round(float(shap_row[j]), 6)
        top_j    = int(np.argmin(shap_row))
        top_feat = available[top_j]
        top_val  = float(X.at[idx, top_feat]) \
            if idx in X.index else np.nan
        df.at[idx, "shap_top_driver"]     = FEATURE_LABELS.get(top_feat, top_feat)
        df.at[idx, "shap_top_driver_val"] = round(top_val, 2)
        feature_vals = X.loc[idx] if idx in X.index else pd.Series()
        df.at[idx, "shap_explanation"] = _build_shap_explanation(
            shap_row, feature_vals, available, feature_means, feature_stds
        )
        df.at[idx, "shap_computed"] = True

    if len(shap_array) > 0:
        global_imp = pd.DataFrame({
            "feature":       available,
            "label":         [FEATURE_LABELS.get(f, f) for f in available],
            "unit":          [FEATURE_UNITS.get(f, "") for f in available],
            "mean_abs_shap": np.abs(shap_array).mean(axis=0).round(6),
            "mean_shap":     shap_array.mean(axis=0).round(6),
            "n_anomalies":   len(anomaly_index),
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        os.makedirs(os.path.dirname(SHAP_GLOBAL_PATH), exist_ok=True)
        global_imp.to_parquet(SHAP_GLOBAL_PATH, index=False)
        print("  SHAP: global importance ranking:")
        for _, row in global_imp.iterrows():
            bar = "█" * int(row["mean_abs_shap"] * 500)
            print(f"    {row['label']:<18} {row['mean_abs_shap']:.5f}  {bar}")

    shap_save_cols = (
        shap_col_names +
        ["shap_top_driver", "shap_top_driver_val",
         "shap_explanation", "shap_computed"]
    )
    os.makedirs(os.path.dirname(SHAP_RESULTS_PATH), exist_ok=True)
    df[shap_save_cols].to_parquet(SHAP_RESULTS_PATH, index=False)

    n_explained = int(df["shap_computed"].sum())
    print(f"  SHAP: {n_explained:,} anomalies explained.")

    return df


# ── Phase 4: LOF Training and Detection ──────────────────────────────────────

def train_lof(df: pd.DataFrame,
              force_retrain: bool = False) -> tuple:
    """
    Phase 4 (Part A): Train or load cached Local Outlier Factor model.

    How LOF works:
    LOF computes the local density of each observation relative to
    its k nearest neighbours. If a point's neighbourhood is much
    less dense than its neighbours' neighbourhoods, it is anomalous.

    This is fundamentally different to Isolation Forest:
    - IF uses global random partitioning (good for global outliers)
    - LOF uses local neighbourhood density (good for local outliers)

    A reading that is not globally extreme but sits in an unusually
    sparse local region of the feature space will be caught by LOF
    but potentially missed by IF. This is why combining them works.

    n_neighbors: how many neighbours to consider when estimating
    local density. 20 is standard for medium-large datasets.
    Higher values = smoother decision boundary, less sensitive to noise.

    Returns (lof_model, scaler, X_clean) or (None, None, empty).
    """
    if (not force_retrain and os.path.exists(LOF_CACHE_PATH)
            and os.path.exists(SCALER_CACHE_PATH)):
        try:
            with open(LOF_CACHE_PATH, "rb") as f:
                lof_model = pickle.load(f)
            with open(SCALER_CACHE_PATH, "rb") as f:
                scaler = pickle.load(f)
            print("  LOF: loaded cached model.")
            _, X, complete_mask = _get_clean_features(df)
            return lof_model, scaler, X[complete_mask]
        except Exception as e:
            print(f"  LOF: cache load failed ({e}), retraining.")

    print("  LOF: training Local Outlier Factor...")
    available, X, complete_mask = _get_clean_features(df)
    X_clean = X[complete_mask]

    if len(available) < 2:
        print("  LOF: need at least 2 features. Skipping.")
        return None, None, pd.DataFrame()
    if len(X_clean) < LOF_N_NEIGHBORS + 1:
        print(f"  LOF: too few rows ({len(X_clean)}) for "
              f"n_neighbors={LOF_N_NEIGHBORS}.")
        return None, None, X_clean

    print(f"  LOF: training on {len(X_clean):,} observations, "
          f"{len(available)} features, "
          f"n_neighbors={LOF_N_NEIGHBORS}...")

    # Load scaler from IF (same scaler for fair comparison)
    scaler = None
    if os.path.exists(SCALER_CACHE_PATH):
        try:
            with open(SCALER_CACHE_PATH, "rb") as f:
                scaler = pickle.load(f)
        except Exception:
            pass

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X_clean)

    X_scaled = scaler.transform(X_clean)

    # novelty=True allows predict() on new data
    # Without this we can only use fit_predict on training data
    lof_model = LocalOutlierFactor(
        n_neighbors=LOF_N_NEIGHBORS,
        contamination=CONTAMINATION,
        novelty=True,
        n_jobs=-1,
        algorithm="auto",
        leaf_size=30,
    )
    lof_model.fit(X_scaled)

    os.makedirs(os.path.dirname(LOF_CACHE_PATH), exist_ok=True)
    with open(LOF_CACHE_PATH, "wb") as f:
        pickle.dump(lof_model, f)

    expected = int(CONTAMINATION * len(X_clean))
    print(f"  LOF: training complete. "
          f"~{expected:,} anomalies expected ({CONTAMINATION*100:.0f}%)")

    return lof_model, scaler, X_clean


def run_lof_detection(df: pd.DataFrame,
                       lof_model: LocalOutlierFactor,
                       scaler: StandardScaler) -> pd.DataFrame:
    """
    Phase 4 (Part B): Score every observation with LOF.

    LOF score interpretation:
    - Score of 1.0 means the observation is as dense as its neighbours
    - Score > 1.0 means less dense than neighbours (outlier)
    - We use negative_outlier_factor_ convention:
      more negative = more anomalous (consistent with IF convention)

    Adds columns:
    - lof_anomaly_flag:   True if LOF flags as anomalous
    - lof_anomaly_score:  Negative LOF score (more negative = more anomalous)
    - lof_severity:       Normal / Mild / Moderate / Severe
    - lof_primary_driver: Feature with highest z-score deviation
    """
    if lof_model is None or scaler is None:
        print("  LOF: no model, skipping detection.")
        return df

    available, X, complete_mask = _get_clean_features(df)
    X_complete = X[complete_mask]

    result = df.copy()
    result["lof_anomaly_flag"]   = False
    result["lof_anomaly_score"]  = np.nan
    result["lof_severity"]       = "Unknown"
    result["lof_primary_driver"] = ""

    if X_complete.empty:
        return result

    print(f"  LOF: scoring {len(X_complete):,} observations...")

    X_scaled    = scaler.transform(X_complete)
    predictions = lof_model.predict(X_scaled)          # 1=normal, -1=anomaly
    lof_scores  = lof_model.score_samples(X_scaled)    # more negative = anomalous

    result.loc[complete_mask, "lof_anomaly_flag"]  = (predictions == -1)
    result.loc[complete_mask, "lof_anomaly_score"] = lof_scores.round(4)

    def _lof_severity(score, flag):
        if not flag:
            return "Normal"
        if score <= LOF_SCORE_SEVERE:
            return "Severe"
        if score <= LOF_SCORE_MODERATE:
            return "Moderate"
        return "Mild"

    result["lof_severity"] = result.apply(
        lambda r: _lof_severity(
            r["lof_anomaly_score"], r["lof_anomaly_flag"]
        ),
        axis=1
    )

    # LOF primary driver (z-score based, same approach as IF Phase 2)
    feature_means = {col: X_complete[col].mean() for col in available}
    feature_stds  = {col: max(X_complete[col].std(), 0.001)
                     for col in available}
    lof_anom_idx  = result[result["lof_anomaly_flag"]].index

    if not lof_anom_idx.empty:
        anom_rows  = result.loc[lof_anom_idx, available].copy()
        for col in available:
            anom_rows[col] = pd.to_numeric(anom_rows[col], errors="coerce")
        deviations = pd.DataFrame(index=anom_rows.index)
        for col in available:
            deviations[col] = (
                (anom_rows[col] - feature_means[col]).abs()
                / feature_stds[col]
            )
        primary = deviations.idxmax(axis=1)
        result.loc[lof_anom_idx, "lof_primary_driver"] = (
            primary.map(FEATURE_LABELS)
        )

    n_total   = len(X_complete)
    n_anomaly = int(result["lof_anomaly_flag"].sum())
    n_severe  = int((result["lof_severity"] == "Severe").sum())
    n_mod     = int((result["lof_severity"] == "Moderate").sum())

    print(f"  LOF results: {n_anomaly:,}/{n_total:,} anomalies "
          f"({n_anomaly/n_total*100:.1f}%) | "
          f"Severe: {n_severe} | Moderate: {n_mod}")

    return result


# ── Phase 5: Ensemble Voting ──────────────────────────────────────────────────

def compute_ensemble(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 5: Consensus anomaly detection.

    Combines IF and LOF results. An observation is flagged as an
    ensemble anomaly only when BOTH models independently agree it
    is anomalous.

    Why this matters:
    - IF alone: ~5% false positive rate (by contamination setting)
    - LOF alone: ~5% false positive rate (by contamination setting)
    - Both agree: probability both independently wrong is ~0.25%
      This dramatically reduces false positives for operational users.

    Adds columns:
    - if_only:        flagged by IF but NOT LOF
    - lof_only:       flagged by LOF but NOT IF
    - ensemble_anomaly: flagged by BOTH IF and LOF (high confidence)
    - ensemble_severity: severity based on worst of the two models
    """
    if "ml_anomaly_flag" not in df.columns:
        print("  Ensemble: IF results missing, skipping.")
        return df
    if "lof_anomaly_flag" not in df.columns:
        print("  Ensemble: LOF results missing, skipping.")
        return df

    result = df.copy()

    if_flag  = result["ml_anomaly_flag"].fillna(False).astype(bool)
    lof_flag = result["lof_anomaly_flag"].fillna(False).astype(bool)

    result["if_only"]           = if_flag  & ~lof_flag
    result["lof_only"]          = lof_flag & ~if_flag
    result["ensemble_anomaly"]  = if_flag  & lof_flag

    # Ensemble severity: take the worse of the two
    def _ensemble_severity(row):
        if not row.get("ensemble_anomaly", False):
            return "Not Flagged"
        sev_order = {"Severe": 3, "Moderate": 2, "Mild": 1,
                     "Normal": 0, "Unknown": 0}
        if_sev  = sev_order.get(row.get("ml_severity",  "Normal"), 0)
        lof_sev = sev_order.get(row.get("lof_severity", "Normal"), 0)
        worst   = max(if_sev, lof_sev)
        return {3: "Severe", 2: "Moderate", 1: "Mild", 0: "Normal"}.get(
            worst, "Normal"
        )

    result["ensemble_severity"] = result.apply(_ensemble_severity, axis=1)

    # Summary
    n_if_only     = int(result["if_only"].sum())
    n_lof_only    = int(result["lof_only"].sum())
    n_ensemble    = int(result["ensemble_anomaly"].sum())
    n_total       = len(result)
    n_either      = int((if_flag | lof_flag).sum())
    n_severe_ens  = int(
        (result["ensemble_severity"] == "Severe").sum()
    )

    print(f"\n  Ensemble voting results:")
    print(f"  IF only (not LOF)   : {n_if_only:,} "
          f"({n_if_only/n_total*100:.1f}%)")
    print(f"  LOF only (not IF)   : {n_lof_only:,} "
          f"({n_lof_only/n_total*100:.1f}%)")
    print(f"  Both agree (high confidence): {n_ensemble:,} "
          f"({n_ensemble/n_total*100:.2f}%)")
    print(f"  Ensemble severe     : {n_severe_ens}")
    print(f"  Total flagged by either: {n_either:,}")

    return result


# ── Station-level summaries ───────────────────────────────────────────────────

def compute_station_anomaly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ML, LOF, and ensemble results to station level.
    Returns one row per station with anomaly statistics.
    """
    if "ml_anomaly_flag" not in df.columns:
        return pd.DataFrame()

    agg_dict = dict(
        total_observations  = ("ml_anomaly_flag", "count"),
        if_anomaly_count    = ("ml_anomaly_flag", "sum"),
        avg_if_score        = ("ml_anomaly_score", "mean"),
        min_if_score        = ("ml_anomaly_score", "min"),
        if_severe_count     = ("ml_severity",
                               lambda x: (x == "Severe").sum()),
        if_moderate_count   = ("ml_severity",
                               lambda x: (x == "Moderate").sum()),
        primary_driver      = ("ml_primary_driver",
                               lambda x: x.mode().iloc[0]
                               if not x.mode().empty else ""),
        state               = ("state", "first"),
        latitude            = ("latitude", "first"),
        longitude           = ("longitude", "first"),
    )

    if "lof_anomaly_flag" in df.columns:
        agg_dict["lof_anomaly_count"] = ("lof_anomaly_flag", "sum")
        agg_dict["avg_lof_score"]     = ("lof_anomaly_score", "mean")
        agg_dict["lof_severe_count"]  = ("lof_severity",
                                          lambda x: (x == "Severe").sum())

    if "ensemble_anomaly" in df.columns:
        agg_dict["ensemble_count"] = ("ensemble_anomaly", "sum")

    if "shap_top_driver" in df.columns:
        agg_dict["shap_driver"] = (
            "shap_top_driver",
            lambda x: x[x != ""].mode().iloc[0]
            if not x[x != ""].mode().empty else ""
        )

    summary = (
        df.groupby("station_name")
        .agg(**agg_dict)
        .reset_index()
    )

    summary["if_anomaly_rate_pct"] = (
        summary["if_anomaly_count"] /
        summary["total_observations"] * 100
    ).round(1)

    if "lof_anomaly_count" in summary.columns:
        summary["lof_anomaly_rate_pct"] = (
            summary["lof_anomaly_count"] /
            summary["total_observations"] * 100
        ).round(1)

    if "ensemble_count" in summary.columns:
        summary["ensemble_rate_pct"] = (
            summary["ensemble_count"] /
            summary["total_observations"] * 100
        ).round(2)

    # Risk based on ensemble where available, else IF
    def _risk(row):
        if "ensemble_count" in row.index:
            if row.get("ensemble_count", 0) > 0:
                return "High"
            if row.get("if_severe_count", 0) > 0:
                return "Medium"
            if row.get("if_anomaly_count", 0) > 0:
                return "Low"
            return "Normal"
        else:
            if row.get("if_severe_count", 0) > 0:
                return "High"
            if row.get("if_moderate_count", 0) > 2:
                return "Medium"
            if row.get("if_anomaly_count", 0) > 0:
                return "Low"
            return "Normal"

    summary["station_risk"]     = summary.apply(_risk, axis=1)
    summary["avg_if_score"]     = summary["avg_if_score"].round(4)
    summary["min_if_score"]     = summary["min_if_score"].round(4)

    return summary.sort_values("min_if_score")


def save_ml_results(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(ML_RESULTS_PATH), exist_ok=True)
    df.to_parquet(ML_RESULTS_PATH, index=False)
    size_kb = os.path.getsize(ML_RESULTS_PATH) / 1024
    print(f"  ML results: {ML_RESULTS_PATH} "
          f"({len(df)} rows, {size_kb:.1f} KB)")


# ── Main pipeline function ────────────────────────────────────────────────────

def run_ml_pipeline(df: pd.DataFrame,
                    force_retrain:  bool = False,
                    force_shap:     bool = False) -> pd.DataFrame:
    """
    Run the complete ML pipeline: Phases 1 through 5.

    Phase 1: Train/load Isolation Forest
    Phase 2: IF scoring and severity
    Phase 3: SHAP attribution for IF anomalies
    Phase 4: Train/load LOF, score all observations
    Phase 5: Ensemble voting (IF AND LOF consensus)
    """
    print("\nML Anomaly Detection Pipeline")
    print("-" * 40)

    if df.empty:
        print("  Empty DataFrame, skipping.")
        return df

    available = [c for c in DETECTION_FEATURES if c in df.columns]
    if len(available) < 2:
        print(f"  Only {len(available)} features, need 2+. Skipping.")
        return df

    # Phase 1: Train IF
    model, scaler, X_clean = train_isolation_forest(
        df, force_retrain=force_retrain
    )
    if model is None:
        print("  Phase 1 failed.")
        return df

    # Phase 2: IF detection
    df = run_detection(df, model, scaler)

    # Phase 3: SHAP
    df = run_shap_analysis(
        df, model, scaler, X_clean,
        force_recompute=force_shap
    )

    # Phase 4: LOF
    print("\n  -- Phase 4: Local Outlier Factor --")
    lof_model, lof_scaler, _ = train_lof(
        df, force_retrain=force_retrain
    )
    if lof_model is not None:
        df = run_lof_detection(df, lof_model, lof_scaler)
    else:
        print("  LOF training failed, skipping Phase 4 and 5.")
        save_ml_results(df)
        return df

    # Phase 5: Ensemble voting
    print("\n  -- Phase 5: Ensemble Voting --")
    df = compute_ensemble(df)

    save_ml_results(df)
    return df


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pipeline.storage import load_parquet
    from config import MERGED_DATA_PATH

    print("=" * 60)
    print("AusClimate ML Pipeline -- Phases 1 through 5")
    print("=" * 60)

    df = load_parquet(MERGED_DATA_PATH)
    if df.empty:
        print("No merged data. Run python main.py first.")
    else:
        df_ml = run_ml_pipeline(df, force_retrain=True, force_shap=True)

        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)

        if "ml_anomaly_flag" in df_ml.columns:
            print(f"IF anomalies  : {df_ml['ml_anomaly_flag'].sum():,}")
        if "lof_anomaly_flag" in df_ml.columns:
            print(f"LOF anomalies : {df_ml['lof_anomaly_flag'].sum():,}")
        if "ensemble_anomaly" in df_ml.columns:
            print(f"Consensus     : {df_ml['ensemble_anomaly'].sum():,}")

        print("\nTop 5 highest-confidence anomalies (ensemble):")
        if "ensemble_anomaly" in df_ml.columns:
            ens_cols = [c for c in [
                "station_name", "temperature_c", "humidity_pct",
                "ml_anomaly_score", "lof_anomaly_score",
                "ensemble_severity", "ml_primary_driver"
            ] if c in df_ml.columns]
            top5 = (
                df_ml[df_ml["ensemble_anomaly"] == True][ens_cols]
                .sort_values("ml_anomaly_score")
                .head(5)
            )
            for _, row in top5.iterrows():
                print(f"\n  Station: {row.get('station_name','?')}")
                print(f"  IF score : {row.get('ml_anomaly_score','?')}")
                print(f"  LOF score: {row.get('lof_anomaly_score','?')}")
                print(f"  Severity : {row.get('ensemble_severity','?')}")
                print(f"  Driver   : {row.get('ml_primary_driver','?')}")