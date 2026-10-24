"""
forecasting/hazard_index.py
Derived natural-hazard risk (bushfire, flash flood, heatwave) built from
BOM observation history and the Prophet forecast in forecaster.py. No new
data sources needed, everything here comes from temperature, humidity,
wind speed and rainfall already collected.

Methods used:
- Bushfire: Keetch-Byram Drought Index (KBDI) -> Griffith's Drought Factor
  -> McArthur Mk5 Forest Fire Danger Index (FFDI). The published
  Australian method, not a custom formula.
- Flash flood: a simplified indicative score (rainfall intensity + soil
  wetness from KBDI). Not an official flood forecast, real flood
  forecasting needs catchment and streamflow data this project doesn't
  have. Treat it as a localised heavy-rainfall signal only.
- Heatwave: a one-day adaptation of the Excess Heat Factor (EHF), the
  method BOM and health agencies use, comparing the forecast period's
  mean temperature against a seasonal threshold and recent
  acclimatisation.
"""

import pandas as pd
import numpy as np

# ── Tunable constants ───────────────────────────────────────────────────────

MIN_DAYS_FOR_HAZARD = 5      # minimum days of daily history required
KBDI_MAX            = 203.2  # mm-equivalent scale, standard KBDI ceiling
RAIN_DAY_THRESHOLD  = 2.0    # mm -- what counts as a "rain day" for drought factor
RAIN_RUNOFF_MM      = 5.0    # first 5mm of a rain event assumed lost to runoff

FFDI_BANDS = [
    (12,  "Low-Moderate"),
    (24,  "High"),
    (49,  "Very High"),
    (74,  "Severe"),
    (99,  "Extreme"),
    (float("inf"), "Catastrophic"),
]

FLOOD_BANDS = [
    (2.5, "Low"),
    (5.0, "Moderate"),
    (7.5, "High"),
    (float("inf"), "Extreme"),
]

EHF_BANDS = [
    (0,            "No Heatwave"),
    (2,            "Low-Intensity Heatwave"),
    (4,            "Severe Heatwave"),
    (float("inf"), "Extreme Heatwave"),
]


def _band(value: float, bands: list) -> str:
    for threshold, label in bands:
        if value <= threshold:
            return label
    return bands[-1][1]


# ── Daily aggregation ────────────────────────────────────────────────────────

def _daily_aggregates(history: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse sub-daily BOM observations into one row per day.

    rainfall_mm from BOM is a running total since 9am (BOM's "rain_trace"
    field), not a per-reading amount, so each day's total is the MAX value
    seen that rain-day, never a sum of readings. Temperature uses a plain
    calendar-day max/mean.
    """
    if "temperature_c" not in history.columns or "timestamp" not in history.columns:
        return pd.DataFrame()

    h = history.dropna(subset=["timestamp"]).copy()
    if h.empty:
        return pd.DataFrame()

    h["timestamp"] = pd.to_datetime(h["timestamp"], errors="coerce")
    h = h.dropna(subset=["timestamp"])
    if h.empty:
        return pd.DataFrame()

    h["calendar_date"] = h["timestamp"].dt.date
    h["rain_date"] = (h["timestamp"] - pd.Timedelta(hours=9)).dt.date

    temp_daily = h.groupby("calendar_date")["temperature_c"].agg(
        temp_max="max", temp_mean="mean"
    )

    if "rainfall_mm" in h.columns:
        rain_daily = h.groupby("rain_date")["rainfall_mm"].max().rename("rainfall_total")
    else:
        rain_daily = pd.Series(dtype=float, name="rainfall_total")

    daily = temp_daily.join(rain_daily, how="left")
    daily["rainfall_total"] = daily["rainfall_total"].fillna(0.0)
    daily = daily.dropna(subset=["temp_max"])
    daily.index.name = "date"
    return daily.reset_index().sort_values("date")


# ── KBDI / Drought Factor ────────────────────────────────────────────────────

def _kbdi_step(q_prev: float, temp_max_c: float, rainfall_total_mm: float,
               mean_annual_rainfall_mm: float) -> float:
    """One day's KBDI update. q_prev and the return value are on the
    0-203.2mm scale."""
    net_rain = max(0.0, rainfall_total_mm - RAIN_RUNOFF_MM)
    q_after_rain = max(0.0, q_prev - net_rain)

    r = max(mean_annual_rainfall_mm, 1.0)  # guard divide-by-zero
    numerator = (
        (KBDI_MAX - q_after_rain)
        * (0.968 * np.exp(0.0875 * temp_max_c + 1.5552) - 8.30)
        * 1e-3
    )
    denominator = 1 + 10.88 * np.exp(-0.001736 * r)
    delta = numerator / denominator

    q_today = q_after_rain + delta
    return float(np.clip(q_today, 0.0, KBDI_MAX))


def compute_kbdi_series(daily: pd.DataFrame, mean_annual_rainfall_mm: float) -> pd.DataFrame:
    """Runs the KBDI recursion across a station's full daily history."""
    daily = daily.sort_values("date").reset_index(drop=True)
    kbdi_values = []
    q = 0.0
    for _, row in daily.iterrows():
        q = _kbdi_step(q, row["temp_max"], row["rainfall_total"], mean_annual_rainfall_mm)
        kbdi_values.append(q)
    daily = daily.copy()
    daily["kbdi"] = kbdi_values
    return daily


def _days_since_rain(daily: pd.DataFrame) -> tuple:
    """Scans backward from the most recent day for the last rain-day
    (>= RAIN_DAY_THRESHOLD mm). Returns (days_since_rain, that day's total)."""
    rows = daily.sort_values("date", ascending=False)
    for i, (_, row) in enumerate(rows.iterrows()):
        if row["rainfall_total"] >= RAIN_DAY_THRESHOLD:
            return i, float(row["rainfall_total"])
    return len(daily), 0.0


def compute_drought_factor(kbdi: float, days_since_rain: int, last_rain_mm: float) -> float:
    n = max(days_since_rain, 0)
    p = max(last_rain_mm, 0.0)
    numerator = 0.191 * (kbdi + 104) * (n + 1) ** 1.5
    denominator = 3.52 * (n + 1) ** 1.5 + p - 1
    denominator = denominator if denominator > 0 else 1.0
    df = numerator / denominator
    return float(np.clip(df, 0.0, 10.0))


# ── FFDI (bushfire) ──────────────────────────────────────────────────────────

def compute_ffdi(temp_c: float, humidity_pct: float, wind_kmh: float,
                  drought_factor: float) -> float:
    df_safe = max(drought_factor, 1.0)  # ln(DF) undefined at 0
    ffdi = 2 * np.exp(
        -0.45
        + 0.987 * np.log(df_safe)
        - 0.0345 * humidity_pct
        + 0.0338 * temp_c
        + 0.0234 * wind_kmh
    )
    return float(max(ffdi, 0.0))


def classify_ffdi(value: float) -> str:
    return _band(value, FFDI_BANDS)


# ── Flash flood (indicative, not an official forecast) ──────────────────────

def compute_flash_flood_score(forecast_rainfall_mm: float, kbdi: float) -> float:
    """0-10 indicative score. Rainfall intensity is the primary driver,
    existing soil wetness (inverse of KBDI) is a secondary modifier. A
    simplified heuristic, not an official flood forecasting method."""
    intensity = np.clip(max(forecast_rainfall_mm, 0.0) / 50.0, 0.0, 1.0)
    wetness = np.clip((KBDI_MAX - kbdi) / KBDI_MAX, 0.0, 1.0)
    score = 10 * (0.7 * intensity + 0.3 * wetness)
    return float(np.clip(score, 0.0, 10.0))


def classify_flood(value: float) -> str:
    return _band(value, FLOOD_BANDS)


# ── Heatwave (Excess Heat Factor, 24h-forecast adaptation) ──────────────────

def compute_ehf(forecast_temp_mean: float, t95: float, t30_avg: float) -> float:
    ehi_sig = forecast_temp_mean - t95
    ehi_accl = forecast_temp_mean - t30_avg
    return float(ehi_sig * max(1.0, ehi_accl))


def classify_ehf(value: float) -> str:
    return _band(value, EHF_BANDS)


# ── Orchestrator ──────────────────────────────────────────────────────────────

def compute_station_hazards(station_name: str, history_all: pd.DataFrame,
                             forecast_df: pd.DataFrame) -> dict:
    """
    Computes bushfire, flash flood and heatwave risk for one station using
    its accumulated BOM history plus its Prophet forecast. Always returns
    an "available" key, check it before reading the rest.
    """
    out = {"available": False, "reason": "", "station_name": station_name}

    history = history_all[history_all["station_name"] == station_name] \
        if "station_name" in history_all.columns else pd.DataFrame()
    daily = _daily_aggregates(history)

    if len(daily) < MIN_DAYS_FOR_HAZARD:
        out["reason"] = (
            f"Need at least {MIN_DAYS_FOR_HAZARD} days of accumulated "
            f"history for this station to compute drought-based risk. "
            f"Currently have {len(daily)} day(s)."
        )
        return out

    span_days = max((daily["date"].max() - daily["date"].min()).days, 1)
    mean_annual_rainfall = daily["rainfall_total"].sum() / span_days * 365.25
    mean_annual_rainfall = max(mean_annual_rainfall, 50.0)  # sane floor

    daily_kbdi = compute_kbdi_series(daily, mean_annual_rainfall)
    today_kbdi = float(daily_kbdi.iloc[-1]["kbdi"])
    days_since_rain, last_rain_mm = _days_since_rain(daily)
    today_df = compute_drought_factor(today_kbdi, days_since_rain, last_rain_mm)

    station_fc = forecast_df[forecast_df["station_name"] == station_name] \
        if "station_name" in forecast_df.columns else pd.DataFrame()

    if station_fc.empty:
        out["reason"] = "No forecast available for this station yet."
        return out

    wide = station_fc.pivot_table(
        index="forecast_datetime", columns="variable",
        values="predicted_value", aggfunc="first",
    ).reset_index().sort_values("forecast_datetime")

    for col in ["temperature_c", "humidity_pct", "wind_speed_kmh", "rainfall_mm"]:
        if col not in wide.columns:
            wide[col] = np.nan

    rainfall_available = wide["rainfall_mm"].notna().any()
    wide["rainfall_mm"] = wide["rainfall_mm"].clip(lower=0).fillna(0.0)

    core = wide.dropna(subset=["temperature_c", "humidity_pct", "wind_speed_kmh"])
    if core.empty:
        out["reason"] = "Forecast is missing temperature, humidity or wind for this station."
        return out

    # project drought factor one day forward using the forecast window's
    # own daily max temp and rainfall total, held constant across the
    # 24h window since drought factor moves on a daily timescale
    fc_temp_max = float(core["temperature_c"].max())
    fc_rain_total = float(wide["rainfall_mm"].sum())

    tomorrow_kbdi = _kbdi_step(today_kbdi, fc_temp_max, fc_rain_total, mean_annual_rainfall)
    if fc_rain_total >= RAIN_DAY_THRESHOLD:
        tomorrow_days_since_rain, tomorrow_last_rain = 0, fc_rain_total
    else:
        tomorrow_days_since_rain, tomorrow_last_rain = days_since_rain + 1, last_rain_mm
    tomorrow_df = compute_drought_factor(tomorrow_kbdi, tomorrow_days_since_rain, tomorrow_last_rain)

    core = core.copy()
    core["ffdi"] = core.apply(
        lambda r: compute_ffdi(r["temperature_c"], r["humidity_pct"],
                                r["wind_speed_kmh"], tomorrow_df), axis=1)
    core["ffdi_rating"] = core["ffdi"].apply(classify_ffdi)
    core["flood_score"] = core["rainfall_mm"].apply(
        lambda r: compute_flash_flood_score(r, tomorrow_kbdi))
    core["flood_rating"] = core["flood_score"].apply(classify_flood)

    # heatwave is a single value for the whole forecast window, matching
    # the real EHF definition as a period-mean index, not an hourly one
    t95 = float(daily["temp_mean"].quantile(0.95))
    last_30 = daily.sort_values("date").tail(30)
    t30_avg = float(last_30["temp_mean"].mean())
    fc_temp_mean = float(core["temperature_c"].mean())
    ehf = compute_ehf(fc_temp_mean, t95, t30_avg)

    out.update({
        "available": True,
        "today_kbdi": today_kbdi,
        "today_drought_factor": today_df,
        "forecast_drought_factor": tomorrow_df,
        "mean_annual_rainfall_estimate_mm": mean_annual_rainfall,
        "history_days": len(daily),
        "rainfall_forecast_available": bool(rainfall_available),
        "ffdi_hourly": core[["forecast_datetime", "temperature_c", "humidity_pct",
                              "wind_speed_kmh", "ffdi", "ffdi_rating"]],
        "flood_hourly": core[["forecast_datetime", "rainfall_mm",
                               "flood_score", "flood_rating"]],
        "ehf": ehf,
        "ehf_rating": classify_ehf(ehf),
        "forecast_temp_mean": fc_temp_mean,
        "t95": t95,
        "t30_avg": t30_avg,
        "daily_recent": daily[["date", "temp_mean"]].tail(60),
    })
    return out