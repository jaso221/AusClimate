"""
forecasting/forecaster.py
Short-term forecasting using Facebook Prophet.
Forecasts temperature, humidity, wind speed per station, 24 hours ahead.
"""

import os
import pandas as pd
import logging
import warnings

warnings.filterwarnings("ignore", module="prophet")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

from prophet import Prophet

from concurrent.futures import ProcessPoolExecutor, as_completed

from config import (
    STATION_LIST_PATH,
    BOM_HISTORY_PATH,
    BARRA2_CACHE_PATH,
    FORECAST_OUTPUT_PATH,
)

logger = logging.getLogger(__name__)

FORECAST_VARIABLES = {
    "temperature_c":   {"floor": -10, "cap": 55},
    "humidity_pct":    {"floor": 0,   "cap": 100},
    "wind_speed_kmh":  {"floor": 0,   "cap": 200},
    "rainfall_mm":     {"floor": 0,   "cap": None},
}

FORECAST_HORIZON_HOURS = 24
FORECAST_CACHE_HOURS = 24
MIN_HISTORY_POINTS = 48
MIN_HISTORY_DAYS_FOR_SEASONALITY = 2

RISK_THRESHOLDS = {
    "temperature_c":  {"high": 40, "low": 0},
    "humidity_pct":   {"high": 95, "low": 10},
    "wind_speed_kmh": {"high": 60, "low": None},
}


def _safe_read(path):
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def load_station_history(station_name: str) -> pd.DataFrame:
    """
    Pull accumulated BOM history for one station. BARRA2 is skipped as a
    seed source here since its schema (barra2_temp_c etc, and whether it
    carries a timestamp column) differs from live BOM columns -- verify
    data/processed/barra2_baseline.parquet before wiring it in as a seed.
    """
    hist = _safe_read(BOM_HISTORY_PATH)
    if hist.empty or "station_name" not in hist.columns:
        return pd.DataFrame()

    hist = hist[hist["station_name"] == station_name].copy()
    if hist.empty:
        return pd.DataFrame()

    hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce")
    hist = hist.dropna(subset=["timestamp"]).drop_duplicates(subset="timestamp")
    return hist.sort_values("timestamp")


def fit_and_forecast(history: pd.DataFrame, variable: str) -> pd.DataFrame | None:
    series = history[["timestamp", variable]].dropna() if variable in history.columns else pd.DataFrame()
    series = series.rename(columns={"timestamp": "ds", variable: "y"})

    if len(series) < MIN_HISTORY_POINTS:
        return None

    bounds = FORECAST_VARIABLES[variable]
    growth = "linear"

    if bounds["floor"] is not None and bounds["cap"] is not None:
        series["floor"] = bounds["floor"]
        series["cap"] = bounds["cap"]
        growth = "logistic"

    span_days = (series["ds"].max() - series["ds"].min()).days
    daily_seasonality = span_days >= MIN_HISTORY_DAYS_FOR_SEASONALITY

    model = Prophet(
        growth=growth,
        daily_seasonality=daily_seasonality,
        weekly_seasonality=False,
        yearly_seasonality=False,
        interval_width=0.80,
        changepoint_prior_scale=0.05,
    )

    try:
        model.fit(series)
    except Exception as e:
        logger.warning(f"Prophet fit failed for {variable}: {e}")
        return None

    future = model.make_future_dataframe(periods=FORECAST_HORIZON_HOURS, freq="h")
    if growth == "logistic":
        future["floor"] = bounds["floor"]
        future["cap"] = bounds["cap"]

    forecast = model.predict(future)
    cutoff = series["ds"].max()
    forecast = forecast[forecast["ds"] > cutoff]

    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={
        "ds": "forecast_datetime",
        "yhat": "predicted_value",
        "yhat_lower": "lower_bound",
        "yhat_upper": "upper_bound",
    })
    result["variable"] = variable
    return result

def _forecast_station(station_name: str, history: pd.DataFrame) -> list[pd.DataFrame]:
    results = []
    for variable in FORECAST_VARIABLES:
        result = fit_and_forecast(history, variable)
        if result is None:
            continue
        result["station_name"] = station_name
        results.append(result)
    return results

def flag_risk_breaches(forecast_df: pd.DataFrame) -> pd.DataFrame:
    def check_row(row):
        thresholds = RISK_THRESHOLDS.get(row["variable"], {})
        high = thresholds.get("high")
        low = thresholds.get("low")
        if high is not None and row["predicted_value"] >= high:
            return "high"
        if low is not None and row["predicted_value"] <= low:
            return "low"
        return "normal"

    forecast_df["risk_flag"] = forecast_df.apply(check_row, axis=1)
    return forecast_df


def run_forecasting(station_names: list[str] | None = None, max_workers: int | None = None) -> pd.DataFrame:
    history_all = _safe_read(BOM_HISTORY_PATH)
    if history_all.empty or "station_name" not in history_all.columns:
        logger.warning("No BOM history found. Run the pipeline a few times first.")
        return pd.DataFrame()

    history_all["timestamp"] = pd.to_datetime(history_all["timestamp"], errors="coerce")
    history_all = history_all.dropna(subset=["timestamp"])

    if station_names is None:
        station_list = _safe_read(STATION_LIST_PATH)
        if station_list.empty or "station_name" not in station_list.columns:
            logger.warning("No station list found. Run BOM ingestion first.")
            return pd.DataFrame()
        station_names = station_list["station_name"].unique().tolist()

    wanted = set(station_names)
    tasks = []
    for name, grp in history_all.groupby("station_name"):
        if name not in wanted:
            continue
        grp = grp.drop_duplicates(subset="timestamp").sort_values("timestamp")
        tasks.append((name, grp))

    skipped = len(station_names) - len(tasks)
    all_results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_forecast_station, name, hist): name
            for name, hist in tasks
        }
        for future in as_completed(futures):
            try:
                all_results.extend(future.result())
            except Exception as e:
                logger.warning(f"Forecast worker failed for {futures[future]}: {e}")

    if not all_results:
        logger.warning("Forecasting produced no results -- bom_history.parquet needs more accumulated runs.")
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    combined = flag_risk_breaches(combined)
    combined = combined[[
        "station_name", "variable", "forecast_datetime",
        "predicted_value", "lower_bound", "upper_bound", "risk_flag",
    ]]

    combined.to_parquet(FORECAST_OUTPUT_PATH, index=False)
    logger.info(
        f"Forecasting complete: {combined['station_name'].nunique()} stations, "
        f"{skipped} skipped (insufficient history)"
    )
    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_forecasting()