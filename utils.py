import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import joblib

ARCHIVE_BASE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_BASE = "https://api.open-meteo.com/v1/forecast"

def requests_retry_session(retries=5, backoff_factor=0.5, status_forcelist=(500, 502, 503, 504)):
    session = requests.Session()
    retry = Retry(total=retries, read=retries, connect=retries, backoff_factor=backoff_factor,
                  status_forcelist=status_forcelist, raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_hourly_history_archive(lat, lon, start_iso, end_iso, timezone="UTC"):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_iso[:10],
        "end_date": end_iso[:10],
        "hourly": "temperature_2m,relativehumidity_2m,pressure_msl,windspeed_10m",
        "timezone": timezone
    }
    r = requests_retry_session().get(ARCHIVE_BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "hourly" not in data:
        return pd.DataFrame()
    times = pd.to_datetime(data["hourly"]["time"])
    df = pd.DataFrame({
        "time": times,
        "temp": data["hourly"].get("temperature_2m", [None]*len(times)),
        "rh": data["hourly"].get("relativehumidity_2m", [None]*len(times)),
        "press": data["hourly"].get("pressure_msl", [None]*len(times)),
        "wind": data["hourly"].get("windspeed_10m", [None]*len(times))
    })
    return df


def get_past_range_hours(hours):
    end = datetime.utcnow()
    start = end - timedelta(hours=hours)
    return start.isoformat(), end.isoformat()


def fetch_hourly_forecast(lat, lon, hours=48, timezone="UTC"):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "forecast_days": int(hours / 24) + 1,
        "timezone": timezone
    }
    r = requests_retry_session().get(FORECAST_BASE, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    times = pd.to_datetime(data["hourly"]["time"])
    return pd.DataFrame({"time": times, "temp": data["hourly"]["temperature_2m"]}).head(hours)


def hourly_to_daily_maxmin(df_hourly):
    df = df_hourly.copy()
    df["date"] = df["time"].dt.date
    daily = df.groupby("date")["temp"].agg(["max", "min"]).reset_index().rename(columns={"max": "temp_max", "min": "temp_min"})
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def get_location_name(lat, lon):
    try:
        r = requests.get("https://nominatim.openstreetmap.org/reverse",
                         params={"lat": lat, "lon": lon, "format": "jsonv2"},
                         headers={"User-Agent": "ml-weather"}, timeout=5)
        if r.status_code == 200:
            return r.json().get("display_name", f"{lat:.4f},{lon:.4f}")
    except Exception:
        pass
    return f"{lat:.4f},{lon:.4f}"


def load_hourly_model_and_predict(path, df_recent, input_lags=48, horizon=24):
    """Load XGBoost hourly models saved in model_hourly_xgb.joblib and predict next `horizon` temps."""
    m = joblib.load(path)
    models = m["hourly_models"]
    feat_cols = m["feat_cols"]

    df = df_recent.copy().sort_values("time").reset_index(drop=True)
    df["hour"] = df["time"].dt.hour
    df["doy"] = df["time"].dt.dayofyear
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["sin_doy"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    df["temp_roll_3"] = df["temp"].rolling(3, min_periods=1).mean()
    df["temp_roll_6"] = df["temp"].rolling(6, min_periods=1).mean()
    df["temp_roll_24"] = df["temp"].rolling(24, min_periods=1).mean()
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"temp_lag_{lag}"] = df["temp"].shift(lag)
    df = df.dropna().reset_index(drop=True)

    block = df.loc[len(df) - input_lags:len(df) - 1, feat_cols].values.flatten().reshape(1, -1)
    preds = [float(models[h].predict(block)[0]) for h in range(horizon)]
    return preds


def load_daily_models_and_predict(path, df_hourly_long, input_lags_days=60, horizon_days=30):
    """Load daily_xgb models and predict 30-day max temps."""
    m = joblib.load(path)
    models = m["daily_max_models"]  # list of models per day
    # build daily table
    daily = hourly_to_daily_maxmin(df_hourly_long).sort_values("date").reset_index(drop=True)
    past = daily.tail(input_lags_days)[["temp_max"]].values.flatten()
    last_date = daily["date"].iloc[-1]
    doy = last_date.dayofyear
    feat = np.concatenate([past, [np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]])
    preds = [float(models[h].predict(feat.reshape(1, -1))[0]) for h in range(horizon_days)]
    dates = [last_date + timedelta(days=i + 1) for i in range(horizon_days)]
    return pd.DataFrame({"date": dates, "pred_max": preds})

def fetch_current_weather_code(lat, lon):
    """
    Uses Open-Meteo current_weather to retrieve current temp and weathercode.
    Returns (temp_c, weathercode)
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "current_weather": True,
        "timezone": "UTC"
    }
    r = requests_retry_session().get(url, params=params, timeout=10)
    r.raise_for_status()
    j = r.json()
    cw = j.get("current_weather", {})
    temp = cw.get("temperature")
    code = cw.get("weathercode", 0)
    return temp, code

def map_weathercode_to_condition(code):
    """
    Map Open-Meteo weathercode to simple condition categories used by UI.
    Categories: clear, cloudy, rain, snow, fog, night
    """
    # Open-Meteo weather codes: https://open-meteo.com/en/docs (0 clear, 1-3 partly-cloudy, 45-48 fog,61-67 rain,71-77 snow, 80-82 showers,95 thunder)
    if code == 0:
        return "clear"
    if code in (1,2,3):
        return "cloudy"
    if 45 <= code <= 48:
        return "fog"
    if 61 <= code <= 67 or 80 <= code <= 82 or 95 <= code <= 99:
        return "rain"
    if 71 <= code <= 77:
        return "snow"
    # fallback: consider night if between 18:00-06:00 UTC (simple heuristic)
    hr = datetime.utcnow().hour
    if hr >= 18 or hr < 6:
        return "night"
    return "clear"
    








