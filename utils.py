# utils.py
import requests
import pandas as pd
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim

def get_location_name(lat, lon):
    """Return a readable place name (city, country) for given coordinates."""
    geolocator = Nominatim(user_agent="weather_app")
    location = geolocator.reverse((lat, lon), language="en", timeout=10)
    if location and location.address:
        return location.address
    else:
        return f"Lat: {lat}, Lon: {lon}"


OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"

def fetch_hourly_history(lat, lon, start_iso, end_iso, timezone="UTC"):
    """
    Fetch hourly temperature (2m) between start_iso and end_iso using Open-Meteo.
    start_iso/end_iso example: "2025-11-01T00:00:00Z"
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relativehumidity_2m,pressure_msl,windspeed_10m",
        "start": start_iso,
        "end": end_iso,
        "timezone": timezone
    }
    r = requests.get(OPEN_METEO_BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    # Build DataFrame
    times = data["hourly"]["time"]
    df = pd.DataFrame({
        "time": pd.to_datetime(times),
        "temp": data["hourly"]["temperature_2m"],
        "rh": data["hourly"].get("relativehumidity_2m", [None]*len(times)),
        "pressure": data["hourly"].get("pressure_msl", [None]*len(times)),
        "wind": data["hourly"].get("windspeed_10m", [None]*len(times))
    })
    return df

def fetch_current_weather(lat, lon, timezone="UTC"):
    """
    Quick fetch of current hour values using short-range forecast endpoint.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relativehumidity_2m,pressure_msl,windspeed_10m",
        "current_weather": "true",
        "timezone": timezone
    }
    r = requests.get(OPEN_METEO_BASE, params=params, timeout=20)
    r.raise_for_status()
    return r.json().get("current_weather", None)

def get_past_range_hours(hours_back=168):
    """Return start_iso and end_iso strings for past `hours_back` hours until now (UTC)."""
    end = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=hours_back)
    # Open-Meteo expects ISO without 'Z' if timezone set, but we'll use ISO with 'Z' (UTC)
    return start.isoformat() + "Z", end.isoformat() + "Z"

def create_features(df, lags=[1,2,3,6,12,24]):
    """
    Create lag features from temperature column.
    df must be sorted by time ascending and have 'temp' column.
    Returns X (features), y (target temp for next hour), times for y.
    """
    df = df.sort_values("time").reset_index(drop=True)
    for lag in lags:
        df[f"lag_{lag}"] = df["temp"].shift(lag)
    # add hour of day, day of week
    df["hour"] = df["time"].dt.hour
    df["dow"] = df["time"].dt.dayofweek
    # target is temp 1 hour ahead
    df["target"] = df["temp"].shift(-1)
    df = df.dropna().reset_index(drop=True)
    feature_cols = [c for c in df.columns if c.startswith("lag_")] + ["hour", "dow", "rh", "pressure", "wind"]
    X = df[feature_cols]
    y = df["target"]
    times = df["time"]
    return X, y, times
