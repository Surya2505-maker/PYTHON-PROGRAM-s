import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import requests
from xgboost import XGBRegressor

ARCHIVE_BASE = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = ["temperature_2m"]

def fetch_hourly_archive(lat, lon, start_iso, end_iso):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_iso[:10],
        "end_date": end_iso[:10],
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC"
    }
    r = requests.get(ARCHIVE_BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "hourly" not in data:
        return pd.DataFrame()
    times = pd.to_datetime(data["hourly"]["time"])
    temps = data["hourly"]["temperature_2m"]
    return pd.DataFrame({"time": times, "temp": temps})

def get_past_range(days_back):
    end = datetime.utcnow()
    start = end - timedelta(days=days_back)
    return start.isoformat(), end.isoformat()

def aggregate_daily_max(df_hourly):
    df_hourly["date"] = df_hourly["time"].dt.date
    daily = df_hourly.groupby("date")["temp"].max().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily

def featurize_daily(df):
    df = df.sort_values("date").reset_index(drop=True)
    df["doy"] = df["date"].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    for lag in range(1, 61):  # 60-day lags
        df[f"temp_max_lag_{lag}"] = df["temp"].shift(lag)
    df = df.dropna().reset_index(drop=True)
    return df

def create_daily_samples(df, input_lags=60, horizon=30):
    feat_cols = [f"temp_max_lag_{lag}" for lag in range(input_lags,0,-1)] + ["sin_doy", "cos_doy"]
    X, y = [], []
    for i in range(input_lags, len(df) - horizon):
        X.append(df.loc[i, feat_cols].values)
        y.append(df.loc[i:i+horizon-1, "temp"].values)
    return np.array(X), np.array(y), feat_cols

def train_daily_model(lat, lon, days_history=365, input_lags=60, horizon=30, out_dir="models"):
    print("Fetching hourly data...")
    start_iso, end_iso = get_past_range(days_history)
    df_hourly = fetch_hourly_archive(lat, lon, start_iso, end_iso)
    if df_hourly.empty:
        raise RuntimeError("No data fetched.")

    print("Aggregating daily max temps...")
    df_daily = aggregate_daily_max(df_hourly)

    print("Featurizing daily data...")
    df_feat = featurize_daily(df_daily)
    X, y, feat_cols = create_daily_samples(df_feat, input_lags, horizon)

    print(f"Training {horizon} XGBoost daily max temperature models...")
    models = []
    for h in range(horizon):
        model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=42)
        model.fit(X, y[:,h])
        print(f" Trained horizon day +{h+1}")
        models.append(model)

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model_daily_xgb.joblib")
    joblib.dump({"models": models, "feat_cols": feat_cols, "input_lags": input_lags, "horizon": horizon}, model_path)
    print(f"Saved model to {model_path}")
    return model_path

def predict_next_30_days(model_path, df_hourly_recent, input_lags=60):
    m = joblib.load(model_path)
    models = m["models"]
    feat_cols = m["feat_cols"]
    horizon = m["horizon"]

    df_daily = aggregate_daily_max(df_hourly_recent)
    df_feat = featurize_daily(df_daily)
    if len(df_feat) < 1:
        raise RuntimeError("Not enough recent daily data for prediction.")

    X_pred = df_feat.loc[len(df_feat)-1, feat_cols].values.reshape(1, -1)
    preds = [float(models[h].predict(X_pred)[0]) for h in range(horizon)]

    last_date = df_daily["date"].max()
    pred_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
    return pd.DataFrame({"date": pred_dates, "pred_max_temp": preds})

if __name__ == "__main__":
    LAT, LON = 11.1128, 77.3460  # NYC example

    # Train daily model for 30-day forecast with 1 year data
    model_file = train_daily_model(LAT, LON, days_history=365)

    # Fetch recent 90 days hourly data for predictions input
    start_iso, end_iso = get_past_range(90)
    recent_hourly = fetch_hourly_archive(LAT, LON, start_iso, end_iso)

    # Predict next 30 days max temperatures
    df_pred = predict_next_30_days(model_file, recent_hourly)
    print("30-day max temperature predictions:")
    print(df_pred)
