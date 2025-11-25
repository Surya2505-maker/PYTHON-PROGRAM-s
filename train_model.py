import os
import joblib
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

# XGBoost sklearn wrapper
from xgboost import XGBRegressor

# HTTP retry helpers
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = ["temperature_2m", "relativehumidity_2m", "pressure_msl", "windspeed_10m"]


def requests_retry_session(retries=5, backoff_factor=0.5, status_forcelist=(500, 502, 503, 504)):
    session = requests.Session()
    retry = Retry(total=retries, read=retries, connect=retries, backoff_factor=backoff_factor,
                  status_forcelist=status_forcelist, raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_years(lat, lon, years=5):
    """Download hourly archive data for N years (chunked by 365 days)."""
    end = datetime.utcnow()
    start = end - timedelta(days=years * 365)
    dfs = []
    p = start
    sess = requests_retry_session()
    while p < end:
        chunk_start = p.date().isoformat()
        chunk_end = min((p + timedelta(days=365)).date(), end.date()).isoformat()
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": chunk_start,
            "end_date": chunk_end,
            "hourly": ",".join(HOURLY_VARS),
            "timezone": "UTC",
        }
        print(f"Fetching {chunk_start} -> {chunk_end}")
        r = sess.get(BASE_URL, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        if "hourly" in data:
            times = pd.to_datetime(data["hourly"]["time"])
            df = pd.DataFrame({
                "time": times,
                "temp": data["hourly"].get("temperature_2m", [None] * len(times)),
                "rh": data["hourly"].get("relativehumidity_2m", [None] * len(times)),
                "press": data["hourly"].get("pressure_msl", [None] * len(times)),
                "wind": data["hourly"].get("windspeed_10m", [None] * len(times)),
            })
            dfs.append(df)
        p += timedelta(days=365)
    if not dfs:
        return pd.DataFrame()
    df_all = pd.concat(dfs).dropna().reset_index(drop=True)
    return df_all


def featurize_hourly(df):
    df = df.sort_values("time").reset_index(drop=True)
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

    return df.dropna().reset_index(drop=True)


def build_hourly_samples(df, input_lags=48, forecast_h=24):
    feat_cols = [c for c in df.columns if c not in ("time",)]
    X, ys = [], []
    n = len(df)
    for i in range(input_lags, n - forecast_h + 1):
        block = df.loc[i - input_lags:i - 1, feat_cols].values.flatten()
        X.append(block)
        ys.append(df.loc[i:i + forecast_h - 1, "temp"].values)
    return np.array(X), np.array(ys), feat_cols


def build_daily_dataset(df_hourly, input_lags_days=60, forecast_days=30):
    df = df_hourly.copy()
    df["date"] = df["time"].dt.date
    daily = df.groupby("date")["temp"].agg(["max"]).reset_index().rename(columns={"max": "temp_max"})
    daily["date"] = pd.to_datetime(daily["date"])
    rows, y = [], []
    for i in range(input_lags_days, len(daily) - forecast_days + 1):
        past = daily.loc[i - input_lags_days:i - 1, ["temp_max"]].values.flatten()
        doy = daily.loc[i - 1, "date"].dayofyear
        features = np.concatenate([past, [np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]])
        rows.append(features)
        y.append(daily.loc[i:i + forecast_days - 1, "temp_max"].values)
    if not rows:
        return None, None, None
    return np.array(rows), np.array(y), daily


def train_xgb_models(X_tr, y_tr, X_val, y_val, n_estimators=500):
    H = y_tr.shape[1]
    models = []
    for h in range(H):
        print(f"Training XGBoost model for horizon +{h+1}")
        m = XGBRegressor(n_estimators=n_estimators, learning_rate=0.05, max_depth=6,
                         objective="reg:squarederror", tree_method="auto", n_jobs=-1, random_state=42)
        try:
            m.fit(X_tr, y_tr[:, h], eval_set=[(X_val, y_val[:, h])], early_stopping_rounds=50, verbose=False)
        except TypeError:
            m.fit(X_tr, y_tr[:, h])
        models.append(m)
    return models


def train_models(lat, lon, years=5, input_lags=48, hourly_h=24, daily_days=30, out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)

    print("Fetching hourly archive data...")
    df = fetch_years(lat, lon, years)
    if df.empty:
        raise RuntimeError("No archive data downloaded.")

    print("Featurizing hourly...")
    df_feat = featurize_hourly(df)
    X, y_hourly, feat_cols = build_hourly_samples(df_feat, input_lags=input_lags, forecast_h=hourly_h)

    split_idx = int(0.9 * len(X))
    X_tr, X_val = X[:split_idx], X[split_idx:]
    y_tr, y_val = y_hourly[:split_idx], y_hourly[split_idx:]

    print("Training hourly XGBoost models (24 horizons)...")
    hourly_models = train_xgb_models(X_tr, y_tr, X_val, y_val, n_estimators=400)

    preds = np.column_stack([m.predict(X_val) for m in hourly_models])

    #  ACCURACY METRICS — HOURLY (MAE, RMSE, MAPE)

    print("\n=== HOURLY 24-HOUR FORECAST ACCURACY ===")
    hourly_mae = []
    hourly_rmse = []
    hourly_mape = []

    for h in range(24):
        true_h = y_val[:, h]
        pred_h = preds[:, h]

        mae_h = np.mean(np.abs(true_h - pred_h))
        rmse_h = np.sqrt(np.mean((true_h - pred_h) ** 2))
        mape_h = np.mean(np.abs((true_h - pred_h) / true_h)) * 100

        hourly_mae.append(mae_h)
        hourly_rmse.append(rmse_h)
        hourly_mape.append(mape_h)

        print(f"+{h+1}h → MAE: {mae_h:.3f} °C   RMSE: {rmse_h:.3f} °C   MAPE: {mape_h:.2f}%")

    avg_hourly_mae = np.mean(hourly_mae)
    print(f"\n24-hour Average MAE: {avg_hourly_mae:.3f} °C")

    # Save hourly model
    joblib.dump({
        "hourly_models": hourly_models,
        "feat_cols": feat_cols,
        "input_lags": input_lags,
        "forecast_hourly": hourly_h,
        "trained_until": df["time"].max().isoformat(),
        "mae_hourly_val": avg_hourly_mae,
        "per_hour_mae": hourly_mae,
        "per_hour_rmse": hourly_rmse,
        "per_hour_mape": hourly_mape
    }, os.path.join(out_dir, "model_hourly_xgb.joblib"))

    print("Saved hourly models to model_hourly_xgb.joblib")

    # DAILY MODEL
    print("Building daily dataset...")
    Xd, yd, daily_table = build_daily_dataset(df, input_lags_days=60, forecast_days=daily_days)
    if Xd is None:
        print("Not enough daily data. Skipping daily model.")
        return True

    split_idx = int(0.9 * len(Xd))
    Xd_tr, Xd_val = Xd[:split_idx], Xd[split_idx:]
    yd_tr, yd_val = yd[:split_idx], yd[split_idx:]

    print("Training daily XGBoost models (30 horizons)...")
    daily_models = train_xgb_models(Xd_tr, yd_tr, Xd_val, yd_val, n_estimators=500)

    #  ACCURACY METRICS — DAILY 30-DAY MAX TEMP

    print("\n=== DAILY 30-DAY FORECAST ACCURACY ===")
    daily_preds = np.column_stack([m.predict(Xd_val) for m in daily_models])

    daily_mae = []
    daily_rmse = []

    for d in range(30):
        true_d = yd_val[:, d]
        pred_d = daily_preds[:, d]

        mae_d = np.mean(np.abs(true_d - pred_d))
        rmse_d = np.sqrt(np.mean((true_d - pred_d) ** 2))

        daily_mae.append(mae_d)
        daily_rmse.append(rmse_d)

        print(f"Day +{d+1} → MAE: {mae_d:.3f} °C   RMSE: {rmse_d:.3f} °C")

    avg_daily_mae = np.mean(daily_mae)
    print(f"\nAverage 30-day MAE: {avg_daily_mae:.3f} °C")

    joblib.dump({
        "daily_max_models": daily_models,
        "input_lags_days": 60,
        "forecast_days": daily_days,
        "trained_until": df["time"].max().isoformat(),
        "daily_mae": daily_mae,
        "daily_rmse": daily_rmse,
        "avg_daily_mae": avg_daily_mae
    }, os.path.join(out_dir, "model_daily_xgb.joblib"))

    print("Saved daily models to model_daily_xgb.joblib")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--input", type=int, default=48)
    parser.add_argument("--hourly", type=int, default=24)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--out", type=str, default="models")
    args = parser.parse_args()

    train_models(args.lat, args.lon, years=args.years,
                 input_lags=args.input, hourly_h=args.hourly,
                 daily_days=args.days, out_dir=args.out)
    print("Training complete.")
    