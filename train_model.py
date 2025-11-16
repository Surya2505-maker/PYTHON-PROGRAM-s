# train_model.py
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from utils import fetch_hourly_history, get_past_range_hours, create_features
from datetime import datetime, timedelta

def main(lat, lon, hours_back=720, model_path="rf_weather.joblib"):
    # fetch last hours_back hours
    end = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=hours_back)
    start_iso = start.isoformat() + "Z"
    end_iso = end.isoformat() + "Z"

    print(f"Fetching data from {start_iso} to {end_iso} ...")
    df = fetch_hourly_history(lat, lon, start_iso, end_iso, timezone="UTC")
    print("Rows fetched:", len(df))
    X, y, times = create_features(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, shuffle=False)
    print("Training size:", len(X_train), "Validation size:", len(X_val))

    model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    print(f"Validation MAE: {mae:.3f} °C")

    # save model and column order
    joblib.dump({"model": model, "columns": X.columns.tolist()}, model_path)
    print("Saved model to", model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--hours", type=int, default=720)
    parser.add_argument("--out", type=str, default="rf_weather.joblib")
    args = parser.parse_args()
    main(args.lat, args.lon, hours_back=args.hours, model_path=args.out)
