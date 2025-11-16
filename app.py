# app.py
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objs as go

from utils import fetch_hourly_history, fetch_current_weather, get_past_range_hours, create_features
from utils import get_location_name

# 3️⃣ Now you can fetch weather, train ML model, etc. using lat, lon

MODEL_PATH = "rf_weather.joblib"

st.set_page_config(page_title="ML Weather App", layout="wide")

st.title("ML Weather App — Short-term Temperature Forecast")
st.write("Simple baseline: RandomForest forecasting next hours using recent history (Open-Meteo API).")

with st.sidebar:
    st.header("Location & Options")
    lat = st.number_input("Latitude", value=13.0827, format="%.6f")
    lon = st.number_input("Longitude", value=80.2707, format="%.6f")
    hours_history = st.selectbox("History hours to fetch (training)", [168, 336, 720], index=0)
    retrain = st.button("Retrain model now (uses selected history)")

if retrain:
    st.info("Retraining — this may take some minutes. Run `python train_model.py` in shell for better control.")
    st.stop()

# Load model
try:
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    cols = saved["columns"]
except Exception as e:
    st.warning(f"Model not found ({e}). Please run `python train_model.py --lat {lat} --lon {lon}` to create model.")
    model = None
    cols = None

# Fetch recent data for display & forecasting
end = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
start = end - timedelta(hours=72)
start_iso = start.isoformat() + "Z"
end_iso = end.isoformat() + "Z"
st.title("🌤 Weather Forecast App")
# 2️⃣ Get location name
place_name = get_location_name(lat, lon)
st.subheader(f"📍 Location: {place_name}")


with st.spinner("Fetching recent weather history..."):
    hist = fetch_hourly_history(lat, lon, start_iso, end_iso, timezone="UTC")

st.subheader("Current / Recent Data")
st.write(f"Location: {lat:.4f}, {lon:.4f}")
st.line_chart(hist.set_index("time")["temp"].rename("Temperature (°C)"))

current = None
try:
    current = fetch_current_weather(lat, lon, timezone="UTC")
except Exception:
    current = None

if current:
    st.metric("Current Temperature (°C)", value=current.get("temperature"))

# Forecasting next 24 hours using the model
if model is not None:
    # Prepare features using the last rows of history
    X, y, times = create_features(hist)
    last_row = X.iloc[-1:].copy()
    preds_hours = 24
    forecast_times = []
    forecast_vals = []
    # iterative multi-step forecast: predict next hour, append as lag, repeat
    current_features = last_row.copy()
    for i in range(preds_hours):
        p = model.predict(current_features[cols])[0]
        forecast_vals.append(p)
        next_time = hist["time"].max() + pd.Timedelta(hours=i+1)
        forecast_times.append(next_time)
        # shift lag features: drop largest lag, insert new p as lag_1, shift others
        for c in [c for c in current_features.columns if c.startswith("lag_")]:
            # convert col name e.g. lag_1 -> 1
            pass
        # build new features dict manually
        new = current_features.copy()
        # shift lags
        lag_cols = [c for c in new.columns if c.startswith("lag_")]
        lag_nums = sorted([int(c.split("_")[1]) for c in lag_cols])
        # shift descending: highest lag gets previous lower lag
        for n in reversed(lag_nums):
         col = f"lag_{n}"
         if n == 1:
                # The newest predicted value becomes lag_1
          new[col] = p
         else:
          prev_col = f"lag_{n-1}"
                # Only shift if previous lag column exists
          if prev_col in new.columns:
            new[col] = new[prev_col].values

        # update hour and dow for the new timestamp
        new["hour"] = [next_time.hour]
        new["dow"] = [next_time.dayofweek]
        # rh/pressure/wind left unchanged (no predictions) — keep last observed
        current_features = new

    # show forecast
    df_fore = pd.DataFrame({"time": forecast_times, "temp_forecast": forecast_vals})
    st.subheader("ML Forecast (next 24 hours)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["time"], y=hist["temp"], name="Observed"))
    fig.add_trace(go.Scatter(x=df_fore["time"], y=df_fore["temp_forecast"], name="Forecast"))
    fig.update_layout(xaxis_title="Time (UTC)", yaxis_title="Temperature (°C)")
    st.plotly_chart(fig, use_container_width=True)

    # table
    st.dataframe(df_fore.set_index("time").round(2))

else:
    st.info("Model missing. Train model with `python train_model.py` and reload.")

st.write("""
### Notes
This is a simple baseline:

- Model uses lag features (previous hours), hour-of-day, day-of-week, and a few meteorological features when available.  
- Iterative forecasting is naive and accumulates error over steps.  
- For multi-hour forecasting consider direct multi-output models or sequence models (RNN/Transformer).
""")
