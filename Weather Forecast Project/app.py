import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import timedelta, datetime
from streamlit_javascript import st_javascript
import plotly.graph_objs as go
from PIL import Image
from utils import (
    fetch_hourly_history_archive,
    fetch_hourly_forecast,
    get_past_range_hours,
    hourly_to_daily_maxmin,
    get_location_name,
    load_hourly_model_and_predict,
    load_daily_models_and_predict,
    fetch_current_weather_code,     # returns (temperature, weathercode)
    map_weathercode_to_condition    # returns "clear", "cloudy", "rain", "snow", "fog", "night"
)

# Paths for models
MODEL_HOURLY = os.path.join("models", "model_hourly_xgb.joblib")
MODEL_DAILY = os.path.join("models", "model_daily_xgb.joblib")

st.set_page_config(page_title="ML Weather Forecasting", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root {
  --card-bg: rgba(255,255,255,0.85);
  --muted: #6b7280;
  --accent: #2563eb;
  --glass: rgba(255,255,255,0.6);
  --shadow: 0 8px 30px rgba(3,10,30,0.12);
  --card-radius: 15px;
}
body, .block-container {
  padding-top: 1rem;
  padding-left: 1rem;
  padding-right: 1rem;
}
/* glass card */
.card {
  background: var(--card-bg);
  border-radius: var(--card-radius);
  padding: 14px;
  box-shadow: var(--shadow);
  color: #061224;
}
/* hourly strip */
.hour-strip { display:flex; gap:8px; overflow-x:auto; padding-bottom:8px; }
.hour-item { min-width:86px; padding:10px; border-radius:10px; text-align:center; background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(247,250,255,0.9)); box-shadow: 0 6px 18px rgba(3,10,30,0.06); }
/* responsive */
@media (max-width: 900px) {
  .hour-item { min-width:72px; padding:8px; }
}
.kv { font-size:20px; font-weight:700; }
.small { color:var(--muted); font-size:13px; }
.header-xx { font-size:22px; font-weight:700; margin:0; padding:0; }
.footer { color:var(--muted); font-size:12px; margin-top:10px; }
</style>
""", unsafe_allow_html=True)

# ----------------- BACKGROUND HERO (responsive to weather) -----------------
def hero_css_for_condition(cond):
    # cond in ["clear","cloudy","rain","snow","fog","night"]
    if cond == "clear":
        bg = "linear-gradient(180deg,#a5d8ff 0%, #e6f4ff 40%, #ffffff 100%);"
        icon = "☀️"
    elif cond == "cloudy":
        bg = "linear-gradient(180deg,#dbe7ff 0%, #f3f6ff 40%, #ffffff 100%);"
        icon = "☁️"
    elif cond == "rain":
        bg = "linear-gradient(180deg,#cfe7ff 0%, #eaf6ff 40%, #ffffff 100%);"
        icon = "🌧️"
    elif cond == "snow":
        bg = "linear-gradient(180deg,#eaf6ff 0%, #ffffff 40%, #ffffff 100%);"
        icon = "❄️"
    elif cond == "fog":
        bg = "linear-gradient(180deg,#ebf0f3 0%, #f7fafc 40%, #ffffff 100%);"
        icon = "🌫️"
    else:
        # night
        bg = "linear-gradient(180deg,#0f172a 0%, #0b1220 50%, #081427 100%); color: #f8fafc;"
        icon = "🌙"
    css = f"""
    <style>
    .hero {{
      background: {bg}
      border-radius: 18px;
      padding:18px;
      margin-bottom: 12px;
        box-shadow: var(--shadow);

    }}
    .hero .big-temp {{ font-size:48px; font-weight:800; margin:0; }}
    .hero .small {{}}
    
    </style>
    """
    return css, icon

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("## Location & Controls")

    # ---------- AUTO GPS BUTTON ----------
    if st.button("📍 Use My Location (GPS)"):
        result = st_javascript("""
        navigator.geolocation.getCurrentPosition(
            (pos) => {
                const coords = {lat: pos.coords.latitude, lon: pos.coords.longitude};
                window.parent.postMessage({ type: 'streamlit:setComponentValue', value: coords }, '*')
            },
            (err) => {
                window.parent.postMessage({ type: 'streamlit:setComponentValue', value: null }, '*')
            }
        )
        """)
        if result:
            st.session_state["lat"] = result["lat"]
            st.session_state["lon"] = result["lon"]
            st.success(f"Detected: {result['lat']:.5f}, {result['lon']:.5f}")

    # default lat/lon if empty
    lat = st.number_input("Latitude", value=st.session_state.get("lat", 11.1127), format="%.6f")
    lon = st.number_input("Longitude", value=st.session_state.get("lon", 77.3407), format="%.6f")

    history_hours = st.select_slider("History (hours)", options=[24,72,240,720,1440], value=720)
    train_models_now = st.button(" Retrain Models (XGBoost)")
    blend_api = st.slider("Blend: ML vs API (ML weight)", 0.0, 1.0, 0.6, 0.05)
    accent_color = st.color_picker("Accent color", "#2563eb")
    st.markdown("---")

# ----------------- TRAIN BUTTON -----------------
if train_models_now:
    st.sidebar.info("Retraining — please wait until 'Training Completed' appears.")
    try:
        from train_model import train_models
        train_models(lat, lon, years=5, input_lags=48, hourly_h=24, daily_days=30, out_dir="models")
        st.sidebar.success("Training completed.")
    except Exception as e:
        st.sidebar.error(f"Training failed: {e}")

# ----------------- GET CURRENT CONDITION -----------------
with st.spinner("Fetching current weather..."):
    try:
        curr_temp, curr_code = fetch_current_weather_code(lat, lon)
        condition = map_weathercode_to_condition(curr_code)
    except Exception:
        # fallback
        condition = "clear"
        curr_temp = None

hero_css, hero_icon = hero_css_for_condition(condition)
st.markdown(hero_css, unsafe_allow_html=True)

# ---------- HERO CARD ----------
st.markdown(f"<div class='hero card'>", unsafe_allow_html=True)
c1, c2 = st.columns([1.2, 2])
with c1:
    if curr_temp is not None:
        st.markdown(f"<div class='big-temp'>{curr_temp:.1f}°C</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='big-temp'>--°C</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small'>Condition: <b>{condition.title()}</b> {hero_icon}</div>", unsafe_allow_html=True)
with c2:
    place = get_location_name(lat, lon)
    st.markdown(f"<div style='text-align:right'><h3 class='header-xx'>{place}</h3><div class='small-muted'>Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</div></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ----------------- FETCH HISTORICAL + API -----------------
with st.spinner("Fetching history & API forecast..."):
    try:
        start_iso, end_iso = get_past_range_hours(history_hours)
        hist_hour = fetch_hourly_history_archive(lat, lon, start_iso, end_iso)
    except Exception as e:
        st.error(f"Archive fetch failed: {e}")
        hist_hour = pd.DataFrame()
    try:
        api_hour = fetch_hourly_forecast(lat, lon, hours=48)
    except Exception:
        api_hour = pd.DataFrame()

# ----------------- TOP METRICS -----------------
col1, col2, col3, col4 = st.columns([1,1,1,1])
with col1:
    if not hist_hour.empty:
        st.markdown("<div class='card'><div class='kv'>"+f"{hist_hour['temp'].iloc[-1]:.1f}°C"+"</div><div class='small'>Latest (archive)</div></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='card'><div class='kv'>--°C</div><div class='small'>Latest (archive)</div></div>", unsafe_allow_html=True)
with col2:
    # ML hourly model availability
    has_hourly = os.path.exists(MODEL_HOURLY)
    st.markdown(f"<div class='card'><div class='kv'>{'Ready' if has_hourly else 'Missing'}</div><div class='small'>Hourly ML model</div></div>", unsafe_allow_html=True)
with col3:
    has_daily = os.path.exists(MODEL_DAILY)
    st.markdown(f"<div class='card'><div class='kv'>{'Ready' if has_daily else 'Missing'}</div><div class='small'>Daily ML model</div></div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='card'><div class='kv'>{'ML' if blend_api>=0.5 else 'API'}</div><div class='small'>Primary source</div></div>", unsafe_allow_html=True)

st.markdown("---")

# ----------------- HISTORICAL CHART -----------------
if not hist_hour.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_hour["time"], y=hist_hour["temp"], name="Archive", mode="lines", line=dict(width=2)))
    if not api_hour.empty:
        fig.add_trace(go.Scatter(x=api_hour["time"], y=api_hour["temp"], name="API Forecast", mode="lines", line=dict(dash="dash")))
    fig.update_layout(margin=dict(l=0,r=0,t=30,b=0), height=320, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# ----------------- ML HOURLY (24h) -----------------
st.markdown("### Next 24 Hours — Hourly Forecast")
hour_cols = st.container()
if os.path.exists(MODEL_HOURLY) and not hist_hour.empty:
    try:
        preds_ml = load_hourly_model_and_predict(MODEL_HOURLY, hist_hour, input_lags=48, horizon=24)
        times = [hist_hour["time"].iloc[-1] + timedelta(hours=i+1) for i in range(24)]
        df_ml = pd.DataFrame({"time": times, "temp_ml": preds_ml})
        # blend with API where available
        if not api_hour.empty:
            api_sub = api_hour[api_hour["time"].isin(df_ml["time"])].set_index("time")
            blended = []
            for t,row in df_ml.iterrows():
                api_t = api_sub.loc[row["time"]]["temp"] if row["time"] in api_sub.index else np.nan
                if np.isfinite(api_t):
                    val = blend_api * row["temp_ml"] + (1-blend_api) * api_t
                else:
                    val = row["temp_ml"]
                blended.append(val)
            df_ml["temp"] = blended
        else:
            df_ml["temp"] = df_ml["temp_ml"]

        # hourly strip
        html = "<div class='hour-strip card'>"
        for i,row in df_ml.head(24).iterrows():
            hr = row["time"].hour
            t = row["temp"]
            html += f"<div class='hour-item'><div style='font-weight:700'>{hr:02d}:00</div><div style='margin-top:6px'>{t:.1f}°C</div></div>"
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

        # chart small
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=df_ml["time"], y=df_ml["temp"], name="ML blended", mode="lines+markers"))
        fig_h.update_layout(height=260, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig_h, use_container_width=True)

    except Exception as e:
        st.error(f"Hourly ML prediction error: {e}")
else:
    st.info("Hourly ML model not available — train models using the sidebar.")

st.markdown("---")

# ----------------- DAILY 30-DAY (MAX) -----------------
st.markdown("### 30-Day Forecast — Max Temperature (Trend + Variation)")

if os.path.exists(MODEL_DAILY):
    try:
        start_long, end_long = get_past_range_hours(24 * 365)
        hist_long = fetch_hourly_history_archive(lat, lon, start_long, end_long)

        if hist_long.empty:
            st.warning("Not enough history for daily ML model.")
        else:
            df30 = load_daily_models_and_predict(
                MODEL_DAILY, hist_long,
                input_lags_days=60,
                horizon_days=30
            )

            df30["smooth"] = df30["pred_max"].ewm(span=4).mean()

            st.dataframe(df30.set_index("date").round(1), height=380)

            fig_d = go.Figure()

            fig_d.add_trace(go.Scatter(
                x=df30["date"], y=df30["pred_max"],
                name="Daily ML Prediction",
                mode="lines+markers",
                line=dict(width=2)
            ))

            fig_d.add_trace(go.Scatter(
                x=df30["date"], y=df30["smooth"],
                name="Smoothed Trend (EMA)",
                mode="lines",
                line=dict(width=4)
            ))

            fig_d.update_layout(
                title="30-Day ML Forecast — Trend & Smoothing",
                xaxis_title="Date",
                yaxis_title="Max Temperature °C",
                height=380,
                template="plotly_white",
                margin=dict(l=0, r=0, t=50, b=0)
            )

            st.plotly_chart(fig_d, use_container_width=True)

    except Exception as e:
        st.error(f"Daily ML prediction error: {e}")
else:
    st.info("Daily ML model not available — train using sidebar.")

st.markdown("---")
st.markdown("<div class='footer'>Built with Machine Learning + Open-Meteo • Blend ML + API for robustness</div>", unsafe_allow_html=True)






