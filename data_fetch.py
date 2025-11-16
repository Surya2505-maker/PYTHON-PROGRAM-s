# data_fetch.py
from utils import fetch_hourly_history, get_past_range_hours
import pandas as pd

def fetch_for_training(lat, lon, hours_back=720):
    """
    Fetch `hours_back` hours of history (default 720 ~ 30 days hourly).
    Returns DataFrame.
    """
    start_iso, end_iso = get_past_range_hours(hours_back)
    df = fetch_hourly_history(lat, lon, start_iso, end_iso, timezone="UTC")
    return df

if __name__ == "__main__":
    # quick test for local run
    lat = 13.0827
    lon = 80.2707
    df = fetch_for_training(lat, lon, hours_back=168)
    print(df.tail())
