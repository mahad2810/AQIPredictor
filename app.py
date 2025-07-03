# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from AQIPredict import create_features, load_city_data, load_weather_data, forecast_next_days

st.set_page_config(page_title="AQI Forecaster", layout="centered")

# --------------------- CONFIG ---------------------
CITIES = {
    "Delhi": (28.61, 77.23),
    "Mumbai": (19.07, 72.87),
    "Kolkata": (22.574, 88.363),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.385, 78.4867),
    "Bengaluru": (12.9716, 77.5946)
}
START_YEAR, END_YEAR = 2021, 2025

# --------------------- UI ---------------------
st.title("🔮 AQI Forecasting App")
city = st.selectbox("Select a City", list(CITIES.keys()))
lat, lon = CITIES[city]

st.write("📍 Forecasting AQI for:", city)

model_path = f"aqi_model_{city.lower()}.pkl"
if not os.path.exists(model_path):
    st.warning("❌ Model not trained yet. Please run `AQIPredict.py` to train first.")
    st.stop()

model = joblib.load(model_path)

# Load data
with st.spinner("Loading data..."):
    df_aqi, df_pollutant = load_city_data(city.lower(), START_YEAR, END_YEAR)
    df_weather = load_weather_data(lat, lon, df_aqi["Date"].min(), df_aqi["Date"].max())
    df_merged = pd.merge(df_aqi, df_weather, on="Date", how="inner")
    df_merged = pd.merge(df_merged, df_pollutant, on="Date", how="left")

    for pol in ["PM10", "PM2.5", "OZONE", "NO2", "CO"]:
        df_merged[pol] = 0
    for i, row in df_merged.iterrows():
        if pd.notna(row["Pollutants"]):
            for pol in row["Pollutants"].split(','):
                if pol.strip().upper() in df_merged.columns:
                    df_merged.at[i, pol.strip().upper()] = 1

    df_feat = create_features(df_merged)
    forecast_dates, forecast_values = forecast_next_days(model, df_feat, future_days=7)

# --------------------- OUTPUT ---------------------
st.subheader("📈 7-Day Forecast")
forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecasted AQI": forecast_values})
st.line_chart(forecast_df.set_index("Date"))

st.subheader("📊 Forecast Data Table")
st.dataframe(forecast_df.style.format({"Forecasted AQI": "{:.2f}"}))

st.success("✅ Forecasting Complete!")

