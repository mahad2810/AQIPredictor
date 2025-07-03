import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
import numpy as np
import os
import matplotlib.pyplot as plt

# Load all required model dependencies
from AQIPredict import create_features, forecast_next_days  # <-- Replace with actual module/file name

# ------------------------- Streamlit App ------------------------- #

st.set_page_config(page_title="AQI Forecast", layout="centered")

st.title("🌫️ 3-Day AQI Forecast App")
st.markdown("Select a city to get the predicted Air Quality Index (AQI) for the next 3 days.")

CITIES = {
    "delhi": (28.61, 77.23), "mumbai": (19.07, 72.87), "kolkata": (22.574, 88.363),
    "chennai": (13.0827, 80.2707), "hyderabad": (17.385, 78.4867), "bengaluru": (12.9716, 77.5946)
}

city = st.selectbox("Choose a City", options=list(CITIES.keys()), format_func=lambda x: x.title())

model_path = f"aqi_model_{city}.pkl"
data_path = f"data_feat_{city}.csv"  # Assuming you save preprocessed feature data per city

if st.button("Forecast AQI for Next 3 Days"):
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        st.error(f"Model or data not found for {city.title()}!")
    else:
        model = joblib.load(model_path)
        df_feat = pd.read_csv(data_path, parse_dates=["Date"])

        # Recompute required columns if necessary
        if "AQI_lag_1" not in df_feat.columns:
            df_feat = create_features(df_feat)

        forecast_dates, forecast_values = forecast_next_days(model, df_feat, future_days=3)

        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Predicted AQI": [round(val, 2) for val in forecast_values]
        })

        st.subheader(f"📈 AQI Forecast for {city.title()}")
        st.dataframe(forecast_df)

        # Plot
        st.line_chart(forecast_df.set_index("Date"))

        # Text Summary
        aqi_mean = np.mean(forecast_values)
        if aqi_mean <= 50:
            st.success("Air quality is Good ✅")
        elif aqi_mean <= 100:
            st.info("Air quality is Moderate ☁️")
        elif aqi_mean <= 200:
            st.warning("Air quality is Poor ⚠️")
        else:
            st.error("Air quality is Very Poor or Hazardous ❗")

