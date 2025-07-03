import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
from datetime import datetime
from AQIPredict import load_city_data, load_weather_data, create_features, forecast_next_days

st.set_page_config(page_title="AQI Forecaster", layout="centered")

st.title("🌫️ Air Quality Index (AQI) Forecaster")
st.markdown("Select a city to view historical AQI and weather trends, and forecast the upcoming air quality.")

CITIES = {
    "Delhi": (28.61, 77.23), "Mumbai": (19.07, 72.87), "Kolkata": (22.574, 88.363),
    "Chennai": (13.0827, 80.2707), "Hyderabad": (17.385, 78.4867), "Bengaluru": (12.9716, 77.5946)
}

city = st.selectbox("Choose a city", list(CITIES.keys()))
lat, lon = CITIES[city.lower().capitalize()]

with st.spinner("Loading historical data..."):
    df_aqi, df_pollutant = load_city_data(city.lower(), 2021, 2025)
    if df_aqi.empty:
        st.error(f"No AQI data found for {city}.")
        st.stop()

    df_weather = load_weather_data(lat, lon, df_aqi["Date"].min(), df_aqi["Date"].max())
    df_merged = pd.merge(df_aqi, df_weather, on="Date", how="inner")
    df_merged = pd.merge(df_merged, df_pollutant, on="Date", how="left")

    # Plotting historical data with severity bands
    st.subheader("📊 Historical AQI Trends with Weather Data")

    def get_severity_color(aqi):
        if aqi <= 50: return "#00e400"
        elif aqi <= 100: return "#ffff00"
        elif aqi <= 200: return "#ff7e00"
        elif aqi <= 300: return "#ff0000"
        elif aqi <= 400: return "#99004c"
        else: return "#7e0023"

    def get_aqi_category(aqi):
        if aqi <= 50: return "Good ✅"
        elif aqi <= 100: return "Satisfactory 😊"
        elif aqi <= 200: return "Moderate 😐"
        elif aqi <= 300: return "Poor 😷"
        elif aqi <= 400: return "Very Poor 😵"
        else: return "Severe ❌"

    def plot_historical(df, year):
        df_year = df[df["Date"].dt.year == year]
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(df_year["Date"], df_year["AQI"], label="AQI", color="black")
        for _, row in df_year.iterrows():
            ax1.axvspan(row["Date"], row["Date"], color=get_severity_color(row["AQI"]), alpha=0.2)

        ax2 = ax1.twinx()
        ax2.plot(df_year["Date"], df_year["TempAvg"], label="Temp Avg (°C)", color="blue", linestyle="--", alpha=0.6)
        ax2.plot(df_year["Date"], df_year["WindSpeed"], label="Wind Speed (km/h)", color="green", linestyle=":", alpha=0.6)

        ax1.set_ylabel("AQI")
        ax2.set_ylabel("Weather")
        ax1.set_title(f"{city} - {year} AQI with Weather")
        ax1.grid(True, linestyle='--', alpha=0.3)

        aqi_legend = [
            mpatches.Patch(color="#00e400", label="Good (0-50)"),
            mpatches.Patch(color="#ffff00", label="Satisfactory (51-100)"),
            mpatches.Patch(color="#ff7e00", label="Moderate (101-200)"),
            mpatches.Patch(color="#ff0000", label="Poor (201-300)"),
            mpatches.Patch(color="#99004c", label="Very Poor (301-400)"),
            mpatches.Patch(color="#7e0023", label="Severe (401+)")
        ]
        ax1.legend(handles=aqi_legend, loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)

        st.pyplot(fig)
        avg = df_year["AQI"].mean()
        st.info(f"**{year} Summary:** Average AQI was {avg:.2f} → {get_aqi_category(avg)}")

    for yr in sorted(df_merged["Date"].dt.year.unique()):
        with st.expander(f"📅 {yr} - Click to view graph"):
            plot_historical(df_merged, yr)

if st.button("📈 Forecast Next 7 Days"):
    try:
        model = joblib.load(f"aqi_model_{city.lower()}.pkl")
    except:
        st.error("Model not found. Train the model first using AQIPredict.py")
        st.stop()

    with st.spinner("Forecasting AQI for the next 7 days..."):
        df_merged = df_merged.dropna(subset=["AQI", "TempAvg", "WindSpeed"])
        for pol in ["PM10", "PM2.5", "OZONE", "NO2", "CO"]:
            df_merged[pol] = 0
        for i, row in df_merged.iterrows():
            if pd.notna(row["Pollutants"]):
                for pol in row["Pollutants"].split(','):
                    if pol.strip().upper() in df_merged.columns:
                        df_merged.at[i, pol.strip().upper()] = 1

        df_feat = create_features(df_merged)
        forecast_dates, forecast_values = forecast_next_days(model, df_feat)

        st.subheader("🔮 7-Day AQI Forecast")
        df_forecast = pd.DataFrame({"Date": forecast_dates, "Predicted AQI": forecast_values})
        df_forecast["Severity"] = df_forecast["Predicted AQI"].apply(get_aqi_category)
        st.dataframe(df_forecast.style.highlight_max(axis=0, color="lightcoral"))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df_forecast["Date"], df_forecast["Predicted AQI"], marker="o", color="black")
        for _, row in df_forecast.iterrows():
            ax.axvspan(row["Date"], row["Date"], color=get_severity_color(row["Predicted AQI"]), alpha=0.3)
        ax.set_title(f"{city} - 7-Day Forecast")
        ax.set_ylabel("AQI")
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)

        st.info(f"**Next 7 Days Summary:** Peak AQI expected is {max(forecast_values):.2f} → {get_aqi_category(max(forecast_values))}")
