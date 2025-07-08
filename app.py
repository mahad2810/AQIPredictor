import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime
from AQIPredict import load_city_data, load_weather_data, create_features, forecast_next_days

# ----------------------------- CONFIG ----------------------------- #
st.set_page_config(page_title="AQI Forecaster App", layout="wide")

# ----------------------------- CUSTOM STYLING ----------------------------- #
st.markdown("""
<style>
html, body, [class*="css"] {
    background: linear-gradient(135deg, #1a1a1a 0%, #000000 50%, #1a1a1a 100%) !important;
    color: #d1d5db !important;
    font-family: 'Segoe UI', sans-serif;
}

/* Navbar */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    padding: 1rem 2rem;
    border-radius: 0 0 12px 12px;
    margin-bottom: 20px;
}
.navbar img {
    height: 40px;
    margin-right: 1rem;
}
.navbar .brand {
    font-size: 1.6rem;
    color: #00ffff;
    font-weight: bold;
    text-shadow: 0 0 10px #06b6d4;
    display: flex;
    align-items: center;
}

h1, h2, h3 {
    color: #ffffff !important;
    text-shadow: 0 0 10px #06b6d4;
}

section[data-testid="stSidebar"] {
    background-color: #111827 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}

.stButton>button {
    background-color: #06b6d4;
    color: #000000;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: bold;
    box-shadow: 0 0 10px #06b6d4;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #00ffff;
    box-shadow: 0 0 15px #00ffff;
}

[data-testid="stMetricValue"] {
    font-size: 1.8rem;
    color: #00ff88 !important;
    text-shadow: 0 0 5px #00ff88;
}

thead tr th {
    background-color: #374151 !important;
    color: #ffffff !important;
}
tbody tr td {
    background-color: #1a1a1a !important;
    color: #d1d5db !important;
}

.css-1lcbmhc, .css-1y4p8pa, .stDataFrame {
    background: rgba(0, 0, 0, 0.2) !important;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(20px);
    padding: 1rem;
    margin-top: 10px;
    margin-bottom: 20px;
}

.glow-cyan {
    text-shadow: 0 0 5px #06b6d4, 0 0 10px #06b6d4;
}

/* Weather animation (sun icon example) */
.weather-icon {
    font-size: 2rem;
    color: #fbbf24;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { text-shadow: 0 0 5px #fbbf24; }
    50% { text-shadow: 0 0 20px #fbbf24; }
    100% { text-shadow: 0 0 5px #fbbf24; }
}
</style>
""", unsafe_allow_html=True)

# ----------------------------- NAVBAR ----------------------------- #
# ----------------------------- NAVBAR ----------------------------- #
st.markdown("""
<div class="navbar">
  <div class="brand">
    <img src="https://raw.githubusercontent.com/mahad2810/AQIPredictor/main/MaveriqAir-Logo.png" alt="logo" style="height: 40px; margin-right: 10px;">
    <span>MaveriqAir</span>
  </div>
  <div class="weather-icon">☀️</div>
</div>
""", unsafe_allow_html=True)

st.title("🌍 Air Quality Forecaster")

# ----------------------------- CITIES ----------------------------- #
CITIES = {
    "Delhi": (28.61, 77.23),
    "Mumbai": (19.07, 72.87),
    "Kolkata": (22.574, 88.363),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.385, 78.4867),
    "Bengaluru": (12.9716, 77.5946)
}

START_YEAR, END_YEAR = 2021, 2025

# ----------------------------- SIDEBAR ----------------------------- #
st.sidebar.header("🔍 Choose City")
city_name = st.sidebar.selectbox("City", list(CITIES.keys()))
lat, lon = CITIES[city_name]

# ----------------------------- LOAD DATA ----------------------------- #
st.info(f"Loading historical AQI & weather data for {city_name}...")
df_aqi, df_pollutant = load_city_data(city_name.lower(), START_YEAR, END_YEAR)

if df_aqi.empty:
    st.error("❌ No AQI data available for this city.")
    st.stop()

weather_start = df_aqi["Date"].min()
weather_end = df_aqi["Date"].max()
df_weather = load_weather_data(lat, lon, weather_start, weather_end)

# Merge and preprocess
df_merged = pd.merge(df_aqi, df_weather, on="Date", how="inner")
df_merged = pd.merge(df_merged, df_pollutant, on="Date", how="left")

for pol in ["PM10", "PM2.5", "OZONE", "NO2", "CO"]:
    df_merged[pol] = 0
for i, row in df_merged.iterrows():
    if pd.notna(row["Pollutants"]):
        for pol in row["Pollutants"].split(','):
            pol = pol.strip().upper()
            if pol in df_merged.columns:
                df_merged.at[i, pol] = 1

keep_cols = ["Date", "AQI", "TempAvg", "TempMin", "TempMax", "WindSpeed", "PM10", "PM2.5", "OZONE", "NO2", "CO"]
df_merged = df_merged[keep_cols].dropna()

# ----------------------------- SHARED YEAR SLIDER ----------------------------- #
df_merged["Year"] = df_merged["Date"].dt.year
available_years = sorted(df_merged["Year"].unique())

selected_year = st.select_slider("📅 Select Year", options=available_years, value=available_years[-1])

# ----------------------------- YEARLY SUMMARY SECTION ----------------------------- #
st.markdown(f'<h2 class="glow-cyan">📌 Summary for {city_name} - {selected_year}</h2>', unsafe_allow_html=True)
df_year_summary = df_merged[df_merged["Year"] == selected_year]

summary_container = st.container()
with summary_container:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌫️ AQI Stats")
        aqi_stats = df_year_summary["AQI"].describe()[["mean", "min", "max", "std"]].round(2)
        st.metric("Average AQI", f"{aqi_stats['mean']}")
        st.metric("Max AQI", f"{aqi_stats['max']}")
        st.metric("Min AQI", f"{aqi_stats['min']}")
        st.metric("Std Deviation", f"{aqi_stats['std']}")

    with col2:
        st.markdown("### 🌤️ Weather Averages")
        st.metric("Avg Temperature (°C)", f"{df_year_summary['TempAvg'].mean():.2f}")
        st.metric("Min Temperature (°C)", f"{df_year_summary['TempMin'].mean():.2f}")
        st.metric("Max Temperature (°C)", f"{df_year_summary['TempMax'].mean():.2f}")
        st.metric("Avg Wind Speed (km/h)", f"{df_year_summary['WindSpeed'].mean():.2f}")

# Pollutant frequency
st.markdown("### 🧪 Pollutant Detection Frequency")
pollutant_cols = ["PM10", "PM2.5", "OZONE", "NO2", "CO"]
pollutant_counts = df_year_summary[pollutant_cols].sum().astype(int)

pollutant_df = pd.DataFrame({
    "Pollutant": pollutant_counts.index,
    "Detected Days": pollutant_counts.values
})
st.dataframe(pollutant_df.style.background_gradient(cmap="YlGnBu"))

# ----------------------------- HISTORICAL DATA CHARTS ----------------------------- #
st.markdown(f'<h2 class="glow-cyan">📊 Historical AQI + Weather Trends for {city_name} (2021–2025)</h2>', unsafe_allow_html=True)

df_merged["Year"] = df_merged["Date"].dt.year
year_groups = dict(tuple(df_merged.groupby("Year")))
available_years = sorted(year_groups.keys())

df_year = df_merged[df_merged["Year"] == selected_year]

with st.container():
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df_year["Date"], df_year["AQI"], color="crimson", label="AQI", linewidth=2)
    ax1.set_ylabel("AQI", color="crimson")
    ax1.tick_params(axis='y', labelcolor='crimson')
    ax1.set_title(f"{city_name} - Year {selected_year}", fontsize=14)

    ax2 = ax1.twinx()
    ax2.plot(df_year["Date"], df_year["TempAvg"], color="darkgreen", label="TempAvg (°C)", linestyle="--")
    ax2.plot(df_year["Date"], df_year["WindSpeed"], color="blue", label="Wind Speed (km/h)", linestyle=":")
    ax2.set_ylabel("Weather", color="gray")
    ax2.tick_params(axis='y', labelcolor='gray')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)

# ----------------------------- FORECAST ----------------------------- #
st.markdown("""<h2 class='glow-cyan'>📈 7-Day Forecast</h2>""", unsafe_allow_html=True)
st.markdown("Click the button below to run the forecast model for the selected city.")

if st.button("🔮 Run Forecast"):
    with st.spinner("Generating forecast..."):
        df_feat = create_features(df_merged)
        model_path = f"aqi_model_{city_name.lower()}.pkl"
        if not os.path.exists(model_path):
            st.error("❌ Forecast model not found for this city.")
            st.stop()

        model = joblib.load(model_path)
        forecast_dates, forecast_values = forecast_next_days(model, df_feat, future_days=7)

        df_forecast = pd.DataFrame({"Date": forecast_dates, "Predicted AQI": forecast_values})
        df_forecast["Date"] = pd.to_datetime(df_forecast["Date"])

        st.line_chart(df_forecast.set_index("Date"))
        st.dataframe(df_forecast.style.background_gradient(cmap="OrRd"))
