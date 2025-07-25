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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Root Variables */
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --glass-bg: rgba(255, 255, 255, 0.08);
    --glass-border: rgba(255, 255, 255, 0.18);
    --text-primary: #ffffff;
    --text-secondary: #e2e8f0;
    --text-muted: #94a3b8;
    --accent-blue: #3b82f6;
    --accent-purple: #8b5cf6;
    --success-green: #10b981;
    --warning-orange: #f59e0b;
    --danger-red: #ef4444;
}

/* Global Styles */
html, body, [class*="css"] {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #533483 100%) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    font-weight: 400;
    line-height: 1.6;
}

/* Enhanced Glassmorphism Container */
.glass-container {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 16px !important;
    box-shadow: 
        0 8px 32px rgba(0, 0, 0, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    padding: 1.5rem !important;
    margin: 1rem 0 !important;
    position: relative;
    overflow: hidden;
}

.glass-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
}

/* Premium Navbar */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(15, 15, 35, 0.95) !important;
    backdrop-filter: blur(40px) saturate(180%) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 20px !important;
    padding: 1.25rem 2rem !important;
    margin: 1rem 0 2rem 0 !important;
    box-shadow: 
        0 8px 32px rgba(0, 0, 0, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    position: relative;
    overflow: hidden;
}

.navbar::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, rgba(103, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    z-index: -1;
}

.navbar .brand {
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    display: flex !important;
    align-items: center !important;
    letter-spacing: -0.025em !important;
}

.navbar img {
    height: 45px !important;
    margin-right: 12px !important;
    filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.3)) !important;
}

/* Enhanced Weather Icon */
.weather-icon {
    font-size: 2.5rem !important;
    background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    animation: premium-pulse 3s ease-in-out infinite !important;
    filter: drop-shadow(0 0 20px rgba(251, 191, 36, 0.4)) !important;
}

@keyframes premium-pulse {
    0%, 100% { 
        transform: scale(1);
        filter: drop-shadow(0 0 20px rgba(251, 191, 36, 0.4));
    }
    50% { 
        transform: scale(1.05);
        filter: drop-shadow(0 0 30px rgba(251, 191, 36, 0.6));
    }
}

/* Typography Enhancements */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    letter-spacing: -0.025em !important;
    margin-bottom: 1rem !important;
}

h1 {
    font-size: 2.5rem !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}

h2 {
    font-size: 1.875rem !important;
    color: var(--text-primary) !important;
}

.glow-cyan {
    background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    filter: drop-shadow(0 0 10px rgba(6, 182, 212, 0.3)) !important;
}

/* Premium Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15, 15, 35, 0.95) 0%, rgba(26, 26, 46, 0.95) 100%) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
}

section[data-testid="stSidebar"] > div {
    background: transparent !important;
}

/* Enhanced Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.025em !important;
    box-shadow: 
        0 4px 15px rgba(103, 126, 234, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
    overflow: hidden !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 
        0 8px 25px rgba(103, 126, 234, 0.6),
        inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* Premium Metrics */
[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    filter: drop-shadow(0 0 10px rgba(16, 185, 129, 0.3)) !important;
}

[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* Enhanced Tables */
.stDataFrame, [data-testid="stDataFrame"] {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(20px) !important;
    border-radius: 16px !important;
    border: 1px solid var(--glass-border) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    overflow: hidden !important;
}

thead tr th {
    background: linear-gradient(135deg, rgba(103, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    padding: 1rem !important;
    border: none !important;
}

tbody tr td {
    background: rgba(255, 255, 255, 0.02) !important;
    color: var(--text-secondary) !important;
    padding: 0.875rem 1rem !important;
    border: none !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
}

tbody tr:hover td {
    background: rgba(255, 255, 255, 0.05) !important;
}

/* Info/Alert Boxes */
.stAlert, [data-testid="stAlert"] {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
}

/* Select Slider Enhancement */
.stSelectSlider {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(20px) !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
}

/* Chart Containers */
.stPlotlyChart, [data-testid="stPlotlyChart"] {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(20px) !important;
    border-radius: 16px !important;
    border: 1px solid var(--glass-border) !important;
    padding: 1rem !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
}

/* Spinner Enhancement */
.stSpinner {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(20px) !important;
    border-radius: 12px !important;
    border: 1px solid var(--glass-border) !important;
}

/* Container Enhancements */
.main .block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}

/* Custom AQI Status Card */
.aqi-status-card {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    text-align: center !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    box-shadow: 
        0 8px 32px rgba(0, 0, 0, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    position: relative !important;
    overflow: hidden !important;
}

.aqi-status-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #10b981, #059669);
}

/* Scrollbar Styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
}

/* Enhanced Recommendation Section Styles */
.recommendation-section {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    margin: 1.5rem 0 !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
}

.context-selector {
    margin-bottom: 1rem !important;
}

/* Premium AQI Card */
.premium-aqi-card {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 20px !important;
    padding: 2rem !important;
    margin: 2rem 0 !important;
    box-shadow: 
        0 12px 40px rgba(0, 0, 0, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    position: relative !important;
    overflow: hidden !important;
    text-align: center !important;
}

.premium-aqi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #667eea, #764ba2);
    border-radius: 20px 20px 0 0;
}

.aqi-header {
    margin-bottom: 1.5rem !important;
}

.aqi-label {
    font-size: 0.875rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--text-muted) !important;
    margin-bottom: 0.5rem !important;
    font-weight: 500 !important;
}

.aqi-location {
    font-size: 1rem !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
}

.aqi-main {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    gap: 1rem !important;
    margin-bottom: 1rem !important;
}

.aqi-value {
    font-size: 3.5rem !important;
    font-weight: 800 !important;
    line-height: 1 !important;
    text-shadow: 0 0 20px currentColor !important;
}

.aqi-status {
    padding: 0.5rem 1.5rem !important;
    border-radius: 25px !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

.aqi-indicator {
    position: absolute !important;
    bottom: 0 !important;
    left: 0 !important;
    right: 0 !important;
    height: 4px !important;
    border-radius: 0 0 20px 20px !important;
}

/* Enhanced Radio Button Styling */
.stRadio > div {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    margin: 1rem 0 !important;
}

.stRadio > div > label {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    padding: 0.75rem 1.5rem !important;
    margin: 0 0.5rem !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
    font-weight: 500 !important;
}

.stRadio > div > label:hover {
    background: rgba(255, 255, 255, 0.1) !important;
    border-color: rgba(103, 126, 234, 0.5) !important;
}

.stRadio > div > label[data-checked="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border-color: #667eea !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(103, 126, 234, 0.4) !important;
}

/* Enhanced Button Spacing */
.stButton {
    margin: 1.5rem 0 !important;
}

.stButton > button {
    width: 100% !important;
    min-height: 3rem !important;
    font-size: 1rem !important;
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    box-shadow: 
        0 6px 20px rgba(16, 185, 129, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    box-shadow: 
        0 8px 25px rgba(16, 185, 129, 0.6),
        inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
}

/* Advice Container */
.advice-container {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    margin: 1.5rem 0 !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    border-left: 4px solid var(--success-green) !important;
}

.advice-content {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    margin-top: 1rem !important;
    line-height: 1.7 !important;
    color: var(--text-secondary) !important;
}

.advice-content p {
    margin-bottom: 1rem !important;
}

/* Spinner Enhancement */
.stSpinner > div {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    padding: 2rem !important;
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

# ----------------------------- HEALTH RECOMMENDATIONS ----------------------------- #
import google.generativeai as genai
import requests

# Setup Gemini Flash
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# 🌐 Fetch real-time AQI from AirVisual API
def get_current_aqi_cn(lat, lon, api_key):
    url = f"https://api.airvisual.com/v2/nearest_city?lat={lat}&lon={lon}&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data.get("status") == "success":
            aqi_cn = data["data"]["current"]["pollution"]["aqicn"]
            return aqi_cn
        else:
            raise ValueError("API response error:", data.get("status"))
    else:
        raise ConnectionError(f"Failed to fetch data: {response.status_code}")

# 🧠 Gemini Prompt Generator
def generate_health_prompt(city, aqi, context="living"):
    return f"""
You are a certified air quality and health advisor. Provide clear, non-technical health recommendations for a person {context} in {city}, India, where the current AQI is {aqi}.

Guidelines:
- Mention which health groups are at risk
- Suggest outdoor activity limits
- Recommend masks, filters, or air purifiers if needed
- Suggest general wellness tips (hydration, food, exercise)
- Be brief, friendly, and practical
- Use bullet points
"""

# 🔮 Gemini Call
def get_recommendation(city, aqi, context):
    prompt = generate_health_prompt(city, aqi, context)
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

# ----------------------------- UI: Health Recommendation Section ----------------------------- #
st.markdown("""<h2 class='glow-cyan'>🩺 Personalized Health Recommendations</h2>""", unsafe_allow_html=True)

# Enhanced context selection with better spacing
st.markdown("""
<div class="recommendation-section">
    <div class="context-selector">
        <h4 style="color: var(--text-secondary); font-size: 1rem; margin-bottom: 1rem; font-weight: 500;">
            Are you currently living in or planning to travel to this city?
        </h4>
    </div>
</div>
""", unsafe_allow_html=True)

context_choice = st.radio(
    "", 
    ["Living", "Travelling"], 
    horizontal=True,
    key="context_radio"
)

st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

# Fetch live AQI
try:
    live_aqi = get_current_aqi_cn(lat, lon, api_key="e5e635df-37a8-4ac7-b712-2a0207578385")
except Exception as e:
    st.error(f"⚠️ Failed to fetch live AQI: {e}")
    live_aqi = int(df_year_summary["AQI"].mean())  # fallback

# Color logic for AQI severity
aqi_color = "#22c55e" if live_aqi <= 100 else "#facc15" if live_aqi <= 200 else "#f97316" if live_aqi <= 300 else "#ef4444"
aqi_status = "Good" if live_aqi <= 50 else "Moderate" if live_aqi <= 100 else "Unhealthy for Sensitive" if live_aqi <= 150 else "Unhealthy" if live_aqi <= 200 else "Very Unhealthy" if live_aqi <= 300 else "Hazardous"

# Enhanced AQI status card
st.markdown(f"""
<div class="premium-aqi-card">
    <div class="aqi-header">
        <div class="aqi-label">Real-time Air Quality Index</div>
        <div class="aqi-location">{city_name} • {context_choice}</div>
    </div>
    <div class="aqi-main">
        <div class="aqi-value" style="color: {aqi_color};">{live_aqi}</div>
        <div class="aqi-status" style="background: {aqi_color}20; color: {aqi_color}; border: 1px solid {aqi_color}40;">
            {aqi_status}
        </div>
    </div>
    <div class="aqi-indicator" style="background: {aqi_color};"></div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)

# Enhanced Generate button with proper spacing
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_advice = st.button(
        "🧠 Generate Health Advice", 
        key="generate_health_advice",
        use_container_width=True
    )

st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

# Generate Gemini Recommendations
if generate_advice:
    with st.spinner("Analyzing real-time AQI and generating personalized advice..."):
        advice = get_recommendation(city_name, live_aqi, context_choice.lower())
        
        # Enhanced advice display
        st.markdown("""
        <div class="advice-container">
            <div class="advice-header">
                <h3 style="color: var(--success-green); margin-bottom: 0.5rem;">
                    ✅ Health Tips Based on Current Air Quality
                </h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Format advice with better styling
        formatted_advice = advice.replace('\n', '\n\n').replace('•', '🔸')
        st.markdown(f"""
        <div class="advice-content">
            {formatted_advice}
        </div>
        """, unsafe_allow_html=True)
