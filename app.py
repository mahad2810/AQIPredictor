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

context_choice = st.radio("Are you currently living in or planning to travel to this city?", ["Living", "Travelling"], horizontal=True)

# Fetch live AQI
try:
    live_aqi = get_current_aqi_cn(lat, lon, api_key="e5e635df-37a8-4ac7-b712-2a0207578385")
except Exception as e:
    st.error(f"⚠️ Failed to fetch live AQI: {e}")
    live_aqi = int(df_year_summary["AQI"].mean())  # fallback

# Color logic for AQI severity
aqi_color = "#22c55e" if live_aqi <= 100 else "#facc15" if live_aqi <= 200 else "#f97316" if live_aqi <= 300 else "#ef4444"

# Display live AQI card
st.markdown(f"""
<div style="background-color: {aqi_color}; padding: 1rem; border-radius: 10px; color: black; text-align: center; font-weight: bold; font-size: 1.1rem;">
Real-time AQI (CN) in {city_name}: {live_aqi} ({context_choice.lower()})
</div>
""", unsafe_allow_html=True)

# Generate Gemini Recommendations
if st.button("🧠 Generate Health Advice"):
    with st.spinner("Analyzing real-time AQI and generating personalized advice..."):
        advice = get_recommendation(city_name, live_aqi, context_choice.lower())
        st.success("Health Tips Based on Current Air Quality:")
        st.markdown(advice.replace('\n', '\n\n'))  # markdown friendly format
