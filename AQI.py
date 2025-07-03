# Updated AQIPredict.py with Prominent Pollutant Features and Improved Parameters

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from meteostat import Daily, Stations
import joblib

# ------------------------- STEP 1: Load AQI Data ------------------------- #
def load_city_data(city, start_year, end_year, data_dir="AQIData"):
    df_all = []
    df_pollutants = []
    for year in range(start_year, end_year + 1):
        file_path = os.path.join(data_dir, f"AQI_daily_city_level_{city}_{year}_{city}_{year}.xlsx")
        if not os.path.exists(file_path):
            print(f"\u274c File not found: {file_path}")
            continue

        try:
            xls = pd.ExcelFile(file_path)
            aqi_sheet = next((s for s in xls.sheet_names if "prominent" not in s.lower()), xls.sheet_names[0])
            prom_sheet = next((s for s in xls.sheet_names if "prominent" in s.lower()), None)

            # Load AQI
            df = pd.read_excel(xls, sheet_name=aqi_sheet)
            if df.columns[0].lower() == "date":
                df.rename(columns={df.columns[0]: "Day"}, inplace=True)

            months = [col for col in df.columns if col in list(pd.date_range('2000-01-01', '2000-12-01', freq='MS').strftime('%B'))]
            df_long = df.melt(id_vars="Day", value_vars=months, var_name="Month", value_name="AQI")
            df_long["Month"] = pd.to_datetime(df_long["Month"], format='%B').dt.month
            df_long["Year"] = year
            df_long["Date"] = pd.to_datetime(df_long[["Year", "Month", "Day"]], errors='coerce')
            df_aqi = df_long[["Date", "AQI"]].dropna()
            df_all.append(df_aqi)

            # Load Prominent Pollutants
            if prom_sheet:
                dfp = pd.read_excel(xls, sheet_name=prom_sheet)
                dfp_long = dfp.melt(id_vars="Day", value_vars=months, var_name="Month", value_name="Pollutants")
                dfp_long["Month"] = pd.to_datetime(dfp_long["Month"], format='%B').dt.month
                dfp_long["Year"] = year
                dfp_long["Date"] = pd.to_datetime(dfp_long[["Year", "Month", "Day"]], errors='coerce')
                df_pollutants.append(dfp_long[["Date", "Pollutants"]].dropna())

        except Exception as e:
            print(f"\u26a0\ufe0f Error reading {file_path}: {e}")
            continue

    df_all = pd.concat(df_all).drop_duplicates().sort_values("Date").reset_index(drop=True)
    df_pollutants = pd.concat(df_pollutants).drop_duplicates().sort_values("Date").reset_index(drop=True)
    
    if df_all.empty:
        print("\U0001F6AB No valid data loaded for city:", city)
    return df_all, df_pollutants


# ------------------------- STEP 2: Load Weather Data ------------------------- #
def load_weather_data(lat, lon, start_date, end_date):
    station = Stations().nearby(lat, lon).fetch(1)
    data = Daily(station, start_date, end_date).fetch()
    data = data.reset_index().rename(columns={
        "time": "Date", "tavg": "TempAvg", "tmin": "TempMin",
        "tmax": "TempMax", "wspd": "WindSpeed"
    })
    return data


# ------------------------- STEP 3: Feature Engineering ------------------------- #
def create_features(df, lags=[1, 2, 3, 7], rolling=[3, 7, 14]):
    df_feat = df.copy()
    for lag in lags:
        df_feat[f"AQI_lag_{lag}"] = df_feat["AQI"].shift(lag)
    for r in rolling:
        df_feat[f"AQI_rollmean_{r}"] = df_feat["AQI"].shift(1).rolling(window=r).mean()
    for col in ["TempAvg", "WindSpeed"]:
        df_feat[f"{col}_lag1"] = df_feat[col].shift(1)

    df_feat["DayOfWeek"] = df_feat["Date"].dt.dayofweek
    df_feat["Month"] = df_feat["Date"].dt.month
    df_feat["DayOfYear"] = df_feat["Date"].dt.dayofyear
    df_feat["WeekOfYear"] = df_feat["Date"].dt.isocalendar().week.astype(int)
    df_feat["IsWeekend"] = df_feat["DayOfWeek"] >= 5

    print("Before dropna:", df_feat.shape)
    df_feat.dropna(inplace=True)
    print("After dropna:", df_feat.shape)

    return df_feat


# ------------------------- STEP 4: Split and Train Model ------------------------- #
def train_test_split(df_feat, test_days=90):
    split_date = df_feat["Date"].max() - timedelta(days=test_days)
    train = df_feat[df_feat["Date"] <= split_date]
    test = df_feat[df_feat["Date"] > split_date]

    X_train = train.drop(columns=["Date", "AQI"])
    y_train = train["AQI"]
    X_test = test.drop(columns=["Date", "AQI"])
    y_test = test["AQI"]
    test_dates = test["Date"]

    model = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, y_pred, X_test, y_test, test_dates


# ------------------------- STEP 5: Plot & Forecast ------------------------- #
def plot_predictions(dates, y_true, y_pred, city="City"):
    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_true, label="Actual AQI", marker="o")
    plt.plot(dates, y_pred, label="Predicted AQI", linestyle="--")
    plt.title(f"{city.capitalize()} AQI Forecast")
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def forecast_next_days(model, df_feat, future_days=3):
    last_row = df_feat.iloc[-1:].copy()
    known_lags = [col for col in df_feat.columns if col.startswith("AQI_lag_")]
    known_rolls = [col for col in df_feat.columns if col.startswith("AQI_rollmean_")]
    lag_days = sorted([int(col.split("_")[-1]) for col in known_lags])
    roll_windows = sorted([int(col.split("_")[-1]) for col in known_rolls])

    future_preds = []
    for i in range(future_days):
        new_features = {}

        # Reconstruct AQI lags
        for lag in lag_days:
            if i < lag:
                col_name = f"AQI_lag_{lag - i}"
                new_features[f"AQI_lag_{lag}"] = last_row[col_name].values[0] if col_name in last_row else future_preds[-1]
            else:
                new_features[f"AQI_lag_{lag}"] = future_preds[-lag] if len(future_preds) >= lag else future_preds[-1]

        # Reconstruct rolling means
        for r in roll_windows:
            recent_vals = [last_row["AQI"].values[0]] + future_preds[-(r - 1):]
            new_features[f"AQI_rollmean_{r}"] = np.mean(recent_vals[-r:])

        # Lagged weather features
        for col in ["TempAvg", "WindSpeed"]:
            new_features[f"{col}_lag1"] = last_row[f"{col}_lag1"].values[0]

        # Static weather values (carry forward)
        for col in ["TempAvg", "TempMin", "TempMax", "WindSpeed"]:
            new_features[col] = last_row[col].values[0]

        # Pollutant binary flags (assume same as last known)
        for pol in ["PM10", "PM2.5", "OZONE", "NO2", "CO"]:
            new_features[pol] = last_row[pol].values[0]

        # Date features
        future_date = last_row["Date"].values[0] + np.timedelta64(i + 1, 'D')
        future_datetime = pd.to_datetime(str(future_date))
        new_features["DayOfWeek"] = future_datetime.dayofweek
        new_features["Month"] = future_datetime.month
        new_features["DayOfYear"] = future_datetime.dayofyear
        new_features["WeekOfYear"] = future_datetime.isocalendar().week
        new_features["IsWeekend"] = int(future_datetime.dayofweek >= 5)

        X_future = pd.DataFrame([new_features])
        X_future = X_future[model.get_booster().feature_names]  # Keep only model-required columns

        y_future = model.predict(X_future)[0]
        future_preds.append(y_future)

    start_date = df_feat["Date"].max() + timedelta(days=1)
    forecast_dates = [start_date + timedelta(days=i) for i in range(future_days)]

    print("\n📈 AQI Forecast for next {} days:".format(future_days))
    for date, val in zip(forecast_dates, future_preds):
        print(f"{date.strftime('%Y-%m-%d')}: {round(val, 2)}")

    return forecast_dates, future_preds


# ------------------------- MAIN PIPELINE FOR MULTIPLE CITIES ------------------------- #
CITIES = {
    "delhi": (28.61, 77.23),
    "mumbai": (19.07, 72.87),
    "kolkata": (22.574, 88.363),
    "chennai": (13.0827, 80.2707),
    "hyderabad": (17.385, 78.4867),
    "bengaluru": (12.9716, 77.5946)
}
START_YEAR, END_YEAR = 2021, 2025

for CITY, (lat, lon) in CITIES.items():
    print(f"\n\n🚀 Processing City: {CITY.upper()} ----------------------------")

    # Load data
    df_aqi, df_pollutant = load_city_data(CITY, START_YEAR, END_YEAR)
    if df_aqi.empty:
        print(f"⚠️ No AQI data for {CITY}, skipping.")
        continue

    df_weather = load_weather_data(lat, lon, df_aqi["Date"].min(), df_aqi["Date"].max())
    df_merged = pd.merge(df_aqi, df_weather, on="Date", how="inner")
    df_merged = pd.merge(df_merged, df_pollutant, on="Date", how="left")

    # One-hot encode pollutants
    for pol in ["PM10", "PM2.5", "OZONE", "NO2", "CO"]:
        df_merged[pol] = 0
    for i, row in df_merged.iterrows():
        if pd.notna(row["Pollutants"]):
            for pol in row["Pollutants"].split(','):
                pol = pol.strip().upper()
                if pol in df_merged.columns:
                    df_merged.at[i, pol] = 1

    # Final cleaning
    keep_cols = ["Date", "AQI", "TempAvg", "TempMin", "TempMax", "WindSpeed", "PM10", "PM2.5", "OZONE", "NO2", "CO"]
    df_merged = df_merged[keep_cols].dropna()

    if df_merged.empty:
        print(f"⚠️ No merged data for {CITY}, skipping.")
        continue

    # Feature engineering
    df_feat = create_features(df_merged)

    if df_feat.empty:
        print(f"⚠️ Feature data empty for {CITY}, skipping.")
        continue

    # Train, evaluate, forecast
    model, y_pred, X_test, y_test, test_dates = train_test_split(df_feat, test_days=90)
    plot_predictions(test_dates, y_test, y_pred, city=CITY)

    # Save model
    joblib.dump(model, f"aqi_model_{CITY}.pkl")

    # Print test performance
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"\n📊 {CITY.upper()} - Model Performance:")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    # Forecast next 7 days
    forecast_dates, forecast_values = forecast_next_days(model, df_feat, future_days=3)
    print(f"\n🔮 {CITY.upper()} - 7-Day Forecast:")
    for date, val in zip(forecast_dates, forecast_values):
        print(f"{date.strftime('%Y-%m-%d')}: {round(float(val), 2)}")
