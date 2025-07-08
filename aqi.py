# Updated AQI Forecasting with Forecast Integration

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from meteostat import Daily, Stations
import joblib

# ------------------------- Load AQI Data ------------------------- #
def load_city_data(city, start_year, end_year, data_dir="AQIData"):
    df_all, df_pollutants = [], []
    for year in range(start_year, end_year + 1):
        file_path = os.path.join(data_dir, f"AQI_daily_city_level_{city}_{year}_{city}_{year}.xlsx")
        if not os.path.exists(file_path): continue

        try:
            xls = pd.ExcelFile(file_path)
            aqi_sheet = next((s for s in xls.sheet_names if "prominent" not in s.lower()), xls.sheet_names[0])
            prom_sheet = next((s for s in xls.sheet_names if "prominent" in s.lower()), None)

            df = pd.read_excel(xls, sheet_name=aqi_sheet)
            if df.columns[0].lower() == "date": df.rename(columns={df.columns[0]: "Day"}, inplace=True)
            months = [col for col in df.columns if col in list(pd.date_range('2000-01-01', '2000-12-01', freq='MS').strftime('%B'))]
            df_long = df.melt(id_vars="Day", value_vars=months, var_name="Month", value_name="AQI")
            df_long["Month"] = pd.to_datetime(df_long["Month"], format='%B').dt.month
            df_long["Year"] = year
            df_long["Date"] = pd.to_datetime(df_long[["Year", "Month", "Day"]], errors='coerce')
            df_all.append(df_long[["Date", "AQI"]].dropna())

            if prom_sheet:
                dfp = pd.read_excel(xls, sheet_name=prom_sheet)
                dfp_long = dfp.melt(id_vars="Day", value_vars=months, var_name="Month", value_name="Pollutants")
                dfp_long["Month"] = pd.to_datetime(dfp_long["Month"], format='%B').dt.month
                dfp_long["Year"] = year
                dfp_long["Date"] = pd.to_datetime(dfp_long[["Year", "Month", "Day"]], errors='coerce')
                df_pollutants.append(dfp_long[["Date", "Pollutants"]].dropna())

        except Exception: continue

    return pd.concat(df_all).drop_duplicates().sort_values("Date"), pd.concat(df_pollutants).drop_duplicates().sort_values("Date")


# ------------------------- Weather Data ------------------------- #
def load_weather_data(lat, lon, start_date, end_date):
    station = Stations().nearby(lat, lon).fetch(1)
    data = Daily(station, start_date, end_date).fetch().reset_index()
    return data.rename(columns={"time": "Date", "tavg": "TempAvg", "tmin": "TempMin", "tmax": "TempMax", "wspd": "WindSpeed"})


# ------------------------- Feature Engineering ------------------------- #
def create_features(df, lags=[1, 2, 3, 7], rolling=[3, 7, 14]):
    df_feat = df.copy()
    for lag in lags:
        df_feat[f"AQI_lag_{lag}"] = df_feat["AQI"].shift(lag)
    for r in rolling:
        df_feat[f"AQI_rollmean_{r}"] = df_feat["AQI"].shift(1).rolling(window=r).mean()

    df_feat["AQI_diff_1day"] = df_feat["AQI"] - df_feat["AQI_lag_1"]
    df_feat["DayOfWeek"] = df_feat["Date"].dt.dayofweek
    df_feat["IsWeekend"] = df_feat["DayOfWeek"] >= 5
    df_feat["Month"] = df_feat["Date"].dt.month
    df_feat["DayOfYear"] = df_feat["Date"].dt.dayofyear
    df_feat["WeekOfYear"] = df_feat["Date"].dt.isocalendar().week.astype(int)
    for col in ["TempAvg", "WindSpeed"]:
        df_feat[f"{col}_lag1"] = df_feat[col].shift(1)
    df_feat.dropna(inplace=True)
    return df_feat


# ------------------------- Forecasting ------------------------- #
from datetime import datetime

def forecast_next_days(model, df_feat, future_days=7):
    today = pd.to_datetime(datetime.now().date())
    if today <= df_feat["Date"].max():
        # If today is covered, shift to the latest available date in the data
        last_row = df_feat[df_feat["Date"] == df_feat["Date"].max()].copy()
    else:
        # Otherwise, simulate forecasting from most recent known data
        last_row = df_feat.iloc[-1:].copy()

    known_lags = [col for col in df_feat.columns if col.startswith("AQI_lag_")]
    known_rolls = [col for col in df_feat.columns if col.startswith("AQI_rollmean_")]
    lag_days = sorted([int(col.split("_")[-1]) for col in known_lags])
    roll_windows = sorted([int(col.split("_")[-1]) for col in known_rolls])
    feature_names = model.get_booster().feature_names

    future_preds = []
    for i in range(future_days):
        new_features = {}
        for lag in lag_days:
            if i < lag:
                col_name = f"AQI_lag_{lag - i}"
                new_features[f"AQI_lag_{lag}"] = last_row[col_name].values[0] if col_name in last_row else future_preds[-1]
            else:
                new_features[f"AQI_lag_{lag}"] = future_preds[-lag] if len(future_preds) >= lag else future_preds[-1]

        for r in roll_windows:
            recent_vals = [last_row["AQI"].values[0]] + future_preds[-(r - 1):]
            new_features[f"AQI_rollmean_{r}"] = np.mean(recent_vals[-r:])

        for col in ["TempAvg", "TempMin", "TempMax", "WindSpeed"]:
            new_features[col] = last_row[col].values[0]
            if col in ["TempAvg", "WindSpeed"]:
                new_features[f"{col}_lag1"] = last_row[f"{col}_lag1"].values[0]

        for pol in ["PM10", "PM2.5", "OZONE", "NO2", "CO"]:
            new_features[pol] = last_row[pol].values[0]

        future_date = today + timedelta(days=i)
        new_features["DayOfWeek"] = future_date.dayofweek
        new_features["IsWeekend"] = int(future_date.dayofweek >= 5)
        new_features["Month"] = future_date.month
        new_features["DayOfYear"] = future_date.dayofyear
        new_features["WeekOfYear"] = future_date.isocalendar()[1]

        X_future = pd.DataFrame([new_features])
        for col in feature_names:
            if col not in X_future.columns:
                X_future[col] = 0
        X_future = X_future[feature_names]

        y_future = model.predict(X_future)[0]
        future_preds.append(y_future)

    forecast_dates = [today + timedelta(days=i) for i in range(future_days)]
    print(f"\n🔮 Forecast for next {future_days} days from today ({today.date()}):")
    for d, v in zip(forecast_dates, future_preds):
        print(f"{d.strftime('%Y-%m-%d')}: {round(v, 2)}")
    return forecast_dates, future_preds

# ------------------------- Model Training + Tuning ------------------------- #
def train_with_cv_and_tuning(X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    model = XGBRegressor(random_state=42)
    param_dist = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    search = RandomizedSearchCV(model, param_distributions=param_dist, cv=tscv, n_iter=10, n_jobs=-1, scoring='neg_mean_absolute_error', verbose=1)
    search.fit(X, y)
    return search.best_estimator_


# ------------------------- Plot ------------------------- #
def plot_predictions(dates, y_true, y_pred, city="City"):
    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_true, label="Actual AQI", marker="o")
    plt.plot(dates, y_pred, label="Predicted AQI", linestyle="--")
    plt.title(f"{city.upper()} AQI Forecast")
    plt.xlabel("Date"); plt.ylabel("AQI")
    plt.legend(); plt.grid(); plt.tight_layout(); plt.show()


# ------------------------- Main Loop ------------------------- #
CITIES = {
    "delhi": (28.61, 77.23), "mumbai": (19.07, 72.87), "kolkata": (22.574, 88.363),
    "chennai": (13.0827, 80.2707), "hyderabad": (17.385, 78.4867), "bengaluru": (12.9716, 77.5946)
}
START_YEAR, END_YEAR = 2021, 2025

city_metrics = []  # <- move this above your for loop

for CITY, (lat, lon) in CITIES.items():
    print(f"\n\n🚀 Processing City: {CITY.upper()} ----------------------------")
    df_aqi, df_pollutant = load_city_data(CITY, START_YEAR, END_YEAR)
    if df_aqi.empty: continue

    df_weather = load_weather_data(lat, lon, df_aqi["Date"].min(), df_aqi["Date"].max())
    df_merged = pd.merge(df_aqi, df_weather, on="Date", how="inner")
    df_merged = pd.merge(df_merged, df_pollutant, on="Date", how="left")

    for pol in ["PM10", "PM2.5", "OZONE", "NO2", "CO"]: df_merged[pol] = 0
    for i, row in df_merged.iterrows():
        if pd.notna(row["Pollutants"]):
            for pol in row["Pollutants"].split(','):
                if pol.strip().upper() in df_merged.columns:
                    df_merged.at[i, pol.strip().upper()] = 1

    keep_cols = ["Date", "AQI", "TempAvg", "TempMin", "TempMax", "WindSpeed", "PM10", "PM2.5", "OZONE", "NO2", "CO"]
    df_merged = df_merged[keep_cols].dropna()
    df_feat = create_features(df_merged)
    X = df_feat.drop(columns=["Date", "AQI"])
    y = df_feat["AQI"]

    model = train_with_cv_and_tuning(X, y)
    y_pred = model.predict(X)
    joblib.dump(model, f"aqi_model_{CITY}.pkl")
    plot_predictions(df_feat["Date"], y, y_pred, city=CITY)
    
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    print("\n📊 Model Evaluation:")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    city_metrics.append({
        "City": CITY.capitalize(),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    forecast_next_days(model, df_feat, future_days=7)
    
# Create DataFrame from collected metrics
df_metrics = pd.DataFrame(city_metrics)

# Plotting the metrics
plt.figure(figsize=(10, 6))
plt.title("Air Quality Model Metrics Across Cities")

# Plot MAE, RMSE, and R2 side by side
bar_width = 0.25
r1 = range(len(df_metrics))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

plt.bar(r1, df_metrics["MAE"], color='salmon', width=bar_width, label='MAE')
plt.bar(r2, df_metrics["RMSE"], color='skyblue', width=bar_width, label='RMSE')
plt.bar(r3, df_metrics["R2"], color='lightgreen', width=bar_width, label='R2')

# Add labels
plt.xticks([r + bar_width for r in range(len(df_metrics))], df_metrics["City"])
plt.ylabel("Metric Value")
plt.legend()
plt.tight_layout()

# Save to file
plt.savefig("all_city_metrics.png", dpi=300)
plt.close()
