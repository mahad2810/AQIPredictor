import os
import re
import shutil
import fitz  # PyMuPDF
import pandas as pd
from openpyxl import load_workbook

# --- CONFIG ---
pdf_folder = r"C:\Users\ASUS\AQIPredictor\AQI"
base_excel_path = r"C:\Users\ASUS\AQIPredictor\AQI\AQI_daily_city_level_{}_2025_{}_2025.xlsx"
cities = ["Bengaluru", "Delhi", "Kolkata", "Mumbai", "Chennai", "Hyderabad"]

# --- UTILITIES ---
def get_date_from_filename(filename):
    months = {"May": 5, "June": 6}
    parts = filename.replace(".pdf", "").split()
    day = int(re.sub(r'\D', '', parts[0]))
    month = months[parts[1]]
    return pd.Timestamp(year=2025, month=month, day=day)

def extract_city_data_from_pdf(pdf_path, city_name):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()

    pattern = re.compile(fr'{city_name}\s+(\w+)\s+(\d+)\s+(.*?)\s+\d+/\d+', re.IGNORECASE)
    match = pattern.search(text)
    if match:
        aqi_value = int(match.group(2))
        pollutant = match.group(3).replace('\n', ' ').strip()
        return aqi_value, pollutant
    return None, None

def process_all_pdfs(folder, city_name):
    data = []
    for file in sorted(os.listdir(folder)):
        if file.lower().endswith(".pdf"):
            try:
                date = get_date_from_filename(file)
                aqi, pollutant = extract_city_data_from_pdf(os.path.join(folder, file), city_name)
                if aqi is not None:
                    print(f"✅ {city_name}: {file} → AQI: {aqi}, Pollutant: {pollutant}")
                    data.append({"day": date.day, "month": date.month, "aqi": aqi, "pollutant": pollutant})
            except Exception as e:
                print(f"❌ {city_name}: Failed to process {file}: {e}")
    return pd.DataFrame(data)

def estimate_april_data(df_may_june):
    april_days = list(range(1, 31))
    avg_aqi = int(df_may_june['aqi'].mean())
    mode_pollutant = df_may_june['pollutant'].mode()[0]
    return pd.DataFrame({
        "day": april_days,
        "month": [4]*30,
        "aqi": [avg_aqi]*30,
        "pollutant": [mode_pollutant]*30
    })

def update_excel_calendar_style(excel_path, df_all, city_name):
    df_aqi = pd.read_excel(excel_path, sheet_name="AQI")
    df_pollutant = pd.read_excel(excel_path, sheet_name="Prominent Parameters")
    month_map = {4: "April", 5: "May", 6: "June"}

    for _, row in df_all.iterrows():
        day = row['day']
        month_name = month_map[row['month']]
        df_aqi.loc[df_aqi['Day'] == day, month_name] = row['aqi']
        df_pollutant.loc[df_pollutant['Day'] == day, month_name] = row['pollutant']

    backup_path = excel_path.replace(".xlsx", "_backup.xlsx")
    if not os.path.exists(backup_path):
        shutil.copy(excel_path, backup_path)
        print(f"📦 Backup created for {city_name} at: {backup_path}")

    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
        df_aqi.to_excel(writer, sheet_name="AQI", index=False)
        df_pollutant.to_excel(writer, sheet_name="Prominent Parameters", index=False)
    print(f"✅ Excel updated for {city_name}: {excel_path}")

# --- MAIN LOOP ---
for city in cities:
    city_file = base_excel_path.format(city.lower(), city.lower())
    if not os.path.exists(city_file):
        print(f"⚠️ Excel file not found for {city}: {city_file}")
        continue

    print(f"\n🔄 Processing: {city}")
    df_may_june = process_all_pdfs(pdf_folder, city)
    if df_may_june.empty:
        print(f"⚠️ No data extracted for {city}")
        continue

    df_april = estimate_april_data(df_may_june)
    df_all = pd.concat([df_may_june, df_april], ignore_index=True)
    update_excel_calendar_style(city_file, df_all, city)
