# pollution_layers.py
import osmnx as ox
import geopandas as gpd
import requests
import pandas as pd
from shapely.geometry import Point

# --- 1. ROAD DENSITY (VEHICLE PROXY) ---
print("⛓️ Fetching road network for Delhi...")
city = "Delhi, India"
G = ox.graph_from_place(city, network_type='drive')
roads = ox.graph_to_gdfs(G, nodes=False)
roads.to_file("delhi_roads.geojson", driver='GeoJSON')
print("✅ Roads saved as delhi_roads.geojson")

# --- 2. INDUSTRIAL ZONES ---
print("🏭 Fetching industrial zones...")
tags = {'landuse': 'industrial'}
industries = ox.geometries_from_place(city, tags)
industries = industries[['geometry']]
industries.to_file("delhi_industrial.geojson", driver='GeoJSON')
print("✅ Industrial zones saved as delhi_industrial.geojson")

# --- 3. FIRE POINTS FROM NASA FIRMS ---
print("🔥 Fetching crop burning fire data from NASA FIRMS...")

API_KEY = "d8805f3d7247755cd578e3456bd85753"  # Replace with your API key
satellite = "VIIRS_SNPP_NRT"
country_code = "IND"
days = 7
end_date = "2025-07-04"

url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{API_KEY}/{satellite}/{country_code}/{days}/{end_date}"

response = requests.get(url)
if response.status_code != 200:
    raise Exception(f"Failed to fetch FIRMS data: {response.status_code}")

with open("fires.csv", "w", encoding="utf-8") as f:
    f.write(response.text)

df = pd.read_csv("fires.csv")
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
gdf = gdf[['geometry', 'acq_date', 'brightness', 'confidence', 'frp', 'daynight']]
gdf.to_file("fires.geojson", driver="GeoJSON")
print("✅ Fire data saved as fires.geojson")
