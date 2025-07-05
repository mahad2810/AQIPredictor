import osmnx as ox
import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point
from concurrent.futures import ThreadPoolExecutor, as_completed

# --------------------------
# ✅ Configuration
# --------------------------
cities = [
    # Existing 20
    "Delhi, India", "Mumbai, India", "Bengaluru, India", "Kolkata, India",
    "Chennai, India", "Hyderabad, India", "Ahmedabad, India", "Pune, India",
    "Jaipur, India", "Lucknow, India", "Surat, India", "Kanpur, India",
    "Nagpur, India", "Indore, India", "Bhopal, India", "Patna, India",
    "Ludhiana, India", "Agra, India", "Nashik, India", "Faridabad, India",

    # +100 more
    "Varanasi, India", "Rajkot, India", "Meerut, India", "Jabalpur, India", "Jamshedpur, India",
    "Asansol, India", "Vijayawada, India", "Dhanbad, India", "Amritsar, India", "Allahabad, India",
    "Gwalior, India", "Coimbatore, India", "Hubli-Dharwad, India", "Aurangabad, India", "Jodhpur, India",
    "Madurai, India", "Raipur, India", "Kota, India", "Guwahati, India", "Chandigarh, India",
    "Thiruvananthapuram, India", "Solapur, India", "Ranchi, India", "Bareilly, India", "Moradabad, India",
    "Mysuru, India", "Tiruchirappalli, India", "Jalandhar, India", "Bhubaneswar, India", "Salem, India",
    "Warangal, India", "Guntur, India", "Bhiwandi, India", "Saharanpur, India", "Gorakhpur, India",
    "Bikaner, India", "Amravati, India", "Noida, India", "Firozabad, India", "Muzaffarnagar, India",
    "Udaipur, India", "Aligarh, India", "Bilaspur, India", "Jhansi, India",
    "Siliguri, India", "Nanded, India", "Belgaum, India", "Cuttack, India", "Akola, India",
    "Bhavnagar, India", "Kollam, India", "Kolhapur, India", "Gaya, India", "Rewa, India",
    "Ujjain, India", "Davangere, India", "Tirunelveli, India", "Erode, India", "Nellore, India",
    "Rourkela, India", "Ajmer, India", "Bellary, India", "Durgapur, India", "Tuticorin, India",
    "Ratlam, India", "Kakinada, India", "Panipat, India", "Anantapur, India", "Karnal, India",
    "Hisar, India", "Rohtak, India", "Sonipat, India", "Alwar, India", "Sambalpur, India",
    "Shimoga, India", "Yamunanagar, India", "Pondicherry, India", "Nizamabad, India", "Bardhaman, India",
    "Karimnagar, India", "Muzaffarpur, India", "Bokaro Steel City, India", "Rampur, India",
    "Haridwar, India", "Barasat, India", "Katihar, India", "Darbhanga, India", "Gopalganj, India",
    "Bidar, India", "Satna, India", "Dewas, India", "Palakkad, India", "Kottayam, India",
    "Bhagalpur, India", "Haldwani, India", "Bhatinda, India", "Ambala, India", "Imphal, India",
    "Agartala, India", "Dimapur, India", "Itanagar, India", "Aizawl, India", "Shillong, India",
    "Kohima, India", "Gangtok, India", "Port Blair, India", "Pali, India", "Gandhinagar, India",
    "Bharuch, India", "Navsari, India", "Valsad, India", "Vellore, India",
    "Kanchipuram, India", "Tirupati, India", "Hosur, India", "Tumakuru, India"
]

road_tags = {
    'highway': [
        'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
        'motorway_link', 'trunk_link', 'primary_link', 'secondary_link'
    ]
}
industry_tags = {'landuse': 'industrial'}

# --------------------------
# 🚀 Functions
# --------------------------
def fetch_city_data(city):
    city_short = city.split(",")[0].lower().replace(" ", "_")
    result = {}

    try:
        # Roads
        roads = ox.features_from_place(city, tags=road_tags)
        roads = roads[roads.geom_type.isin(['LineString', 'MultiLineString'])]
        roads['geometry'] = roads['geometry'].simplify(tolerance=0.0004, preserve_topology=True)
        roads = roads[['geometry', 'name', 'highway']].dropna(subset=['geometry'])
        roads.to_file(f"{city_short}_roads.geojson", driver="GeoJSON")
        result['roads'] = True
    except Exception as e:
        print(f"❌ Roads for {city}: {e}")
        result['roads'] = False

    try:
        # Industries
        industries = ox.features_from_place(city, tags=industry_tags)
        industries = industries[industries.geom_type.isin(['Polygon', 'MultiPolygon'])]
        industries = industries[['geometry']].dropna(subset=['geometry'])
        industries.to_file(f"{city_short}_industries.geojson", driver="GeoJSON")
        result['industries'] = True
    except Exception as e:
        print(f"❌ Industries for {city}: {e}")
        result['industries'] = False

    print(f"✅ Completed: {city} | Roads: {result['roads']} | Industries: {result['industries']}")
    return result

# --------------------------
# 🔥 NASA FIRMS GLOBAL FIRE DATA
# --------------------------
def fetch_global_fire_data():
    print("🔥 Fetching global fire data from NASA FIRMS...")
    API_KEY = "d8805f3d7247755cd578e3456bd85753"
    satellite = "VIIRS_SNPP_NRT"
    region = "world"
    days = 1
    end_date = "2025-07-04"

    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{API_KEY}/{satellite}/{region}/{days}/{end_date}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch FIRMS data: {response.status_code}")
    with open("fires_world.csv", "w", encoding="utf-8") as f:
        f.write(response.text)

    df = pd.read_csv("fires_world.csv")
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        raise ValueError("Missing lat/lon columns.")
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    columns_to_keep = ['geometry', 'acq_date', 'bright_ti4', 'confidence', 'frp', 'daynight']
    available = [col for col in columns_to_keep if col in gdf.columns]
    gdf = gdf[available]
    gdf.to_file("fires.geojson", driver="GeoJSON")
    print("✅ Global fire data saved as fires.geojson")

# --------------------------
# 🚀 Main: Parallel Fetching
# --------------------------
if __name__ == "__main__":
    print(f"🚀 Starting data fetch for {len(cities)} cities across India...")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_city_data, city): city for city in cities}
        for future in as_completed(futures):
            pass  # already prints in function

    fetch_global_fire_data()
    print("🎉 All done! You now have ~250 GeoJSON files (roads + industries + fires) covering all India.")
