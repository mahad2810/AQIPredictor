import requests

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

# Example usage
lat, lon = 22.574, 88.363  # Kolkata
api_key = "e5e635df-37a8-4ac7-b712-2a0207578385"
aqi_cn = get_current_aqi_cn(lat, lon, api_key)
print(f"Current AQI (CN) for coordinates ({lat}, {lon}): {aqi_cn}")
