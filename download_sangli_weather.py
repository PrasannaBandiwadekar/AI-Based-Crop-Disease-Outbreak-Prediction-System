import requests
import os
 
# Correct coordinates for Sangli city, Maharashtra, India
LAT = 16.8524
LON = 74.5816
 
# Output filename encodes the coordinates so it is easy to identify
OUTPUT_FILENAME = "POWER_Point_Daily_20050101_20241231_016d85N_074d58E_LST.csv"
OUTPUT_PATH = os.path.join("raw_data", OUTPUT_FILENAME)
 
# NASA POWER API endpoint
URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
 
params = {
    "parameters": "WS10M,T2M,RH2M,T2M_MIN,T2M_MAX,PRECTOTCORR",
    "community": "AG",
    "longitude": LON,
    "latitude": LAT,
    "start": "20050101",
    "end": "20241231",
    "format": "CSV"
}
 
print("Downloading NASA POWER data for Sangli (16.8524 N, 74.5816 E)...")
print("This may take 30-60 seconds depending on your internet speed.")
 
response = requests.get(URL, params=params, timeout=120)
 
if response.status_code == 200:
    os.makedirs("raw_data", exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        f.write(response.content)
    print(f"SUCCESS — file saved to: {OUTPUT_PATH}")
    print(f"File size: {os.path.getsize(OUTPUT_PATH):,} bytes")
else:
    print(f"ERROR — API returned status code: {response.status_code}")
    print(f"Response: {response.text[:500]}")
