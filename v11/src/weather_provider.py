"""
V11 KG-CTCN Weather Data Provider
--------------------------------------------------
Fetches real historical weather data from NASA POWER API.
Enforces strict causality and daily grid normalization.
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Region to Lat/Lon mapping for production defaults
REGION_COORDS = {
    "Sangli": {"lat": 16.54, "lon": 69.78},
    "Kolhapur": {"lat": 16.7050, "lon": 74.2433},
    "Pune": {"lat": 18.5204, "lon": 73.8567},
    "DEFAULT": {"lat": 16.54, "lon": 69.78}
}

class WeatherDataProvider:
    def __init__(self):
        self.base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        self.parameters = "WS10M,T2M,RH2M,T2M_MIN,T2M_MAX,PRECTOTCORR"
        
    def get_weather_history(self, location, target_date, window_days=400):
        """
        Fetches historical weather data for a location.
        The inference engine needs ~400 days to compute 365-day rolling Z-scores.
        location: region name (str) or dict {"lat": float, "lon": float}
        """
        if isinstance(location, str):
            coords = REGION_COORDS.get(location, REGION_COORDS["DEFAULT"])
        else:
            coords = location
            
        start_date = target_date - timedelta(days=window_days)
        end_date = target_date
        
        # NASA POWER dates are YYYYMMDD
        start_str = start_date.strftime("%Y%m%dd")[:-1] # Remove 'd' if any? No, format is YYYYMMDD
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        
        params = {
            "parameters": self.parameters,
            "community": "AG",
            "longitude": coords["lon"],
            "latitude": coords["lat"],
            "start": start_str,
            "end": end_str,
            "format": "JSON"
        }
        
        print(f"Fetching real weather from NASA POWER: {coords} | {start_str} to {end_str}")
        
        try:
            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            # Extract features
            features = data['properties']['parameter']
            df = pd.DataFrame(features)
            
            # Convert index (YYYYMMDD string) to datetime
            df.index = pd.to_datetime(df.index, format="%Y%m%d")
            df = df.sort_index()
            
            # Reset index to have 'date' column
            df = df.reset_index().rename(columns={'index': 'date'})
            
            # Fill missing values safely (forward fill)
            df = df.ffill().bfill()
            
            # Strict causality check: ensure no future data exists in the returned frame
            df = df[df['date'] <= target_date]
            
            return df
            
        except Exception as e:
            print(f"NASA POWER API Error: {e}")
            raise ConnectionError(f"Failed to fetch real-time weather data: {e}")

if __name__ == "__main__":
    # Test fetch
    provider = WeatherDataProvider()
    test_date = datetime(2023, 8, 1)
    df = provider.get_weather_history("Sangli", test_date)
    print(df.head())
    print(f"Rows fetched: {len(df)}")
