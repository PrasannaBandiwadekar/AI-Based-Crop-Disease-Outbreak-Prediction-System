import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ---------------------------
# 1. DISTRICT MAPPING (Maharashtra Sugarcane Belt)
# ---------------------------
DISTRICTS = {
    "Pune": {"lat": 18.5204, "lon": 73.8567},
    "Satara": {"lat": 17.6805, "lon": 73.9914},
    "Kolhapur": {"lat": 16.7050, "lon": 74.2433},
    "Sangli": {"lat": 16.8524, "lon": 74.5815},
    "Solapur": {"lat": 17.6599, "lon": 75.9064},
    "Ahmednagar": {"lat": 19.0948, "lon": 74.7480}
}

# ---------------------------
# 2. API INTEGRATION
# ---------------------------
def fetch_live_weather(district_name):
    if district_name not in DISTRICTS:
        raise ValueError(f"District {district_name} not found.")
        
    coords = DISTRICTS[district_name]
    
    # Fetch 14 days history + 7 days forecast
    # Open-Meteo parameters
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum,wind_speed_10m_max",
        "timezone": "auto",
        "past_days": 21 # We fetch extra to ensure enough rolling window history
    }
    
    url = "https://api.open-meteo.com/v1/forecast"
    response = requests.get(url, params=params)
    data = response.json()
    
    daily = data['daily']
    df = pd.DataFrame({
        "date": pd.to_datetime(daily['time']),
        "T2M_MAX": daily['temperature_2m_max'],
        "T2M_MIN": daily['temperature_2m_min'],
        "T2M": daily['temperature_2m_mean'],
        "RH2M": daily['relative_humidity_2m_mean'],
        "PRECTOTCORR": daily['precipitation_sum'],
        "WS10M": daily['wind_speed_10m_max']
    })
    
    return process_live_features(df)

# ---------------------------
# 3. LIVE FEATURE ENGINEERING (Biological Port)
# ---------------------------
def process_live_features(df):
    # Standard Rolling Features
    df['temp_range'] = df['T2M_MAX'] - df['T2M_MIN']
    df['temp_mean_3d'] = df['T2M'].rolling(3).mean()
    df['temp_mean_7d'] = df['T2M'].rolling(7).mean()
    df['temp_mean_14d'] = df['T2M'].rolling(14).mean()
    
    df['rain_3d'] = df['PRECTOTCORR'].rolling(3).sum()
    df['rain_7d'] = df['PRECTOTCORR'].rolling(7).sum()
    df['rain_14d'] = df['PRECTOTCORR'].rolling(14).sum()
    
    # Missing Standard Flags
    df['is_rainy'] = (df['PRECTOTCORR'] > 1).astype(int)
    df['dry_spell_7d'] = (df['rain_7d'] < 5).astype(int)
    
    df['rh_3d'] = df['RH2M'].rolling(3).mean()
    df['rh_7d'] = df['RH2M'].rolling(7).mean()
    df['high_humidity'] = (df['RH2M'] > 80).astype(int)
    
    df['wind_3d'] = df['WS10M'].rolling(3).mean()
    
    # 🦠 Biological Logic (Red Rot Specific)
    # 1. Wet Streak: Continuous moisture (Red Rot loves standing water)
    df['wet_streak'] = ((df['PRECTOTCORR'] > 5).rolling(7).sum() >= 4).astype(int)
    
    # 2. Humid Streak: High humidity persistence
    df['humid_streak'] = ((df['RH2M'] > 85).rolling(5).sum() >= 4).astype(int)
    
    # 3. Optimal Temperature Window for Red Rot (25-30°C)
    df['temp_optimal_red_rot'] = df['T2M'].between(25, 30).astype(int)
    
    # 4. Dry-to-Wet Trigger
    df['dry_to_wet_trigger'] = ((df['rain_7d'].shift(1) < 5) & (df['PRECTOTCORR'] > 15)).astype(int)
    
    # 5. Composite Risk
    df['red_rot_risk_composite'] = (
        (df['RH2M'] > 85) & 
        (df['T2M'].between(25, 30)) & 
        (df['rain_7d'] > 20)
    ).astype(int)
    
    # Moisture Stress Gating
    def get_moisture_stress(row):
        r3 = row['rain_3d']
        h3 = row['rh_3d']
        if r3 > 25 and h3 > 80: return 2 # HIGH
        elif r3 > 10 and h3 > 75: return 1 # MED
        return 0 # LOW
    
    df['moisture_stress'] = df.apply(get_moisture_stress, axis=1)
    
    # Time Features
    df['month'] = df['date'].dt.month
    df['dayofyear'] = df['date'].dt.dayofyear
    df['sin_day'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['cos_day'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    
    # Regional Crop Age logic
    def get_crop_age_logic(row):
        m = row['month']
        doy = row['dayofyear']
        if m in [7, 8]: start = 182 # Adsali
        elif m in [10, 11]: start = 274 # Pre-seasonal
        elif m in [1, 2]: start = 1 # Suru
        else: return pd.Series([90, 0], index=['crop_age_days', 'suppress_alert'])
        
        age = max(0, doy - start)
        return pd.Series([age, 1 if age < 60 else 0], index=['crop_age_days', 'suppress_alert'])
        
    df[['crop_age_days', 'suppress_alert']] = df.apply(get_crop_age_logic, axis=1)
    
    # Lag Features (1, 2, 3, 7)
    for lag in [1, 2, 3, 7]:
        df[f'temp_lag_{lag}'] = df['T2M'].shift(lag)
        df[f'rain_lag_{lag}'] = df['PRECTOTCORR'].shift(lag)
        df[f'rh_lag_{lag}'] = df['RH2M'].shift(lag)
        
    df['fungal_risk'] = ((df['RH2M'] > 80) & (df['T2M'] > 20) & (df['PRECTOTCORR'] > 1)).astype(int)
    df['heat_stress'] = (df['T2M_MAX'] > 35).astype(int)
    df['cold_stress'] = (df['T2M_MIN'] < 10).astype(int)
    
    return df.dropna().reset_index(drop=True)

if __name__ == "__main__":
    test_pune = fetch_live_weather("Pune")
    print(f"Fetched {len(test_pune)} days of processed live weather for Pune.")
    print(test_pune.tail(1)[['date', 'T2M', 'rain_3d', 'moisture_stress']])
