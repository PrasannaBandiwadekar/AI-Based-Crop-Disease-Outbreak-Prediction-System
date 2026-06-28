import pandas as pd
import numpy as np

# Load NASA POWER CSV
file_path = "POWER_Point_Daily_20050101_20241231_016d54N_069d78E_LST.csv"

# Skip metadata rows (adjust if needed: 14–16 works for most files)
df = pd.read_csv(file_path, skiprows=14)

# Drop empty rows
df = df.dropna(how='all')

# Create datetime column
df['date'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['DOY'].astype(str), format='%Y-%j')

# Drop original date component columns if they exist
for col in ['YEAR', 'DOY', 'MO', 'DY']:
    if col in df.columns:
        df = df.drop(columns=[col])

# -----------------------
# 🌡️ TEMPERATURE FEATURES
# -----------------------
df['temp_range'] = df['T2M_MAX'] - df['T2M_MIN']
df['temp_mean_3d'] = df['T2M'].rolling(3).mean()
df['temp_mean_7d'] = df['T2M'].rolling(7).mean()
df['temp_mean_14d'] = df['T2M'].rolling(14).mean()

# -----------------------
# 🌧️ RAIN FEATURES
# -----------------------
df['rain_3d'] = df['PRECTOTCORR'].rolling(3).sum()
df['rain_7d'] = df['PRECTOTCORR'].rolling(7).sum()
df['rain_14d'] = df['PRECTOTCORR'].rolling(14).sum()

# Dry / wet flags
df['is_rainy'] = (df['PRECTOTCORR'] > 1).astype(int)
df['dry_spell_7d'] = (df['rain_7d'] < 5).astype(int)

# -----------------------
# 💧 HUMIDITY FEATURES
# -----------------------
df['rh_3d'] = df['RH2M'].rolling(3).mean()
df['rh_7d'] = df['RH2M'].rolling(7).mean()
df['high_humidity'] = (df['RH2M'] > 80).astype(int)

# -----------------------
# 💨 WIND FEATURES
# -----------------------
df['wind_3d'] = df['WS10M'].rolling(3).mean()

# -----------------------
# ⏳ LAG FEATURES (VERY IMPORTANT)
# -----------------------
for lag in [1, 2, 3, 7]:
    df[f'temp_lag_{lag}'] = df['T2M'].shift(lag)
    df[f'rain_lag_{lag}'] = df['PRECTOTCORR'].shift(lag)
    df[f'rh_lag_{lag}'] = df['RH2M'].shift(lag)

# -----------------------
# 📅 TIME FEATURES
# -----------------------
df['month'] = df['date'].dt.month
df['dayofyear'] = df['date'].dt.dayofyear
df['week'] = df['date'].dt.isocalendar().week

# Seasonal encoding (important for ML)
df['sin_day'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
df['cos_day'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

# -----------------------
# 🌩️ MOISTURE REALITY GATES (Stability Layer)
# -----------------------

# 1. Moisture Stress: High/Med/Low (Encoded numerically for ML)
def get_moisture_stress(row):
    r3 = row['rain_3d']
    h3 = row['rh_3d']
    if r3 > 25 and h3 > 80:
        return 2  # HIGH
    elif r3 > 10 and h3 > 75:
        return 1  # MEDIUM
    return 0  # LOW

df['moisture_stress'] = df.apply(get_moisture_stress, axis=1)

# 2. Regional Maharashtra Crop Age Mapping
def get_crop_age_and_suppression(row):
    m = row['month']
    doy = row['dayofyear']
    
    # Season start days (approximate)
    adsali_start = 182    # July 1
    pre_seasonal_start = 274 # Oct 1
    suru_start = 1        # Jan 1
    
    # Default to mature (90)
    age = 90
    
    if m in [7, 8]: # Adsali
        age = max(0, doy - adsali_start)
    elif m in [10, 11]: # Pre-seasonal
        age = max(0, doy - pre_seasonal_start)
    elif m in [1, 2]: # Suru
        age = max(0, doy - suru_start)
        
    return pd.Series([age, 1 if age < 60 else 0], index=['crop_age_days', 'suppress_alert'])

df[['crop_age_days', 'suppress_alert']] = df.apply(get_crop_age_and_suppression, axis=1)

# -----------------------
# 🦠 RED ROT SPECIFIC BIOLOGICAL FEATURES (persistence & trigger)
# -----------------------

# 1. Wet Streak: Continuous moisture (Red Rot loves standing water)
# Logic: At least 4 rainy days (>5mm) in the last 7 days
df['wet_streak'] = ((df['PRECTOTCORR'] > 5).rolling(7).sum() >= 4).astype(int)

# 2. Humid Streak: High humidity persistence
# Logic: At least 4 days of >85% humidity in the last 5 days
df['humid_streak'] = ((df['RH2M'] > 85).rolling(5).sum() >= 4).astype(int)

# 3. Optimal Temperature Window for Red Rot (25-30°C)
df['temp_optimal_red_rot'] = df['T2M'].between(25, 30).astype(int)

# 4. Dry → Wet Trigger (Regime Shift)
# Logic: Heavy rain (>15mm) today after a relatively dry week (<5mm total)
df['dry_to_wet_trigger'] = ((df['rain_7d'].shift(1) < 5) & (df['PRECTOTCORR'] > 15)).astype(int)

# 5. Composite Red Rot Trigger (Combined Danger)
df['red_rot_risk_composite'] = (
    (df['RH2M'] > 85) & 
    (df['T2M'].between(25, 30)) & 
    (df['rain_7d'] > 20)
).astype(int)

# 6. Generic fungal risk (already present but kept for comparison)
df['fungal_risk'] = ((df['RH2M'] > 80) & (df['T2M'] > 20) & (df['PRECTOTCORR'] > 1)).astype(int)

df['heat_stress'] = (df['T2M_MAX'] > 35).astype(int)

df['cold_stress'] = (df['T2M_MIN'] < 10).astype(int)

# -----------------------
# CLEAN DATA
# -----------------------
df = df.dropna()

# Save
df.to_csv("engineered_weather_features.csv", index=False)

print("Done! Saved as engineered_weather_features.csv")