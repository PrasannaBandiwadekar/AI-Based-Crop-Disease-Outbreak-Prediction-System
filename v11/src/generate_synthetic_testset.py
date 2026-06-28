import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTHETIC_DIR = os.path.join(BASE_DIR, "data", "synthetic")
os.makedirs(SYNTHETIC_DIR, exist_ok=True)

RAW_OUT_PATH = os.path.join(SYNTHETIC_DIR, "v11_synthetic_raw.csv")

def generate_synthetic_raw():
    """
    Generates a synthetic NASA POWER-like CSV with EXTREME biological signals.
    """
    start_date = datetime(2023, 1, 1) # Start earlier for more history
    end_date = datetime(2027, 12, 31)
    dates = pd.date_range(start_date, end_date)
    
    df = pd.DataFrame(index=dates)
    df['YEAR'] = df.index.year
    df['DOY'] = df.index.dayofyear
    
    # 1. BASELINE WEATHER
    n = len(df)
    np.random.seed(42)
    df['T2M'] = 28 + np.random.randn(n) * 1 # Warm baseline
    df['T2M_MIN'] = df['T2M'] - 4
    df['T2M_MAX'] = df['T2M'] + 4
    df['RH2M'] = 65 + np.random.randn(n) * 3 # Dry baseline
    df['WS10M'] = 4 + np.random.randn(n) * 0.5
    df['PRECTOTCORR'] = 0.0
    
    # 2. MONSOON SEASON
    monsoon_mask = (df['DOY'] >= 160) & (df['DOY'] <= 280)
    df.loc[monsoon_mask, 'RH2M'] = 80 + np.random.randn(monsoon_mask.sum()) * 2
    df.loc[monsoon_mask, 'PRECTOTCORR'] = np.random.exponential(10, size=monsoon_mask.sum())
    
    # 3. EXTREME OUTBREAK SCENARIO (2025)
    # Cold dip 15 days before peak
    dip_start = datetime(2025, 7, 10)
    dip_end = datetime(2025, 7, 15)
    df.loc[dip_start:dip_end, 'T2M_MIN'] = 12.0 # Brutal cold dip
    
    # Extreme humidity and rain during peak window
    peak_start = datetime(2025, 7, 25)
    peak_end = datetime(2025, 8, 5)
    df.loc[peak_start:peak_end, 'RH2M'] = 98.0
    df.loc[peak_start:peak_end, 'PRECTOTCORR'] = 40.0
    
    # Clip values
    df['RH2M'] = df['RH2M'].clip(0, 100)
    df['PRECTOTCORR'] = df['PRECTOTCORR'].clip(0, 500)
    
    # Format for NASA POWER
    with open(RAW_OUT_PATH, 'w') as f:
        for i in range(14):
            f.write(f"- Metadata line {i+1} -\n")
        cols = ['YEAR', 'DOY', 'WS10M', 'T2M', 'RH2M', 'T2M_MIN', 'T2M_MAX', 'PRECTOTCORR']
        f.write(",".join(cols) + "\n")
        df[cols].to_csv(f, header=False, index=False)
        
    print(f"Generated extreme synthetic raw data: {RAW_OUT_PATH}")

if __name__ == "__main__":
    generate_synthetic_raw()
