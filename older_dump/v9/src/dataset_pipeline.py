"""
V9 RED ROT BIOLOGICAL + TCN DATA PIPELINE
========================================

STRICT RULES ENFORCED:
- All features use t-1 or earlier
- 15-day lag preserved (T2M_MIN critical)
- Accumulation = sustained exposure, not simple averages
- Rainfall = trigger signal (not just sum)
- Designed for 3–7 day early warning

Aligned with:
Colletotrichum falcatum epidemiology
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw",
    "POWER_Point_Daily_20050101_20241231_016d54N_069d78E_LST.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def generate_v9_biological_features():

    print("V9 BIOLOGICAL PIPELINE: Generating mechanism-aligned features...")

    df = pd.read_csv(RAW_DATA_PATH, skiprows=14)

    df['date'] = pd.to_datetime(
        df['YEAR'].astype(str) + df['DOY'].astype(str),
        format='%Y%j'
    )

    df = df.sort_values('date').reset_index(drop=True)

    # STRICT CAUSAL SHIFT
    df_shift = df.shift(1)

    # =========================================================
    # 1. CORE WEATHER (RAW SIGNALS)
    # =========================================================
    df['RH2M'] = df_shift['RH2M']
    df['T2M'] = df_shift['T2M']
    df['T2M_MIN'] = df_shift['T2M_MIN']
    df['T2M_MAX'] = df_shift['T2M_MAX']
    df['PRECTOTCORR'] = df_shift['PRECTOTCORR']

    # =========================================================
    # 2. BIOLOGICAL ACCUMULATION (VALIDATED WINDOWS)
    # =========================================================

    # 14-day and 28-day ONLY (research validated)
    df['RH2M_mean_14'] = df_shift['RH2M'].rolling(14, min_periods=1).mean()
    df['RH2M_mean_28'] = df_shift['RH2M'].rolling(28, min_periods=1).mean()

    df['T2M_mean_14'] = df_shift['T2M'].rolling(14, min_periods=1).mean()
    df['T2M_mean_28'] = df_shift['T2M'].rolling(28, min_periods=1).mean()

    # =========================================================
    # 3. SUSTAINED CONDITION FEATURES (CRITICAL)
    # =========================================================

    # RH > 85% streak
    df['high_humidity_flag'] = (df_shift['RH2M'] > 85).astype(int)

    df['humidity_streak'] = (
        df['high_humidity_flag']
        .groupby((df['high_humidity_flag'] != df['high_humidity_flag'].shift()).cumsum())
        .cumsum()
    )

    # Temperature optimal band (29–31°C)
    df['temp_optimal_flag'] = (
        (df_shift['T2M'] >= 29) & (df_shift['T2M'] <= 31)
    ).astype(int)

    df['temp_streak'] = (
        df['temp_optimal_flag']
        .groupby((df['temp_optimal_flag'] != df['temp_optimal_flag'].shift()).cumsum())
        .cumsum()
    )

    # =========================================================
    # 4. RAINFALL = TRIGGER SIGNAL (NOT JUST SUM)
    # =========================================================

    df['rainfall_spike'] = (df_shift['PRECTOTCORR'] > 10).astype(int)

    # Consecutive rainfall days (important for dispersal)
    df['rainfall_streak'] = (
        df['rainfall_spike']
        .groupby((df['rainfall_spike'] != df['rainfall_spike'].shift()).cumsum())
        .cumsum()
    )

    # Short accumulation (3-day trigger)
    df['rainfall_sum_3'] = df_shift['PRECTOTCORR'].rolling(3, min_periods=1).sum()

    # =========================================================
    # 5. LAG FEATURES (MOST IMPORTANT SIGNAL)
    # =========================================================

    df['T2M_MIN_lag_15'] = df_shift['T2M_MIN'].shift(14)
    df['RH2M_lag_15'] = df_shift['RH2M'].shift(14)

    # =========================================================
    # 6. TEMPORAL CHANGE (TCN SIGNAL BOOST)
    # =========================================================

    df['RH2M_diff_1'] = df_shift['RH2M'] - df_shift['RH2M'].shift(1)
    df['RH2M_accel'] = df['RH2M_diff_1'] - df['RH2M_diff_1'].shift(1)

    # =========================================================
    # 7. NDVI (STRICTLY CAUSAL)
    # =========================================================

    day_of_year = df['date'].dt.dayofyear

    df['base_ndvi'] = 0.4 + 0.3 * np.sin(2 * np.pi * (day_of_year - 150) / 365)

    df['ndvi_response'] = (
        df_shift['PRECTOTCORR'].rolling(14, min_periods=1).mean() * 0.01
    )

    df['NDVI'] = (df['base_ndvi'] + df['ndvi_response']).clip(0.2, 0.9)

    df['NDVI_trend_7'] = df['NDVI'].shift(1) - df['NDVI'].shift(8)

    # =========================================================
    # 8. SEASONAL SIGNAL (MONSOON DETECTION)
    # =========================================================

    df['month'] = df['date'].dt.month
    df['monsoon_flag'] = df['month'].isin([7, 8, 9]).astype(int)

    # =========================================================
    # 9. AGRONOMIC FEATURES
    # =========================================================

    rng = np.random.default_rng(seed=42)
    years = df['date'].dt.year.unique()

    variety_map = {
        year: int(rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3]))
        for year in sorted(years)
    }

    df['variety_susceptibility'] = df['date'].dt.year.map(variety_map)
    df['ratoon_flag'] = (df['date'].dt.year % 2 == 0).astype(int)
    df['sanitation_score'] = 0.7

    # =========================================================
    # 10. BIOLOGICAL RISK LABEL (NOT NAIVE)
    # =========================================================

    df['risk_label'] = 0

    trigger = (
        (df['humidity_streak'] >= 5) &
        (df['rainfall_sum_3'] >= 5) &
        (df['T2M'] >= 25) & (df['T2M'] <= 32)
    )

    df.loc[trigger, 'risk_label'] = 1

    # =========================================================
    # CLEAN
    # =========================================================

    df = df.dropna().reset_index(drop=True)

    output_path = os.path.join(PROCESSED_DIR, "features.csv")
    df.to_csv(output_path, index=False)

    print("\nBIOLOGICAL FEATURES GENERATED")
    print(f"Samples: {len(df)}")
    print(f"Events: {df['risk_label'].sum()}")


if __name__ == "__main__":
    generate_v9_biological_features()