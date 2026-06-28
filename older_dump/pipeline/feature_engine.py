import pandas as pd
import numpy as np

def compute_bio_features(df):
    """
    Computes bio-informed intelligence layer features.
    These features represent agronomic and pathological knowledge.
    """
    d = df.copy()
    
    # 🌾 Moisture Stress Index
    # High temp + Low RH + No rain
    d['moisture_stress'] = (
        (d['T2M'] > 30).astype(int) * 0.3 + 
        (d['RH2M'] < 50).astype(int) * 0.4 + 
        (d['PRECTOTCORR'] < 0.1).astype(int) * 0.3
    )
    
    # 🌦️ Dry-to-Wet Trigger
    # Detect sudden rainfall spike after 7 dry days
    d['is_dry'] = (d['PRECTOTCORR'] < 0.5).astype(int)
    d['dry_spell_7d'] = d['is_dry'].rolling(window=7).min()
    d['dry_to_wet_trigger'] = ((d['dry_spell_7d'].shift(1) == 1) & (d['PRECTOTCORR'] > 5.0)).astype(int)
    
    # 🍄 Fungal Risk Index
    # Moderate temp (20-30) + High Humidity (>80) + Rain
    d['fungal_risk'] = (
        ((d['T2M'] >= 20) & (d['T2M'] <= 30)).astype(int) * 
        (d['RH2M'] > 80).astype(int) * 
        (d['PRECTOTCORR'] > 0.5).astype(int).rolling(window=3, min_periods=1).max()
    )
    
    # 🌾 Red Rot Composite Risk
    # Fungal risk + Dry-to-Wet trigger + High Temp spikes
    d['red_rot_risk_composite'] = (
        d['fungal_risk'] * 0.5 + 
        d['dry_to_wet_trigger'] * 0.3 + 
        (d['T2M_MAX'] > 35).astype(int) * 0.2
    ).rolling(window=5, min_periods=1).mean()
    
    # Heat and Cold Stress
    d['heat_stress'] = (d['T2M_MAX'] > 38).astype(int)
    d['cold_stress'] = (d['T2M_MIN'] < 10).astype(int)
    
    return d.drop(columns=['is_dry', 'dry_spell_7d'])

def add_temporal_features(df):
    """
    Adds lags, rolling means, and seasonality.
    """
    d = df.copy()
    
    # Lags
    for col in ['T2M', 'RH2M', 'PRECTOTCORR']:
        for lag in [1, 2, 3, 7]:
            d[f'{col.lower()}_lag_{lag}'] = d[col].shift(lag)
            
    # Rolling Means
    for col in ['T2M', 'RH2M', 'PRECTOTCORR']:
        for w in [3, 7, 14]:
            d[f'{col.lower()}_{w}d_mean'] = d[col].rolling(window=w).mean()
            
    # Seasonality
    d['date'] = pd.to_datetime(d['YEAR'].astype(str) + d['DOY'].astype(str).str.zfill(3), format='%Y%j')
    d['dayofyear'] = d['date'].dt.dayofyear
    d['sin_day'] = np.sin(2 * np.pi * d['dayofyear'] / 365.25)
    d['cos_day'] = np.cos(2 * np.pi * d['dayofyear'] / 365.25)
    
    return d.dropna()

def build_unified_feature_space(raw_df):
    """
    Complete feature pipeline.
    """
    # 1. Bio Features
    df_bio = compute_bio_features(raw_df)
    
    # 2. Temporal Features
    df_full = add_temporal_features(df_bio)
    
    return df_full
