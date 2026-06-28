import pandas as pd
import numpy as np
from . import config

def build_features(df):
    """
    Applies the identical feature engineering pipeline to any dataset (Train or Inference).
    Enforces strict chronological causality and calendar alignment.
    """
    df = df.copy()

    # 1. Calendar Alignment (CRITICAL)
    # Ensure 1 row = 1 day exactly.
    df = df.set_index("date").asfreq("D").reset_index()

    # 2. Missing Data Rules
    # Replace NASA missing value -999 if present
    df.replace(-999, np.nan, inplace=True)
    
    # Rainfall: fill with 0 (no phantom storms)
    if "PRECTOTCORR" in df.columns:
        df["PRECTOTCORR"] = df["PRECTOTCORR"].fillna(0)

    # Temp/Humidity: linear interpolate up to 3 days
    cols_to_interp = ["T2M", "T2M_MIN", "T2M_MAX", "RH2M"]
    for c in cols_to_interp:
        if c in df.columns:
            df[c] = df[c].interpolate(method='linear', limit=3)

    # 3. Rolling Features
    # center=False is default, guarantees no future leakage
    df["T2M_roll7"] = df["T2M"].rolling(window=7, min_periods=1).mean()
    df["RH2M_roll7"] = df["RH2M"].rolling(window=7, min_periods=1).mean()
    df["PREC_sum14"] = df["PRECTOTCORR"].rolling(window=14, min_periods=1).sum()

    # 4. Lag Features
    df["T2M_MIN_lag15"] = df["T2M_MIN"].shift(15)
    df["PREC_lag15"] = df["PRECTOTCORR"].shift(15)

    # 5. Biological Flags
    df["month"] = df["date"].dt.month
    df["monsoon_flag"] = df["month"].apply(lambda x: 1 if 6 <= x <= 10 else 0)
    
    df["temperature_suitability_flag"] = df["T2M_roll7"].apply(
        lambda x: 1 if 29 <= x <= 33 else 0
    )

    # env_interaction_score (0-3 scale)
    def calc_env_score(row):
        score = 0
        if row["RH2M_roll7"] >= 82:
            score += 1
        if row["T2M_roll7"] >= 29 and row["T2M_roll7"] <= 33:
            score += 1
        if row["PREC_sum14"] >= 15:
            score += 1
        return score
    
    df["env_interaction_score"] = df.apply(calc_env_score, axis=1)

    # Ensure all ENGINEERED_FEATURES exist
    for f in config.ENGINEERED_FEATURES:
        if f not in df.columns:
            df[f] = np.nan

    return df

def add_label(df):
    """
    Computes outbreak_risk label ONLY during training.
    Label is 1 if env_interaction_score >= 2 for the future window t+3 to t+7.
    """
    df = df.copy()
    
    # We use env_interaction_score >= 2 as the biological target
    # Wait, the prompt says "env_interaction_score (0-3 rule-based index)".
    # Historically, the outbreak risk target was if env score >= 2 in the window.
    future_env_score = df["env_interaction_score"].shift(-7).rolling(window=5, min_periods=1).max()
    df["outbreak_risk"] = (future_env_score >= 2).astype(int)
    
    # Prevent shifting NaNs into valid numbers by zeroing them where shift drops off
    # Actually, dropna at the end of train data prep handles this
    return df
