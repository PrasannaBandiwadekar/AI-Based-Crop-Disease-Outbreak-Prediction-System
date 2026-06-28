"""
V7 Dataset Pipeline for Red Rot Early Warning System
Implements weather dynamics -> biological stress transitions -> outbreak onset probability
"""
import os
import numpy as np
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "v5", "data", "raw", "POWER_Point_Daily_20050101_20241231_016d54N_069d78E_LST.csv"))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_and_clean_data(path):
    print("STEP 1: LOAD AND CLEAN DATA")
    df = pd.read_csv(path, skiprows=14)
    # Parse date
    df["date"] = pd.to_datetime(df["YEAR"].astype(str) + df["DOY"].astype(str).str.zfill(3), format="%Y%j")
    df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
    
    # Keep columns
    cols_to_keep = ["date", "RH2M", "PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN"]
    df = df[cols_to_keep].copy()
    
    # Handle missing values
    df["PRECTOTCORR"] = df["PRECTOTCORR"].fillna(0)
    for col in ["T2M", "T2M_MAX", "T2M_MIN", "RH2M"]:
        df[col] = df[col].interpolate(method="linear", limit=3)
        
    # Ensure continuous time index
    df = df.set_index("date").asfreq("D").reset_index()
    for col in ["T2M", "T2M_MAX", "T2M_MIN", "RH2M"]:
        df[col] = df[col].interpolate(method="linear")
    df["PRECTOTCORR"] = df["PRECTOTCORR"].fillna(0)
    
    return df

def rolling_slope(series, window=3):
    slopes = [np.nan] * (window - 1)
    arr = series.values
    for i in range(window - 1, len(arr)):
        y = arr[i - window + 1 : i + 1]
        x = np.arange(window)
        # Suppress RankWarning for perfectly flat lines
        with np.errstate(all='ignore'):
            slope = np.polyfit(x, y, 1)[0]
        slopes.append(slope)
    return pd.Series(slopes, index=series.index)

def engineer_features(df):
    print("STEP 2: FEATURE ENGINEERING")
    
    # Base Features are already present.
    # We use .shift(1) for rolling stats to prevent look-ahead bias on day 't'
    # EXCEPT when evaluating state ON day 't' based on day 't' values, which is valid for label generation.
    # The sequences will use X up to day t-1 to predict day t?
    # Sequence: input last 14 days (t-14 to t-1) -> output risk(t)? 
    # Or input last 14 days ending at t -> output risk(t+X)?
    # The prompt says: "input: last 14 days, output: risk (day t)".
    # If the input includes day t, then predicting risk at day t is trivial if risk(t) is deterministically built from day t features!
    # A true forecasting model must predict day t risk using features UP TO day t-1.
    # So we MUST NOT leak day t's rainfall/humidity into the input features if we are predicting day t.
    # However, the dataset builder typically builds X up to t, and y is t+lead_time.
    # V7 specification: "output: risk (day t)".
    # We will build features for day t using data up to day t.
    
    # Rolling Features
    for w in [3, 5, 7, 14]:
        df[f"RH2M_mean_{w}"] = df["RH2M"].rolling(w).mean()
    
    for w in [3, 5, 7]:
        df[f"rainfall_sum_{w}"] = df["PRECTOTCORR"].rolling(w).sum()
        
    for w in [3, 5]:
        df[f"temp_std_{w}"] = df["T2M"].rolling(w).std()
        
    # Trend Features
    df["RH2M_diff_1"] = df["RH2M"].diff()
    df["RH2M_diff_3"] = df["RH2M"].diff(3)
    df["rainfall_diff_3"] = df["PRECTOTCORR"].diff(3)
    df["RH2M_accel"] = df["RH2M_diff_1"].diff()
    df["RH2M_trend_3"] = rolling_slope(df["RH2M"], window=3)
    
    # Dynamic Biological Features
    df["high_humidity_flag"] = (df["RH2M"] > 80).astype(int)
    
    # Vectorized humidity streak
    streak = df["high_humidity_flag"].copy()
    streak_arr = streak.values
    for i in range(1, len(streak_arr)):
        if streak_arr[i] > 0:
            streak_arr[i] += streak_arr[i-1]
    df["humidity_streak"] = streak_arr
    
    # Rainfall Spike
    # Prompt: rainfall_today > 2x rolling_mean_5 (stricter interpretation)
    # Require a minimum biological rain threshold (e.g. > 5mm) to filter out drizzle noise
    past_mean_5 = df["PRECTOTCORR"].rolling(5).mean().shift(1).fillna(0)
    df["rainfall_spike"] = (df["PRECTOTCORR"] > 2 * past_mean_5) & (df["PRECTOTCORR"] > 5.0)
    
    # Rainfall Onset
    # rain_today > 0 AND last 3 days = 0. Also require > 5mm to be a meaningful onset.
    past_rain_3 = df["PRECTOTCORR"].rolling(3).sum().shift(1).fillna(0)
    df["rainfall_onset"] = (df["PRECTOTCORR"] > 5.0) & (past_rain_3 == 0)
    
    # Temp Stability
    df["temp_stability"] = df["temp_std_5"] < 1.5
    
    return df

def generate_labels(df):
    print("STEP 3-5: BIOLOGICAL STATE AND LABEL GENERATION")
    
    df["state"] = "DRY"
    
    # HUMID_BUILDUP
    buildup_mask = (df["RH2M"] > 75) & (df["RH2M_diff_1"] > 0)
    df.loc[buildup_mask, "state"] = "HUMID_BUILDUP"
    
    # TRIGGER_PHASE
    # CRITICAL: Incorporate transition features! 
    # "Constant humidity = fungal survival, NOT outbreak trigger" -> Require RH2M_diff_3 > 0 (Transitioning upward)
    trigger_mask = (
        (df["humidity_streak"] >= 4) & 
        (df["rainfall_spike"] | df["rainfall_onset"]) & 
        (df["temp_stability"] == True) &
        (df["RH2M_diff_3"] > 0)  # Must be a dynamic buildup phase, not a flat plateau
    )
    df.loc[trigger_mask, "state"] = "TRIGGER_PHASE"
    
    # SATURATED_MONSOON
    saturated_mask = (df["humidity_streak"] >= 10) & ~(df["rainfall_spike"] | df["rainfall_onset"])
    df.loc[saturated_mask, "state"] = "SATURATED_MONSOON"
    
    # LABEL GENERATION
    df["risk"] = 0
    df.loc[df["state"] == "TRIGGER_PHASE", "risk"] = 1
    
    # MONSOON NOISE SUPPRESSION (HARD NEGATIVE)
    noise_mask = (df["humidity_streak"] >= 10) & (df["rainfall_spike"] == False)
    df.loc[noise_mask, "risk"] = 0
    
    # PRE-RISK LABEL
    # humidity_streak in [2, 4], increasing humidity trend (RH2M_trend_3 > 0), rainfall NOT yet triggered
    pre_risk_mask = (df["humidity_streak"].between(2, 4)) & (df["RH2M_trend_3"] > 0) & ~(df["rainfall_spike"] | df["rainfall_onset"])
    df["pre_risk"] = pre_risk_mask.astype(int)
    
    return df

def build_sequences(df):
    print("STEP 6: SEQUENCE BUILDING")
    # To prevent look-ahead leakage, when predicting risk at day t, the sequence X must strictly use days [t-14 ... t-1].
    # "input: last 14 days -> output: risk (day t)"
    
    seq_len = 14
    df_clean = df.dropna().reset_index(drop=True)
    
    feature_cols = [c for c in df_clean.columns if c not in ["date", "state", "risk", "pre_risk"]]
    
    X_list, y_list, dates_list = [], [], []
    
    for i in range(seq_len, len(df_clean)):
        # x is strictly the 14 days PRIOR to day i
        window = df_clean.iloc[i-seq_len : i]
        target = df_clean.iloc[i]
        
        X_list.append(window[feature_cols].values)
        y_list.append(target["risk"])
        dates_list.append(target["date"])
        
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    dates = pd.DatetimeIndex(dates_list)
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    return X, y, dates, df_clean

def run_pipeline():
    df = load_and_clean_data(RAW_DATA_PATH)
    df = engineer_features(df)
    df = generate_labels(df)
    X, y, dates, df_clean = build_sequences(df)
    
    print("STEP 7: SAVE OUTPUT")
    features_path = os.path.join(PROCESSED_DIR, "features.csv")
    labels_path = os.path.join(PROCESSED_DIR, "labels.csv")
    seq_path = os.path.join(PROCESSED_DIR, "sequences.npz")
    
    df_clean.to_csv(features_path, index=False)
    pd.DataFrame({"date": dates, "risk": y}).to_csv(labels_path, index=False)
    np.savez_compressed(seq_path, X=X, y=y, dates=dates.astype(str))
    
    print("\nSTEP 8: VALIDATION CHECKS (AUTO-FAIL SYSTEM)")
    df_eval = pd.DataFrame({"date": dates, "risk": y})
    df_eval["year"] = df_eval["date"].dt.year
    df_eval["month"] = df_eval["date"].dt.month
    
    yearly_events = df_eval[df_eval["risk"] == 1].groupby("year").size()
    total_years = df_eval["year"].nunique()
    
    print("\n--- Total Risk Events per Year ---")
    if yearly_events.empty:
        print("NO RISK EVENTS GENERATED!")
    else:
        for year, count in yearly_events.items():
            first_date = df_eval[(df_eval["year"] == year) & (df_eval["risk"] == 1)]["date"].min().date()
            print(f"Year {year}: {count} events (First event: {first_date})")
    
    years_with_events = len(yearly_events)
    print(f"\nYears with events: {years_with_events} / {total_years}")
    
    if years_with_events == total_years:
        raise AssertionError("FAIL: All years have risk events. Pipeline did not distinguish weak/strong years.")
        
    august_events = df_eval[(df_eval["risk"] == 1) & (df_eval["month"] == 8)]
    total_events = df_eval["risk"].sum()
    august_ratio = len(august_events) / total_events if total_events > 0 else 0
    
    print(f"August Events: {len(august_events)} / {total_events} ({august_ratio:.2%})")
    
    if august_ratio > 0.8:
        raise AssertionError(f"FAIL: Risk events are overwhelmingly clustered in August ({august_ratio:.2%}). Calendar bias detected.")
        
    if len(yearly_events.unique()) == 1 and total_events > 0:
        raise AssertionError("FAIL: Identical event counts across years.")
        
    if total_events == 0:
        print("WARNING: Zero events generated. Adjust biological thresholds.")
    else:
        print("\n[SUCCESS] VALIDATION PASSED. Pipeline generated a biologically grounded, highly selective dataset.")

if __name__ == "__main__":
    run_pipeline()
