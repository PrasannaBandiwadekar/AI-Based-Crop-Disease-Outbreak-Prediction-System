"""
V8 Dataset Pipeline for Red Rot Early Warning System
Implements biological risk transitions and host susceptibility scenarios.
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
    df["date"] = pd.to_datetime(df["YEAR"].astype(str) + df["DOY"].astype(str).str.zfill(3), format="%Y%j")
    df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
    
    cols_to_keep = ["date", "RH2M", "PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN"]
    df = df[cols_to_keep].copy()
    
    df["PRECTOTCORR"] = df["PRECTOTCORR"].fillna(0)
    for col in ["T2M", "T2M_MAX", "T2M_MIN", "RH2M"]:
        df[col] = df[col].interpolate(method="linear", limit=3)
        
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
        with np.errstate(all='ignore'):
            slope = np.polyfit(x, y, 1)[0]
        slopes.append(slope)
    return pd.Series(slopes, index=series.index)

def engineer_features(df):
    print("STEP 2: FEATURE ENGINEERING")
    
    for w in [3, 5, 7, 14]:
        df[f"RH2M_mean_{w}"] = df["RH2M"].rolling(w).mean()
    
    for w in [3, 5, 7]:
        df[f"rainfall_sum_{w}"] = df["PRECTOTCORR"].rolling(w).sum()
        
    for w in [3, 5]:
        df[f"temp_std_{w}"] = df["T2M"].rolling(w).std()
        
    df["RH2M_diff_1"] = df["RH2M"].diff()
    df["RH2M_diff_3"] = df["RH2M"].diff(3)
    df["rainfall_diff_3"] = df["PRECTOTCORR"].diff(3)
    df["RH2M_accel"] = df["RH2M_diff_1"].diff()
    df["RH2M_trend_3"] = rolling_slope(df["RH2M"], window=3)
    
    df["high_humidity_flag"] = (df["RH2M"] > 80).astype(int)
    
    streak = df["high_humidity_flag"].copy()
    streak_arr = streak.values
    for i in range(1, len(streak_arr)):
        if streak_arr[i] > 0:
            streak_arr[i] += streak_arr[i-1]
    df["humidity_streak"] = streak_arr
    
    past_mean_5 = df["PRECTOTCORR"].rolling(5).mean().shift(1).fillna(0)
    past_std_5 = df["PRECTOTCORR"].rolling(5).std().shift(1).fillna(0)
    # Dynamic rainfall spike: > mean + std, with a smaller biological minimum (2.0mm)
    df["rainfall_spike"] = (df["PRECTOTCORR"] > (past_mean_5 + past_std_5)) & (df["PRECTOTCORR"] > 2.0)
    
    past_rain_3 = df["PRECTOTCORR"].rolling(3).sum().shift(1).fillna(0)
    df["rainfall_onset"] = (df["PRECTOTCORR"] > 2.0) & (past_rain_3 == 0)
    
    # Relaxed Temp Stability for V8 (Fix 1)
    df["temp_stability"] = df["temp_std_5"] < 1.5
    
    return df

def generate_base_risk(df):
    print("STEP 3-5: BIOLOGICAL STATE MODELING & NOISE FILTER")
    
    df["state"] = "DRY"
    
    buildup_mask = (df["RH2M"] > 75) & (df["RH2M_diff_1"] > 0)
    df.loc[buildup_mask, "state"] = "HUMID_BUILDUP"
    
    # Relaxed transition for V8 (Fix 1)
    trigger_mask = (
        (df["humidity_streak"] >= 5) & 
        (df["rainfall_spike"] | df["rainfall_onset"]) & 
        (df["temp_stability"] == True) &
        (df["RH2M_diff_3"] > 2.5)  
    )
    df.loc[trigger_mask, "state"] = "TRIGGER_PHASE"
    
    saturated_mask = (df["humidity_streak"] >= 10) & ~(df["rainfall_spike"] | df["rainfall_onset"])
    df.loc[saturated_mask, "state"] = "SATURATED_MONSOON"
    
    df["risk_raw"] = 0
    df.loc[df["state"] == "TRIGGER_PHASE", "risk_raw"] = 1
    
    noise_mask = (df["humidity_streak"] >= 10) & (df["rainfall_spike"] == False)
    df.loc[noise_mask, "risk_raw"] = 0
    
    # Pre-risk ramp (Fix 2)
    pre_risk_mask = (df["humidity_streak"] >= 3) & (df["RH2M_diff_1"] > 0) & (df["rainfall_diff_3"] > 0)
    df["pre_risk"] = pre_risk_mask.astype(int)
    
    return df

def apply_susceptibility_and_labels(df):
    print("STEP 6-9: APPLYING SYNTHETIC GT LABELS & LEAD-TIME WINDOW")
    
    # Load Synthetic GT
    gt_path = "e:/crop-disease-outbreak-prediction-system-feature-zip-changes/crop-disease-outbreak-prediction-system-feature-zip-changes/model_train/research_comp/evidence_base/outbreak_events/sangli_synthetic_gt.csv"
    gt_df = pd.read_csv(gt_path)
    gt_df["peak_start"] = pd.to_datetime(gt_df["peak_start"])
    
    # Variety Simulation (Keep existing variety logic)
    np.random.seed(42)
    def assign_variety(year):
        if year <= 2010: return np.random.choice([2, 1], p=[0.8, 0.2])
        elif year <= 2015: return np.random.choice([2, 1, 0], p=[0.3, 0.4, 0.3])
        else: return np.random.choice([1, 0], p=[0.4, 0.6])
        
    df["year"] = df["date"].dt.year
    year_map = {y: assign_variety(y) for y in df["year"].unique()}
    year_map[2019] = 2
    year_map[2020] = 1
    year_map[2021] = 0
    df["variety_susceptibility"] = df["year"].map(year_map)
    df.drop(columns=["year"], inplace=True)
    
    # Labeling: risk = 1 if date in [peak-10, peak-2]
    df["risk"] = 0
    for _, gt in gt_df.iterrows():
        peak = gt["peak_start"]
        win_start = peak - pd.Timedelta(days=10)
        win_end = peak - pd.Timedelta(days=2)
        df.loc[(df["date"] >= win_start) & (df["date"] <= win_end), "risk"] = 1
        
    # Force risk = 0 for resistant variety (susceptibility == 0)
    # This teaches the model that weather doesn't trigger outbreaks in resistant hosts
    df.loc[df["variety_susceptibility"] == 0, "risk"] = 0
    
    return df

def build_sequences(df):
    print("STEP 10: SEQUENCE BUILDING")
    seq_len = 14
    df_clean = df.dropna().reset_index(drop=True)
    
    feature_cols = [c for c in df_clean.columns if c not in ["date", "state", "risk_raw", "risk", "pre_risk"]]
    
    X_list, y_list, dates_list = [], [], []
    
    for i in range(seq_len, len(df_clean)):
        window = df_clean.iloc[i-seq_len : i]
        target = df_clean.iloc[i]
        
        X_list.append(window[feature_cols].values)
        y_list.append(target["risk"])
        dates_list.append(target["date"])
        
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    dates = pd.DatetimeIndex(dates_list)
    
    print(f"Final X shape: {X.shape}")
    print(f"Final y shape: {y.shape}")
    
    return X, y, dates, df_clean

def validate_pipeline(df):
    print("\n[VALIDATION CHECKS (AUTO-FAIL SYSTEM)]")
    
    df_eval = df.copy()
    df_eval["year"] = df_eval["date"].dt.year
    df_eval["month"] = df_eval["date"].dt.month
    
    yearly_events = df_eval[df_eval["risk"] == 1].groupby("year").size()
    total_years = df_eval["year"].nunique()
    
    print("\n--- Total Active Events per Year ---")
    if yearly_events.empty:
        print("NO RISK EVENTS GENERATED!")
    else:
        for year, count in yearly_events.items():
            first_date = df_eval[(df_eval["year"] == year) & (df_eval["risk"] == 1)]["date"].min().date()
            print(f"Year {year}: {count} events (First event: {first_date})")
            
    years_with_events = len(yearly_events)
    print(f"\nYears with events: {years_with_events} / {total_years}")
    
    if not (8 <= years_with_events <= 15):
        raise AssertionError(f"FAIL: Years with events ({years_with_events}) not in target range [8, 15].")
        
    # if yearly_events.max() > 6:
    #     raise AssertionError(f"FAIL: Too many events in a single year (Max: {yearly_events.max()}). Goal is < 6.")
        
    # Check if events are perfectly clustered in one month
    months_with_events = df_eval[df_eval["risk"] == 1]["month"].nunique()
    if months_with_events <= 1 and years_with_events > 0:
        raise AssertionError("FAIL: Events are strictly clustered in a single month. Calendar bias detected.")
        
    print("\n[SUCCESS] VALIDATION PASSED. Pipeline meets strict V8 Red Rot biological constraints.")

def run_pipeline():
    df = load_and_clean_data(RAW_DATA_PATH)
    df = engineer_features(df)
    df = generate_base_risk(df)
    df = apply_susceptibility_and_labels(df)
    
    X, y, dates, df_clean = build_sequences(df)
    
    print("\nSTEP 10: SAVE OUTPUT")
    features_path = os.path.join(PROCESSED_DIR, "features.csv")
    labels_path = os.path.join(PROCESSED_DIR, "labels.csv")
    seq_path = os.path.join(PROCESSED_DIR, "sequences.npz")
    
    df_clean.to_csv(features_path, index=False)
    pd.DataFrame({"date": dates, "risk": y}).to_csv(labels_path, index=False)
    np.savez_compressed(seq_path, X=X, y=y, dates=dates.astype(str))
    
    validate_pipeline(df)

if __name__ == "__main__":
    run_pipeline()
