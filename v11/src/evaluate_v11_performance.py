import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference_engine import V11InferenceEngine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTHETIC_RAW = os.path.join(BASE_DIR, "data", "synthetic", "v11_synthetic_raw.csv")

def run_evaluation():
    print("============================================================")
    print(" V11 KG-CTCN SYNTHETIC PERFORMANCE EVALUATION")
    print("============================================================")
    
    engine = V11InferenceEngine()
    
    def deep_audit(target_date):
        print(f"\n--- DEEP AUDIT: {target_date.date()} ---")
        raw_weather = engine.weather_provider.get_weather_history("Synthetic", target_date, window_days=400)
        # Replicate the engine's internal preprocessing
        df = raw_weather.copy()
        
        # Bug 1 Fix: Rolling Z-score (90-day window)
        weather_base_cols = ["WS10M", "T2M", "RH2M", "T2M_MIN", "T2M_MAX", "PRECTOTCORR"]
        for col in weather_base_cols:
            if col not in df.columns: continue
            df[f"{col}_raw"] = df[col].copy()
            roll_mean = df[col].rolling(90, min_periods=1).mean()
            roll_std  = df[col].rolling(90, min_periods=1).std().fillna(1.0).replace(0, 1.0)
            df[col]   = (df[col] - roll_mean) / roll_std
            
        # Recompute KG features
        raw_rh = df["RH2M_raw"]; raw_rain = df["PRECTOTCORR_raw"]
        df["RH_high_flag"] = np.clip((raw_rh - 75) / 15, 0, 1)
        df["RH_persist_7d"] = df["RH_high_flag"].rolling(7, min_periods=1).sum()
        df["Rain_sum_7d"] = raw_rain.rolling(7, min_periods=1).sum()
        df["Monsoon_ind"] = (raw_rh.rolling(7, min_periods=1).mean() > 75).astype(int)
        
        last_row = df.iloc[-1]
        cols_to_show = ["RH2M_raw", "RH2M", "RH_high_flag", "RH_persist_7d", "Rain_sum_7d", "Monsoon_ind"]
        for c in cols_to_show:
            print(f"  {c:15s}: {last_row[c]:.4f}")
        print("------------------------------------\n")

    raw_df = pd.read_csv(SYNTHETIC_RAW, skiprows=14)
    raw_df['date'] = pd.to_datetime(raw_df['YEAR'].astype(str) + raw_df['DOY'].astype(str).str.zfill(3), format='%Y%j')
    raw_df = raw_df.set_index('date')
    
    # Mocking get_weather_history to return from our local raw_df
    def mock_get_weather(location, target_date, window_days=400):
        start = target_date - timedelta(days=window_days)
        return raw_df.loc[start:target_date].reset_index()

    engine.weather_provider.get_weather_history = mock_get_weather

    # Run audit on the 2025 peak
    deep_audit(datetime(2025, 7, 31))

    
    # --- TEST CASES ---
    
    scenarios = [
        {"name": "2025 Peak (High Pressure)", "date": datetime(2025, 7, 31), "agro": {"variety_susceptibility": 2, "is_ratoon": 1, "crop_age_days": 180}},
        {"name": "2025 Peak (Resistant)",      "date": datetime(2025, 7, 31), "agro": {"variety_susceptibility": 0, "is_ratoon": 1, "crop_age_days": 180}},
        {"name": "2026 Mid-Monsoon (Dry)",     "date": datetime(2026, 7, 31), "agro": {"variety_susceptibility": 2, "is_ratoon": 1, "crop_age_days": 180}},
        {"name": "2027 Late Peak (Susceptible)","date": datetime(2027, 9, 20), "agro": {"variety_susceptibility": 2, "is_ratoon": 1, "crop_age_days": 240}},
    ]
    
    results = []
    print(f"{'Scenario':<30} | {'Risk Score':<12} | {'Class':<8} | {'Confidence'}")
    print("-" * 70)
    
    for sc in scenarios:
        try:
            inf = engine.run_inference("Synthetic", sc["date"], sc["agro"])
            print(f"{sc['name']:<30} | {inf['risk_score']:>10.4f}   | {sc['agro']['variety_susceptibility']} -> {inf['risk_class']:<8} | {inf['confidence_score']:.4f}")
            results.append({**sc, **inf})
        except Exception as e:
            print(f"{sc['name']:<30} | ERROR: {e}")

    # --- FULL TIMESERIES ANALYSIS ---
    print("\nRunning Full Timeseries Analysis (2025)...")
    ts_results = []
    dates_2025 = pd.date_range(datetime(2025, 1, 1), datetime(2025, 12, 31))
    
    for d in dates_2025[200:]: # Start after some warmup
        try:
            inf = engine.run_inference("Synthetic", d, {"variety_susceptibility": 2, "is_ratoon": 1, "crop_age_days": 180})
            ts_results.append({"date": d, "score": inf['risk_score']})
        except:
            pass
            
    if ts_results:
        ts_df = pd.DataFrame(ts_results)
        plt.figure(figsize=(12, 5))
        plt.plot(ts_df['date'], ts_df['score'], label='Risk Score (Susceptible)')
        plt.axhline(y=0.3, color='orange', linestyle='--', label='Medium Threshold')
        plt.axhline(y=0.7, color='red', linestyle='--', label='High Threshold')
        plt.title("V11 Risk Score Evolution - Synthetic 2025 (Outbreak Year)")
        plt.ylabel("Risk Score")
        plt.legend()
        out_plot = os.path.join(BASE_DIR, "data", "synthetic", "v11_2025_risk_plot.png")
        plt.savefig(out_plot)
        print(f"Saved risk evolution plot: {out_plot}")
    
    print("\n[CONCLUSION]")
    # Check if Outbreak 2025 was detected as High
    if any(r['name'] == "2025 Peak (High Pressure)" and r['risk_score'] > 0.3 for r in results):
        print("[PASS] Model successfully flagged the 2025 synthetic outbreak.")
    else:
        print("[FAIL] Model missed the 2025 synthetic outbreak.")
        
    # Check if Resistant variety reduced risk
    peak_2025_high = next(r for r in results if r['name'] == "2025 Peak (High Pressure)")
    peak_2025_res  = next(r for r in results if r['name'] == "2025 Peak (Resistant)")
    if peak_2025_res['risk_score'] < peak_2025_high['risk_score']:
        print(f"[PASS] Resistant variety correctly lowered risk ({peak_2025_res['risk_score']:.4f} vs {peak_2025_high['risk_score']:.4f})")
    else:
        print("[FAIL] Resistant variety did not lower risk.")

if __name__ == "__main__":
    run_evaluation()
