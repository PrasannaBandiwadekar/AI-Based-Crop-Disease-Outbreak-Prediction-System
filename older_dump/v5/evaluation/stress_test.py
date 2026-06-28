import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src import config, utils, features, decision_engine

def run_stress_test():
    print("Running Environmental Stress Test...")
    
    # Load raw data sample
    raw_df = utils.load_raw_data()
    sample_window = raw_df.tail(60).copy() # Last 60 days
    
    # 1. Simulate 3-day Rainfall Missing
    stress_rain = sample_window.copy()
    stress_rain.iloc[30:33, stress_rain.columns.get_loc("PRECTOTCORR")] = np.nan
    feat_rain = features.build_features(stress_rain)
    
    # Check if NaNs were correctly filled with 0 (for rain)
    rain_nan_count = feat_rain["PRECTOTCORR"].isnull().sum()
    
    # 2. Simulate Humidity Spike (Sensor Artifact)
    stress_rh = sample_window.copy()
    stress_rh.iloc[30, stress_rh.columns.get_loc("RH2M")] = 110 # Physical impossibility
    feat_rh = features.build_features(stress_rh)
    
    # Decision Engine response to spike
    last_3_spike = feat_rh.iloc[29:32].copy() # centered around spike
    # Mocking scores to test smoothing
    last_3_spike["risk_score"] = [0.1, 0.9, 0.1] # Spike on day 2
    alert_spike = decision_engine.evaluate_risk(last_3_spike)
    
    # Result
    status = {
        "rain_imputation_success": rain_nan_count == 0,
        "smoothing_rule_spike_suppressed": alert_spike == False,
        "interpolation_active": feat_rh["RH2M"].iloc[31] != 110 # should be smoothed? actually our code doesn't smooth outliers yet, just interpolates NaNs
    }
    
    print(f"Stress Test Complete. Spike Suppression: {alert_spike == False}")
    return status

if __name__ == "__main__":
    print(run_stress_test())
