import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src import config

def compute_metrics():
    print("Computing V5 Validation Metrics...")
    
    # 1. Load Data
    results_path = os.path.join(config.OUTPUTS_DIR, "backtest_results.csv")
    if not os.path.exists(results_path):
        raise FileNotFoundError("Run backtest_engine.py first.")
    
    results_df = pd.read_csv(results_path)
    results_df["date"] = pd.to_datetime(results_df["date"])
    
    # Load Outbreak Events
    events_path = "e:/crop-disease-outbreak-prediction-system-feature-zip-changes/crop-disease-outbreak-prediction-system-feature-zip-changes/model_train/research_comp/evidence_base/outbreak_events/red_rot_outbreak_events.csv"
    events_df = pd.read_csv(events_path)
    events_df["peak_start"] = pd.to_datetime(events_df["peak_start"])
    
    # 2. Event-Level Recall
    # An outbreak is detected IF at least 1 alert occurs in [peak-7, peak-3]
    detected_count = 0
    total_events = len(events_df)
    
    for _, event in events_df.iterrows():
        peak_date = event["peak_start"]
        valid_window_start = peak_date - pd.Timedelta(days=7)
        valid_window_end = peak_date - pd.Timedelta(days=3)
        
        alerts_in_window = results_df[(results_df["date"] >= valid_window_start) & 
                                      (results_df["date"] <= valid_window_end) & 
                                      (results_df["alert"] == True)]
        
        if not alerts_in_window.empty:
            detected_count += 1
            
    recall = detected_count / total_events if total_events > 0 else 0
    miss_rate = 1 - recall
    
    # 3. False Alert Rate
    # Any alert outside of any [peak-14, peak+7] window (broad window to be fair)
    results_df["is_near_event"] = False
    for _, event in events_df.iterrows():
        peak_date = event["peak_start"]
        mask = (results_df["date"] >= peak_date - pd.Timedelta(days=14)) & \
               (results_df["date"] <= peak_date + pd.Timedelta(days=7))
        results_df.loc[mask, "is_near_event"] = True
        
    false_alerts = results_df[(results_df["alert"] == True) & (results_df["is_near_event"] == False)]
    total_alerts = results_df["alert"].sum()
    false_alert_rate = len(false_alerts) / total_alerts if total_alerts > 0 else 0
    
    metrics = {
        "event_level_recall": recall,
        "miss_rate": miss_rate,
        "false_alert_rate": false_alert_rate,
        "total_events": total_events,
        "detected_events": detected_count,
        "false_alerts_count": len(false_alerts)
    }
    
    print(f"Metrics Computed: Recall={recall:.2%}, Miss Rate={miss_rate:.2%}")
    return metrics

if __name__ == "__main__":
    compute_metrics()
