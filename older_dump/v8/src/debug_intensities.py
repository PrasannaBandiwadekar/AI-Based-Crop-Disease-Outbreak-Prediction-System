import pandas as pd
import numpy as np
import os

def debug_intensities():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS_PATH = os.path.join(BASE_DIR, "outputs", "v8_backtest_results.csv")
    FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "features.csv")
    GT_PATH = "e:/crop-disease-outbreak-prediction-system-feature-zip-changes/crop-disease-outbreak-prediction-system-feature-zip-changes/model_train/research_comp/evidence_base/outbreak_events/sangli_synthetic_gt.csv"
    
    res = pd.read_csv(RESULTS_PATH)
    feat = pd.read_csv(FEATURES_PATH)
    gt = pd.read_csv(GT_PATH)
    
    res["date"] = pd.to_datetime(res["date"])
    feat["date"] = pd.to_datetime(feat["date"])
    gt["peak_start"] = pd.to_datetime(gt["peak_start"])
    
    df = pd.merge(res, feat, on="date")
    df["intensity"] = df["RH2M"] * df["PRECTOTCORR"]
    
    # Identify Hits (approximate)
    hits = []
    for _, g in gt.iterrows():
        peak = g["peak_start"]
        win_start = peak - pd.Timedelta(days=10)
        win_end = peak - pd.Timedelta(days=2)
        
        window_alerts = df[(df["date"] >= win_start) & (df["date"] <= win_end) & (df["alert"] == True)]
        if not window_alerts.empty:
            hits.append({
                "peak": peak,
                "max_intensity": window_alerts["intensity"].max(),
                "alert_count": len(window_alerts)
            })
            
    print(f"Total Hits in window [peak-10, peak-2]: {len(hits)}")
    for h in hits:
        print(f"Hit at {h['peak']}: Max Intensity = {h['max_intensity']:.2f}")
        
    # Get all potential event intensities
    # (Simple clustering for debug)
    df["group"] = (df["alert"] != df["alert"].shift()).cumsum()
    events = df[df["alert"] == True].groupby("group").agg({
        "date": ["min", "max", "count"],
        "intensity": "max"
    })
    
    print(f"\nTotal potential events: {len(events)}")
    print("Intensity Distribution of all events:")
    print(events["intensity"]["max"].describe())
    
    print("\nTop 10 intensities:")
    print(events["intensity"]["max"].sort_values(ascending=False).head(10))

if __name__ == "__main__":
    debug_intensities()
