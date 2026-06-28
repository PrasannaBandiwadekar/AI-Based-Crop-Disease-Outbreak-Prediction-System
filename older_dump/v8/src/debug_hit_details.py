import pandas as pd
import numpy as np
import os

def debug_hit_details():
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
    
    # Simple clustering
    df["group"] = (df["alert"] != df["alert"].shift()).cumsum()
    events = df[df["alert"] == True].groupby("group").agg({
        "date": ["min", "max", "count"],
        "intensity": "max"
    })
    events.columns = ["start", "end", "count", "max_intensity"]
    events["duration"] = (events["end"] - events["start"]).dt.days + 1
    
    hits = []
    for _, g in gt.iterrows():
        peak = g["peak_start"]
        win_start = peak - pd.Timedelta(days=10)
        win_end = peak - pd.Timedelta(days=2)
        
        # Check matching events (onset in window)
        match = events[(events["start"] >= win_start) & (events["start"] <= win_end)]
        if not match.empty:
            for _, m in match.iterrows():
                hits.append({
                    "peak": peak,
                    "event_start": m["start"],
                    "duration": m["duration"],
                    "intensity": m["max_intensity"]
                })
                
    print(f"Total evaluator hits: {len(hits)}")
    for h in hits:
        print(f"Hit at {h['peak']}: Start={h['event_start']}, Dur={h['duration']}, Int={h['intensity']:.2f}")

if __name__ == "__main__":
    debug_hit_details()
