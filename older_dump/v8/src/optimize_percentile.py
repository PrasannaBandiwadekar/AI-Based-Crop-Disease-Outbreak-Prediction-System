import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from post_processing_filter import apply_filters
from v8_evaluator import run_v8_evaluation

def optimize():
    # Load raw results
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FILTERED_RESULTS = os.path.join(BASE_DIR, "outputs", "v8_filtered_results.csv")
    
    # We need to hack post_processing_filter to accept a percentile
    # Or just copy the logic here
    
    RESULTS_PATH = os.path.join(BASE_DIR, "outputs", "v8_backtest_results.csv")
    FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "features.csv")
    
    results_df = pd.read_csv(RESULTS_PATH)
    features_df = pd.read_csv(FEATURES_PATH)
    results_df["date"] = pd.to_datetime(results_df["date"])
    features_df["date"] = pd.to_datetime(features_df["date"])
    df_base = pd.merge(results_df, features_df, on="date")
    
    # Apply variety and plateau once
    df_base.loc[df_base["variety_susceptibility"] == 0, "alert"] = False
    df_base.loc[(df_base["humidity_streak"] >= 10) & (df_base["rainfall_spike"] == False), "alert"] = False
    
    # Form events
    alert_days = df_base[df_base["alert"] == True].sort_values("date").copy()
    events = []
    if not alert_days.empty:
        current_event = [alert_days.iloc[0].to_dict()]
        for i in range(1, len(alert_days)):
            if (alert_days.iloc[i]["date"] - alert_days.iloc[i-1]["date"]).days <= 3:
                current_event.append(alert_days.iloc[i].to_dict())
            else:
                events.append(current_event)
                current_event = [alert_days.iloc[i].to_dict()]
        events.append(current_event)
    
    # Persistence Filter
    surviving_persistence = []
    for e in events:
        edf = pd.DataFrame(e)
        duration = (edf["date"].max() - edf["date"].min()).days + 1
        if duration >= 3:
            surviving_persistence.append({
                "event": e,
                "max_intensity": (edf["RH2M"] * edf["PRECTOTCORR"]).max()
            })
            
    print(f"Events after persistence: {len(surviving_persistence)}")
    
    # Sweep percentile
    best_res = None
    best_fpr = 1.0
    
    for p in range(0, 100, 5):
        intensities = [e["max_intensity"] for e in surviving_persistence]
        thresh = np.percentile(intensities, p)
        
        surviving_dates = set()
        count = 0
        for e in surviving_persistence:
            if e["max_intensity"] >= thresh:
                count += 1
                for day in e["event"]:
                    surviving_dates.add(day["date"])
        
        results_df["alert"] = results_df["date"].isin(surviving_dates)
        results_df.to_csv(FILTERED_RESULTS, index=False)
        
        # Capture stdout to get results
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            run_v8_evaluation(FILTERED_RESULTS)
        out = f.getvalue()
        
        # Parse metrics
        try:
            recall = float(out.split("Recall:")[1].split("%")[0].strip())
            fpr = float(out.split("False Positive Rate (FPR):")[1].split("%")[0].strip())
            predicted = int(out.split("Total Predicted Events:")[1].split("\n")[0].strip())
            
            print(f"P={p:02d} | Events={predicted} | Recall={recall:.2f}% | FPR={fpr:.2f}%")
            
            if recall >= 40 and fpr < best_fpr:
                best_fpr = fpr
                best_res = (p, recall, fpr)
        except:
            continue
            
    print(f"\nBest Percentile: {best_res}")

if __name__ == "__main__":
    optimize()
