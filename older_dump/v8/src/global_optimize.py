import pandas as pd
import numpy as np
import os
import sys
import io
from contextlib import redirect_stdout

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from v8_evaluator import run_v8_evaluation

def global_optimize():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS_PATH = os.path.join(BASE_DIR, "outputs", "v8_backtest_results.csv")
    FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "features.csv")
    FILTERED_RESULTS = os.path.join(BASE_DIR, "outputs", "v8_filtered_results.csv")
    
    res = pd.read_csv(RESULTS_PATH)
    feat = pd.read_csv(FEATURES_PATH)
    res["date"] = pd.to_datetime(res["date"])
    feat["date"] = pd.to_datetime(feat["date"])
    df_base = pd.merge(res, feat, on="date")
    
    best_res = None
    best_score = -1
    
    # Sweep Threshold
    for t in [0.4, 0.5, 0.6, 0.7, 0.8]:
        df_t = df_base.copy()
        df_t["alert"] = df_t["risk_score"] >= t
        
        # Static Filters
        df_t.loc[df_t["variety_susceptibility"] == 0, "alert"] = False
        df_t.loc[(df_t["humidity_streak"] >= 10) & (df_t["rainfall_spike"] == False), "alert"] = False
        
        # Cluster
        alert_days = df_t[df_t["alert"] == True].sort_values("date").copy()
        events = []
        if not alert_days.empty:
            curr = [alert_days.iloc[0].to_dict()]
            for i in range(1, len(alert_days)):
                if (alert_days.iloc[i]["date"] - alert_days.iloc[i-1]["date"]).days <= 3:
                    curr.append(alert_days.iloc[i].to_dict())
                else:
                    events.append(curr)
                    curr = [alert_days.iloc[i].to_dict()]
            events.append(curr)
            
        # Persistence
        surviving_p = []
        for e in events:
            edf = pd.DataFrame(e)
            if (edf["date"].max() - edf["date"].min()).days + 1 >= 3:
                surviving_p.append({
                    "event": e,
                    "max_intensity": (edf["RH2M"] * edf["PRECTOTCORR"]).max()
                })
        
        if not surviving_p: continue
        
        # Sweep Percentile
        for p in range(0, 100, 10):
            intensities = [e["max_intensity"] for e in surviving_p]
            p_thresh = np.percentile(intensities, p)
            
            surviving_dates = set()
            count = 0
            for e in surviving_p:
                if e["max_intensity"] >= p_thresh:
                    count += 1
                    for day in e["event"]:
                        surviving_dates.add(day["date"])
            
            res["alert"] = res["date"].isin(surviving_dates)
            res.to_csv(FILTERED_RESULTS, index=False)
            
            f = io.StringIO()
            with redirect_stdout(f):
                run_v8_evaluation(FILTERED_RESULTS)
            out = f.getvalue()
            
            try:
                recall = float(out.split("Recall:")[1].split("%")[0].strip())
                fpr = float(out.split("False Positive Rate (FPR):")[1].split("%")[0].strip())
                lead = float(out.split("Avg Lead Time (Days):")[1].split("\n")[0].strip())
                
                # Metric: Score = Recall - FPR/2
                score = recall - (fpr / 2)
                
                if recall >= 40:
                    print(f"T={t:.1f} P={p} | Rec={recall:.1f}% FPR={fpr:.1f}% Lead={lead:.1f} | SCORE={score:.1f}")
                    if score > best_score:
                        best_score = score
                        best_res = (t, p, recall, fpr, lead)
            except:
                continue
                
    print(f"\nBEST CONFIG: T={best_res[0]} P={best_res[1]} | Recall={best_res[2]}% FPR={best_res[3]}% Lead={best_res[4]}")

if __name__ == "__main__":
    global_optimize()
