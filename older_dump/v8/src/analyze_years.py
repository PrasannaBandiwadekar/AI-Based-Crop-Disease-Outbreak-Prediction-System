import pandas as pd
import numpy as np
import os

def analyze_years():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS_PATH = os.path.join(BASE_DIR, "outputs", "v8_filtered_results.csv")
    results = pd.read_csv(RESULTS_PATH)
    results["date"] = pd.to_datetime(results["date"])
    results["year"] = results["date"].dt.year
    
    alert_days = results[results["alert"] == True].copy()
    if alert_days.empty:
        print("No alerts.")
        return
        
    # Group by event
    alert_days["group"] = (alert_days["date"].diff() > pd.Timedelta(days=3)).cumsum()
    events = alert_days.groupby("group").agg({
        "year": "first",
        "date": ["min", "max", "count"]
    })
    
    print("Event distribution by year:")
    print(events["year"]["first"].value_counts().sort_index())

if __name__ == "__main__":
    analyze_years()
