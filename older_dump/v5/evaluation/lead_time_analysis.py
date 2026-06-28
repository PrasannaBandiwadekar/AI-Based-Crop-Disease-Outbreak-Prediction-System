import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src import config

def run_lead_time_analysis():
    print("Running Lead Time Analysis...")
    
    results_path = os.path.join(config.OUTPUTS_DIR, "backtest_results.csv")
    df = pd.read_csv(results_path)
    df["date"] = pd.to_datetime(df["date"])
    
    events_path = "e:/crop-disease-outbreak-prediction-system-feature-zip-changes/crop-disease-outbreak-prediction-system-feature-zip-changes/model_train/research_comp/evidence_base/outbreak_events/red_rot_outbreak_events.csv"
    events_df = pd.read_csv(events_path)
    events_df["peak_start"] = pd.to_datetime(events_df["peak_start"])
    
    lead_times = []
    
    for idx, event in events_df.iterrows():
        peak_date = event["peak_start"]
        # Look for first alert in the preceding 14 days
        prior_alerts = df[(df["date"] >= peak_date - pd.Timedelta(days=14)) & 
                          (df["date"] < peak_date) & 
                          (df["alert"] == True)]
        
        if not prior_alerts.empty:
            first_alert_date = prior_alerts["date"].min()
            lead_time = (peak_date - first_alert_date).days
            
            # Categorize
            if 3 <= lead_time <= 7: status = "Valid (3-7d)"
            elif lead_time > 7: status = "Early (>7d)"
            else: status = "Late (<3d)"
            
            lead_times.append({
                "Event_ID": idx,
                "Peak_Date": peak_date.strftime("%Y-%m-%d"),
                "First_Alert": first_alert_date.strftime("%Y-%m-%d"),
                "Lead_Time": lead_time,
                "Status": status
            })
        else:
            lead_times.append({
                "Event_ID": idx,
                "Peak_Date": peak_date.strftime("%Y-%m-%d"),
                "First_Alert": "None",
                "Lead_Time": 0,
                "Status": "Missed"
            })
            
    lead_time_df = pd.DataFrame(lead_times)
    print("Lead Time Analysis Complete.")
    return lead_time_df

if __name__ == "__main__":
    print(run_lead_time_analysis())
