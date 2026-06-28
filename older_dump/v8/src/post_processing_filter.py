import pandas as pd
import numpy as np
import os

def apply_filters():
    print("Executing V8 Stage-2 Post-Processing Filters...")
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS_PATH = os.path.join(BASE_DIR, "outputs", "v8_backtest_results.csv")
    FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "features.csv")
    OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "v8_filtered_results.csv")
    
    # 1. Load Data
    results_df = pd.read_csv(RESULTS_PATH)
    features_df = pd.read_csv(FEATURES_PATH)
    
    results_df["date"] = pd.to_datetime(results_df["date"])
    features_df["date"] = pd.to_datetime(features_df["date"])
    
    # Merge to get features needed for filtering
    df = pd.merge(results_df, features_df[["date", "RH2M", "PRECTOTCORR", "humidity_streak", "rainfall_spike", "variety_susceptibility"]], on="date")
    
    # 2. Daily Filters (Variety & Plateau)
    print("Applying Variety and Plateau Suppression filters...")
    
    # D. Variety Filter: If susceptibility == 0, force risk = 0
    df.loc[df["variety_susceptibility"] == 0, "alert"] = False
    
    # C. Plateau Suppression: If humidity_streak >= 10 AND no rainfall spike, discard day
    df.loc[(df["humidity_streak"] >= 10) & (df["rainfall_spike"] == False), "alert"] = False
    
    # 3. Event Formation (Clustering)
    print("Forming events for cluster-level filtering...")
    alert_days = df[df["alert"] == True].sort_values("date").copy()
    
    if alert_days.empty:
        print("No alerts surviving daily filters. Saving empty results.")
        df[["date", "alert"]].to_csv(OUTPUT_PATH, index=False)
        return
        
    events = []
    current_event = [alert_days.iloc[0].to_dict()]
    
    for i in range(1, len(alert_days)):
        prev_date = alert_days.iloc[i-1]["date"]
        curr_date = alert_days.iloc[i]["date"]
        
        if (curr_date - prev_date).days <= 3:
            current_event.append(alert_days.iloc[i].to_dict())
        else:
            events.append(current_event)
            current_event = [alert_days.iloc[i].to_dict()]
    events.append(current_event)
    
    # 4. Cluster-Level Filters (Persistence & Intensity)
    print(f"Initial event count: {len(events)}")
    
    filtered_events = []
    for event in events:
        event_df = pd.DataFrame(event)
        
        # A. Persistence Filter: duration >= 3 days
        duration = (event_df["date"].max() - event_df["date"].min()).days + 1
        if duration < 3:
            continue
            
        # Calculate Intensity for this event
        event_df["intensity_score"] = event_df["RH2M"] * event_df["PRECTOTCORR"]
        max_intensity = event_df["intensity_score"].max()
        
        filtered_events.append({
            "event": event,
            "max_intensity": max_intensity,
            "duration": duration
        })
        
    print(f"Events after Persistence filter: {len(filtered_events)}")
    
    # B. Intensity Filter: Keep top percentile
    if filtered_events:
        intensities = [e["max_intensity"] for e in filtered_events]
        threshold = np.percentile(intensities, 50)
        print(f"Intensity threshold (50th percentile): {threshold:.2f}")
        
        final_surviving_events = [e for e in filtered_events if e["max_intensity"] >= threshold]
    else:
        final_surviving_events = []
        
    print(f"Final surviving events: {len(final_surviving_events)}")
    
    # 5. Reconstruct Final Alert Column
    surviving_dates = set()
    for e in final_surviving_events:
        for day in e["event"]:
            surviving_dates.add(day["date"])
            
    df["alert"] = df["date"].isin(surviving_dates)
    
    # 6. Save
    df[["date", "alert"]].to_csv(OUTPUT_PATH, index=False)
    print(f"Filtered results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    apply_filters()
