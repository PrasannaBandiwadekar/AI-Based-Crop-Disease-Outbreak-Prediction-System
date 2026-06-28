import pandas as pd
import numpy as np
import json
import os

# Paths
INPUT_DATA = "v5/dataset/canonical_timeseries.csv"
OUTPUT_DIR = "v5/labels"
LABEL_FILE = os.path.join(OUTPUT_DIR, "label.csv")
EVENT_LOG = os.path.join(OUTPUT_DIR, "event_log.json")

def generate_v5_labels():
    print("--- Generating v5 Canonical Labels ---")
    df = pd.read_csv(INPUT_DATA)
    df['date'] = pd.to_datetime(df['date'])
    
    # 1. Define High-Risk Days (Biological Truth)
    # Using raw weather from the canonical dataset (un-lagged)
    # This is the "Truth" we are trying to predict from the past
    df['is_high_risk'] = (
        (df['RH2M'] > 80) & 
        (df['T2M'] >= 20) & (df['T2M'] <= 30) & 
        (df['PRECTOTCORR'] > 0.5)
    ).astype(int)
    
    # 2. Identify Outbreak Events (Min 3 consecutive high-risk days)
    df['group'] = (df['is_high_risk'] != df['is_high_risk'].shift()).cumsum()
    
    events = []
    actual_outbreak_days = np.zeros(len(df))
    
    for g, group_df in df.groupby('group'):
        if group_df['is_high_risk'].iloc[0] == 1 and len(group_df) >= 3:
            event_id = int(g)
            start_idx = group_df.index[0]
            end_idx = group_df.index[-1]
            
            events.append({
                "event_id": event_id,
                "start_date": str(df.loc[start_idx, 'date'].date()),
                "end_date": str(df.loc[end_idx, 'date'].date()),
                "duration": len(group_df),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx)
            })
            actual_outbreak_days[start_idx:end_idx+1] = 1
            
    df['is_in_outbreak'] = actual_outbreak_days.astype(int)
    
    # 3. Create Forecasting Target (target_5d)
    # A prediction at day t is correct if an outbreak starts in [t+3, t+7]
    # Primary lead time = 5 days.
    
    # We define target_5d[t] = 1 if ANY event starts in t + [3 to 7] days
    target_5d = np.zeros(len(df))
    event_start_indices = [e['start_idx'] for e in events]
    
    for t in range(len(df)):
        # Check if any event starts in the window [t+3, t+7]
        window_start = t + 3
        window_end = t + 7
        for start_idx in event_start_indices:
            if window_start <= start_idx <= window_end:
                target_5d[t] = 1
                break
                
    df['target_5d'] = target_5d.astype(int)
    
    # Save Labels
    label_df = df[['date', 'is_high_risk', 'is_in_outbreak', 'target_5d']]
    label_df.to_csv(LABEL_FILE, index=False)
    
    # Save Event Log
    with open(EVENT_LOG, 'w') as f:
        json.dump(events, f, indent=4)
        
    print(f"Labels saved to {LABEL_FILE}")
    print(f"Event log saved to {EVENT_LOG}. Total events found: {len(events)}")
    print(f"Target distribution: {df['target_5d'].value_counts().to_dict()}")

if __name__ == "__main__":
    generate_v5_labels()
