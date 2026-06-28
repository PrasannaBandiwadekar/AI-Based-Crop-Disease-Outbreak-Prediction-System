import pandas as pd
import numpy as np
import os

def run_v8_evaluation(results_path=None):
    print("Executing V8 Red Rot Final Evaluation...")
    
    # 1. Load Data
    if results_path is None:
        results_path = "e:/crop-disease-outbreak-prediction-system-feature-zip-changes/crop-disease-outbreak-prediction-system-feature-zip-changes/model_train/v8/outputs/v8_backtest_results.csv"
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return
    
    results_df = pd.read_csv(results_path)
    results_df["date"] = pd.to_datetime(results_df["date"])
    
    # Load Outbreak Events (Synthetic Sangli GT)
    events_path = "e:/crop-disease-outbreak-prediction-system-feature-zip-changes/crop-disease-outbreak-prediction-system-feature-zip-changes/model_train/research_comp/evidence_base/outbreak_events/sangli_synthetic_gt.csv"
    gt_df = pd.read_csv(events_path)
    gt_df["peak_start"] = pd.to_datetime(gt_df["peak_start"])
    
    # 2. Prediction Event Formation (Clustering)
    print("Forming predicted events (Gap <= 3 days)...")
    alert_days = results_df[results_df["alert"] == True].sort_values("date").copy()
    
    predicted_events = []
    if not alert_days.empty:
        current_event = [alert_days.iloc[0].to_dict()]
        
        for i in range(1, len(alert_days)):
            prev_date = alert_days.iloc[i-1]["date"]
            curr_date = alert_days.iloc[i]["date"]
            
            if (curr_date - prev_date).days <= 3:
                current_event.append(alert_days.iloc[i].to_dict())
            else:
                predicted_events.append(current_event)
                current_event = [alert_days.iloc[i].to_dict()]
        predicted_events.append(current_event)
        
    event_list = []
    for idx, e in enumerate(predicted_events):
        event_list.append({
            "event_id": idx,
            "first_alert": e[0]["date"],
            "last_alert": e[-1]["date"],
            "duration": (e[-1]["date"] - e[0]["date"]).days + 1,
            "alert_count": len(e)
        })
    
    pe_df = pd.DataFrame(event_list)
    
    # 3. Matching Rule (Strict)
    print("Matching predicted events to outbreaks (Strict Onset Rule)...")
    
    hit_map = []
    matched_gt_indices = set()
    
    if not pe_df.empty:
        for idx, pe in pe_df.iterrows():
            is_tp = False
            outbreak_id = -1
            
            for gt_idx, gt in gt_df.iterrows():
                peak = gt["peak_start"]
                win_start = peak - pd.Timedelta(days=10)
                win_end = peak - pd.Timedelta(days=2)
                
                if win_start <= pe["first_alert"] <= win_end:
                    is_tp = True
                    outbreak_id = gt_idx
                    matched_gt_indices.add(gt_idx)
                    break
                    
            hit_map.append({
                "event_id": pe["event_id"],
                "is_tp": is_tp,
                "matched_outbreak_idx": outbreak_id,
                "lead_time": (gt_df.iloc[outbreak_id]["peak_start"] - pe["first_alert"]).days if is_tp else -1
            })
    
    hit_df = pd.DataFrame(hit_map)
    if not pe_df.empty:
        pe_df = pe_df.merge(hit_df, on="event_id")
    
    # 4. Final Metrics
    total_outbreaks = len(gt_df)
    total_predicted = len(pe_df)
    tp_count = len(matched_gt_indices)
    
    recall = tp_count / total_outbreaks if total_outbreaks > 0 else 0
    fpr = (len(pe_df[pe_df["is_tp"] == False]) / total_predicted) if total_predicted > 0 else 0
    avg_lead = pe_df[pe_df["is_tp"]]["lead_time"].mean() if tp_count > 0 else 0
    
    print("\n" + "="*40)
    print("V8 FINAL EVALUATION RESULTS")
    print("="*40)
    print(f"Total Confirmed Outbreaks:  {total_outbreaks}")
    print(f"Total Predicted Events:    {total_predicted}")
    print(f"True Positives (Hits):      {tp_count}")
    print(f"Recall:                    {recall:.2%}")
    print(f"False Positive Rate (FPR): {fpr:.2%}")
    print(f"Avg Lead Time (Days):      {avg_lead:.1f}")
    print("="*40)

if __name__ == "__main__":
    run_v8_evaluation()
