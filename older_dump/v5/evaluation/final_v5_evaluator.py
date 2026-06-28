import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src import config, utils, biological_discrimination_layer as bdl

def run_final_evaluation_lock():
    print("Executing V5 Red Rot Final Evaluation Lock...")
    
    # 1. Load Data
    results_path = os.path.join(config.OUTPUTS_DIR, "backtest_results.csv")
    if not os.path.exists(results_path):
        print("Error: backtest_results.csv not found.")
        return
    
    results_df = pd.read_csv(results_path)
    results_df["date"] = pd.to_datetime(results_df["date"])
    
    # Load Outbreak Events
    events_path = "e:/crop-disease-outbreak-prediction-system-feature-zip-changes/crop-disease-outbreak-prediction-system-feature-zip-changes/model_train/research_comp/evidence_base/outbreak_events/red_rot_outbreak_events.csv"
    gt_df = pd.read_csv(events_path)
    gt_df["peak_start"] = pd.to_datetime(gt_df["peak_start"])
    
    # 2. Prediction Event Formation (Clustering)
    print("Forming predicted events (Gap <= 2 days)...")
    alert_days = results_df[results_df["alert"] == True].sort_values("date").copy()
    
    predicted_events = []
    if not alert_days.empty:
        current_event = [alert_days.iloc[0].to_dict()]
        
        for i in range(1, len(alert_days)):
            prev_date = alert_days.iloc[i-1]["date"]
            curr_date = alert_days.iloc[i]["date"]
            
            # Gap <= 2 days means difference <= 3 days
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
    # Predicted event is HIT if first_alert in [peak_start - 7, peak_start]
    print("Matching predicted events to outbreaks (Strict Onset Rule)...")
    
    hit_map = []
    matched_gt_indices = set()
    
    for idx, pe in pe_df.iterrows():
        is_tp = False
        outbreak_id = -1
        
        for gt_idx, gt in gt_df.iterrows():
            peak = gt["peak_start"]
            win_start = peak - pd.Timedelta(days=7)
            win_end = peak
            
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
    pe_df = pe_df.merge(hit_df, on="event_id")
    
    # 4. Final Metrics
    total_outbreaks = len(gt_df)
    total_predicted = len(pe_df)
    tp_count = len(matched_gt_indices)
    fp_events = pe_df[pe_df["is_tp"] == False]
    
    recall = tp_count / total_outbreaks if total_outbreaks > 0 else 0
    fpr = len(fp_events) / total_predicted if total_predicted > 0 else 0
    avg_lead = pe_df[pe_df["is_tp"]]["lead_time"].mean() if tp_count > 0 else 0
    fi = total_predicted / total_outbreaks if total_outbreaks > 0 else 0
    
    # 5. Biological Tagging (Reporting ONLY)
    # We load raw data to get features for BDL
    print("Adding biological metadata (Reporting only)...")
    raw_df = utils.load_raw_data()
    raw_df["date"] = pd.to_datetime(raw_df["date"])
    
    from src import features # Import features to build on the fly
    
    def get_bdl_tag(d):
        window = raw_df[raw_df["date"] <= d].tail(28).copy()
        feat_df = features.build_features(window)
        res = bdl.calculate_bdl_score(feat_df.tail(14))
        return "PLAUSIBLE" if res["final_decision"] == "ALLOW" else "NOISE"

    pe_df["bio_tag"] = pe_df["first_alert"].apply(get_bdl_tag)
    
    # 6. Generate Report
    print("\n--- FINAL EVALUATION REPORT ---")
    print(f"Recall: {recall:.2%}")
    print(f"FPR (Event-level): {fpr:.2%}")
    print(f"Avg Lead Time: {avg_lead:.1f} days")
    print(f"Fragmentation Index: {fi:.2f}")
    
    verdict = "GO" if recall >= 0.80 and 0.8 <= fi <= 1.2 else "NO GO"
    print(f"\nFINAL VERDICT: {verdict}")
    
    # 7. Save Artifacts
    pe_df.to_csv(os.path.join(config.OUTPUTS_DIR, "final_prediction_events.csv"), index=False)
    
    # Matching Table
    match_table = pe_df[pe_df["is_tp"]].copy()
    match_table["peak_start"] = match_table["matched_outbreak_idx"].apply(lambda x: gt_df.iloc[x]["peak_start"])
    match_table = match_table[["event_id", "first_alert", "peak_start", "lead_time", "bio_tag"]]
    
    # Missed Outbreaks
    missed_outbreaks = gt_df[~gt_df.index.isin(matched_gt_indices)].copy()
    
    # Write Markdown Report
    report_path = os.path.join(config.BASE_DIR, "evaluation", "FINAL_LOCKED_EVALUATION_REPORT.md")
    with open(report_path, "w") as f:
        f.write("# V5 Red Rot Final Evaluation Report (LOCKED)\n\n")
        f.write(f"## 1. Executive Summary\n")
        f.write(f"- **Final Verdict:** {verdict}\n")
        f.write(f"- **Event Recall:** {recall:.2%}\n")
        f.write(f"- **Event FPR:** {fpr:.2%}\n")
        f.write(f"- **Avg Lead Time:** {avg_lead:.1f} days\n")
        f.write(f"- **Fragmentation Index:** {fi:.2f}\n\n")
        
        f.write(f"## 2. Event-Level Matching Table\n")
        f.write(match_table.to_string(index=False) + "\n\n")
        
        f.write(f"## 3. Missed Outbreaks\n")
        f.write(missed_outbreaks[["peak_start", "region"]].to_string(index=False) + "\n\n")
        
        f.write(f"## 4. Predicted Events (All)\n")
        f.write(pe_df[["event_id", "first_alert", "last_alert", "is_tp", "bio_tag"]].to_string(index=False) + "\n\n")
        
        f.write("---\n*Generated by Frozen Evaluation Engine*")

    print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    run_final_evaluation_lock()
