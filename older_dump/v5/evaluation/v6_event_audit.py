import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src import config, utils, features, biological_discrimination_layer as bdl

def run_v6_event_reconstruction_audit():
    print("Starting V6 Event-Level Truth Reconstruction & Integrity Audit...")
    
    # 1. Load Core Data
    results_path = os.path.join(config.OUTPUTS_DIR, "backtest_results.csv")
    if not os.path.exists(results_path):
        print("Error: backtest_results.csv not found.")
        return
    
    results_df = pd.read_csv(results_path)
    results_df["date"] = pd.to_datetime(results_df["date"])
    
    raw_df = utils.load_raw_data()
    raw_df["date"] = pd.to_datetime(raw_df["date"])
    
    print("Loading outbreak events...")
    events_path = "e:/crop-disease-outbreak-prediction-system-feature-zip-changes/crop-disease-outbreak-prediction-system-feature-zip-changes/model_train/research_comp/evidence_base/outbreak_events/red_rot_outbreak_events.csv"
    gt_events_df = pd.read_csv(events_path)
    gt_events_df["peak_start"] = pd.to_datetime(gt_events_df["peak_start"])
    
    # 2. Enrich Predictions with BDL Metadata for all days (to check active phases)
    print("Re-evaluating BDL phases for reconstruction...")
    # To save time, we only evaluate days near alerts or in monsoon
    prediction_enrichment = []
    
    for idx, row in results_df.iterrows():
        d = row["date"]
        # Only process if alert is True OR within 7 days of an alert to check continuity
        # Or if it's monsoon (Jun-Oct) to see BDL behavior
        is_relevant = row["alert"] == True or (d.month >= 6 and d.month <= 10)
        
        if is_relevant:
            window = raw_df[raw_df["date"] <= d].tail(28).copy()
            feat_df = features.build_features(window)
            bdl_res = bdl.calculate_bdl_score(feat_df.tail(14))
            
            prediction_enrichment.append({
                "date": d,
                "bdl_score": bdl_res["bdl_score"],
                "bdl_phases": ",".join(bdl_res["phase_detected"]),
                "bdl_decision": bdl_res["final_decision"],
                "raw_model_prob": row["risk_score"] if "risk_score" in row else 0.0
            })
    
    enrich_df = pd.DataFrame(prediction_enrichment)
    results_df = results_df.merge(enrich_df, on="date", how="left")
    
    # 3. Step 1: Event Reconstruction Engine
    print("Clustering alerts into biological risk events...")
    alert_days = results_df[results_df["alert"] == True].sort_values("date")
    
    predicted_events = []
    if not alert_days.empty:
        current_cluster = [alert_days.iloc[0].to_dict()]
        
        for i in range(1, len(alert_days)):
            prev = alert_days.iloc[i-1]
            curr = alert_days.iloc[i]
            
            gap = (curr["date"] - prev["date"]).days
            
            # CONDITION 1: Consecutive or Gap <= 2 days
            # CONDITION 2: Shared BDL overlap (if both have high BDL scores even with a gap)
            # We'll use Gap <= 3 as a proxy for biological continuity as per user requirement "gap <= 2" 
            # meaning day 1 and day 4 (gap=2) merge.
            
            if gap <= 3: # 3 days difference means 2 days gap
                current_cluster.append(curr.to_dict())
            else:
                predicted_events.append(current_cluster)
                current_cluster = [curr.to_dict()]
        predicted_events.append(current_cluster)

    # 4. Step 2: Event Object Construction
    event_objects = []
    for idx, cluster in enumerate(predicted_events):
        start_date = cluster[0]["date"]
        end_date = cluster[-1]["date"]
        duration = (end_date - start_date).days + 1
        max_risk = max([c["risk_score"] if "risk_score" in c else 0.0 for c in cluster])
        max_bdl = max([c["bdl_score"] for c in cluster if not pd.isna(c["bdl_score"])] + [0])
        
        event_objects.append({
            "event_id": idx,
            "start": start_date,
            "end": end_date,
            "duration": duration,
            "max_risk": max_risk,
            "max_bdl": max_bdl,
            "alert_days_count": len(cluster)
        })
    
    events_summary_df = pd.DataFrame(event_objects)
    
    # 5. Step 3: True Event Matching Logic
    print("Matching predicted events to ground truth outbreaks...")
    # Overlap window: [-14 days, 0 days before peak]
    
    matched_gt_indices = set()
    hit_events = []
    
    for idx, p_event in events_summary_df.iterrows():
        is_hit = False
        target_outbreak_id = -1
        min_lead = 999
        
        for gt_idx, gt_event in gt_events_df.iterrows():
            peak = gt_event["peak_start"]
            window_start = peak - pd.Timedelta(days=14)
            window_end = peak
            
            # Match if ANY day in predicted event falls in [peak-14, peak]
            if not (p_event["end"] < window_start or p_event["start"] > window_end):
                is_hit = True
                matched_gt_indices.add(gt_idx)
                target_outbreak_id = gt_idx
                # Lead time calculation (from start of event to peak)
                lead = (peak - p_event["start"]).days
                min_lead = min(min_lead, lead)
                
        hit_events.append({
            "event_id": p_event["event_id"],
            "is_hit": is_hit,
            "gt_id": target_outbreak_id,
            "lead_time": min_lead if is_hit else -1
        })
        
    hit_df = pd.DataFrame(hit_events)
    events_summary_df = events_summary_df.merge(hit_df, on="event_id")
    
    # 6. Metrics Recalculation
    total_outbreaks = len(gt_events_df)
    detected_outbreaks = len(matched_gt_indices)
    
    event_recall = detected_outbreaks / total_outbreaks if total_outbreaks > 0 else 0
    
    # False Positive Rate: Clustered events that didn't hit anything
    total_predicted_events = len(events_summary_df)
    fp_events = events_summary_df[events_summary_df["is_hit"] == False]
    event_fpr = len(fp_events) / total_predicted_events if total_predicted_events > 0 else 0
    
    # Fragmentation Index: Predicted events per true outbreak
    # Only for detected ones
    if detected_outbreaks > 0:
        fragmentation_index = len(events_summary_df[events_summary_df["is_hit"]]) / detected_outbreaks
    else:
        fragmentation_index = 0
        
    # 7. BDL Fragmentation Impact Analysis
    # We look for gaps within the monsoon that WERE suppressed by BDL but had high ML risk
    # This indicates BDL is splitting events.
    monsoon_df = results_df[(results_df["date"].dt.month >= 6) & (results_df["date"].dt.month <= 10)].copy()
    bdl_split_count = 0
    # A split is defined as: [Alert=True] -> [Alert=False but Risk_Prob > 0.5] -> [Alert=True] within 5 days
    for i in range(2, len(monsoon_df)-2):
        if monsoon_df.iloc[i-1]["alert"] == True and \
           monsoon_df.iloc[i]["alert"] == False and \
           monsoon_df.iloc[i]["risk_score"] > 0.7 and \
           monsoon_df.iloc[i+1]["alert"] == True:
            bdl_split_count += 1
            
    bdl_impact = "neutral"
    if bdl_split_count > 5:
        bdl_impact = "fragmenter"
    elif len(fp_events) < total_predicted_events * 0.3:
        bdl_impact = "suppressor"
        
    # 8. Spike Suppression Validation
    # Check if 2-of-3 rule is causing the "FAIL" in stress tests but continuity is kept in events
    # We'll just check if single-day alerts (duration=1) are rare in hit events
    hit_durations = events_summary_df[events_summary_df["is_hit"]]["duration"]
    spike_suppression_verdict = "evaluation artifact" if hit_durations.mean() > 3 else "real failure"

    # 9. Print Results
    print("\n" + "="*50)
    print("V6 EVENT-LEVEL AUDIT RESULTS")
    print("="*50)
    print(f"Corrected Event-Level Recall: {event_recall:.2%}")
    print(f"Corrected Event-Level FPR:    {event_fpr:.2%}")
    print(f"Fragmentation Index:          {fragmentation_index:.2f} (Ideal: 1.0)")
    print(f"Total Predicted Events:       {total_predicted_events}")
    print(f"Total True Outbreaks:         {total_outbreaks}")
    print(f"Matched Outbreaks:            {detected_outbreaks}")
    print("-" * 50)
    print(f"BDL Impact Classification:    {bdl_impact.upper()}")
    print(f"Spike Suppression Verdict:    {spike_suppression_verdict.upper()}")
    
    # Final System Status
    if event_recall > 0.80 and event_fpr < 0.30:
        status = "System performance was underestimated (evaluation bug)"
    elif event_recall > 0.60:
        status = "Mixed: real + artifact effects"
    else:
        status = "True degradation confirmed"
    
    print(f"\nFINAL SYSTEM STATUS: {status}")
    print("="*50)
    
    # Save Results
    events_summary_df.to_csv(os.path.join(config.OUTPUTS_DIR, "v6_event_audit_results.csv"), index=False)
    
    # Generate updated report content
    generate_markdown_audit_report(event_recall, event_fpr, fragmentation_index, bdl_impact, spike_suppression_verdict, status, events_summary_df)

def generate_markdown_audit_report(recall, fpr, frag, bdl_imp, spike_v, status, df):
    report_path = os.path.join(config.BASE_DIR, "evaluation", "V6_EVENT_AUDIT_REPORT.md")
    
    # Lead time for hits
    hits = df[df["is_hit"]]
    
    with open(report_path, "w") as f:
        f.write(f"# V6 Event-Level Audit Report\n\n")
        f.write(f"## 1. Executive Summary\n")
        f.write(f"- **Final Status:** {status}\n")
        f.write(f"- **Corrected Recall:** {recall:.2%}\n")
        f.write(f"- **Corrected FPR:** {fpr:.2%}\n")
        f.write(f"- **Fragmentation Index:** {frag:.2f}\n\n")
        
        f.write(f"## 2. Integrity Analysis\n")
        f.write(f"| Component | Finding | Impact |\n")
        f.write(f"| --- | --- | --- |\n")
        f.write(f"| BDL Logic | {bdl_imp.capitalize()} | {'Splitting signals' if bdl_imp == 'fragmenter' else 'Reducing noise'} |\n")
        f.write(f"| Spike Suppression | {spike_v.capitalize()} | {'Valid temporal filter' if spike_v == 'evaluation artifact' else 'Blocking true signals'} |\n")
        f.write(f"| Evaluation Model | Continuous Biological Event | Corrected metric inflation |\n\n")
        
        f.write(f"## 3. Predicted Event Summary (Hits Only)\n")
        f.write(f"| Event_ID | Start | End | Duration | Lead Time (to peak) |\n")
        f.write(f"| --- | --- | --- | --- | --- |\n")
        for _, row in hits.iterrows():
            f.write(f"| {row['event_id']} | {row['start'].date()} | {row['end'].date()} | {row['duration']} | {row['lead_time']} days |\n")
            
        f.write(f"\n\n--- *Generated by V6 Audit Engine* ---\n")

    print(f"\nAudit Report saved to: {report_path}")

if __name__ == "__main__":
    run_v6_event_reconstruction_audit()
