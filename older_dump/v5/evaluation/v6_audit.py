import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src import config, utils, features, biological_discrimination_layer as bdl

def run_v6_audit():
    print("Starting V6 False Positive Audit (BDL-Aware)...")
    
    # 1. Load Data
    results_path = os.path.join(config.OUTPUTS_DIR, "backtest_results.csv")
    if not os.path.exists(results_path):
        print("Error: backtest_results.csv not found. Run backtest_engine.py first.")
        return
    
    results_df = pd.read_csv(results_path)
    results_df["date"] = pd.to_datetime(results_df["date"])
    
    raw_df = utils.load_raw_data()
    raw_df["date"] = pd.to_datetime(raw_df["date"])
    
    events_path = "e:/crop-disease-outbreak-prediction-system-feature-zip-changes/crop-disease-outbreak-prediction-system-feature-zip-changes/model_train/research_comp/evidence_base/outbreak_events/red_rot_outbreak_events.csv"
    events_df = pd.read_csv(events_path)
    events_df["peak_start"] = pd.to_datetime(events_df["peak_start"])
    
    # 2. Re-run BDL logic for alerted days to get rich metadata
    print("Re-evaluating alerted days for biological validity...")
    alerted_days = results_df[results_df["alert"] == True].copy()
    
    audit_data = []
    for idx, row in alerted_days.iterrows():
        # Get 14-day window for BDL
        d = row["date"]
        window = raw_df[raw_df["date"] <= d].tail(28).copy() # Extra buffer for features
        feat_df = features.build_features(window)
        bdl_res = bdl.calculate_bdl_score(feat_df.tail(14))
        
        # Check proximity to outbreak
        is_near_outbreak = False
        days_to_peak = 999
        for _, event in events_df.iterrows():
            diff = (event["peak_start"] - d).days
            if 0 <= diff <= 14:
                is_near_outbreak = True
                days_to_peak = min(days_to_peak, diff)
        
        audit_data.append({
            "date": d,
            "bdl_score": bdl_res["bdl_score"],
            "bdl_phases": bdl_res["phase_detected"],
            "is_near_outbreak": is_near_outbreak,
            "days_to_peak": days_to_peak,
            "bdl_allow": bdl_res["final_decision"] == "ALLOW"
        })
        
    audit_df = pd.DataFrame(audit_data)
    
    # 3. Alert Event Clustering (Deduplication)
    print("Clustering alerts into risk events...")
    alert_events = []
    if not audit_df.empty:
        audit_df = audit_df.sort_values("date")
        current_event = [audit_df.iloc[0].to_dict()]
        
        for i in range(1, len(audit_df)):
            prev_date = audit_df.iloc[i-1]["date"]
            curr_date = audit_df.iloc[i]["date"]
            
            # If within 3 days, consider part of the same "risk event"
            if (curr_date - prev_date).days <= 3:
                current_event.append(audit_df.iloc[i].to_dict())
            else:
                alert_events.append(current_event)
                current_event = [audit_df.iloc[i].to_dict()]
        alert_events.append(current_event)
    
    # 4. Classify Alert Events
    event_summary = []
    for idx, event_days in enumerate(alert_events):
        start_date = event_days[0]["date"]
        end_date = event_days[-1]["date"]
        max_bdl = max([d["bdl_score"] for d in event_days])
        near_outbreak = any([d["is_near_outbreak"] for d in event_days])
        min_days_to_peak = min([d["days_to_peak"] for d in event_days])
        
        # Classification
        # 1. True Hit: leads to outbreak in 14 days
        # 2. Biological Precursor: high BDL but no outbreak followed
        # 3. Noise: Low BDL and no outbreak
        
        if near_outbreak:
            cat = "TRUE_HIT"
        elif max_bdl >= 0.70:
            cat = "BIO_PRECURSOR_ONLY"
        else:
            cat = "NOISE"
            
        event_summary.append({
            "event_id": idx,
            "start": start_date,
            "end": end_date,
            "duration": (end_date - start_date).days + 1,
            "max_bdl": max_bdl,
            "near_outbreak": near_outbreak,
            "min_lead": min_days_to_peak,
            "category": cat,
            "month": start_date.month
        })
        
    summary_df = pd.DataFrame(event_summary)
    
    # 5. Metrics Calculation
    total_alert_days = len(alerted_days)
    total_events = len(summary_df)
    
    # Strict FPR: Raw alerts not leading to outbreak
    strict_fp_days = alerted_days[~alerted_days["date"].isin(audit_df[audit_df["is_near_outbreak"]]["date"])]
    strict_fpr = len(strict_fp_days) / len(results_df[results_df["alert"] == True]) if len(alerted_days) > 0 else 0
    
    # Operational FPR: Events that are NOISE
    operational_fps = summary_df[summary_df["category"] == "NOISE"]
    operational_fpr = len(operational_fps) / total_events if total_events > 0 else 0
    
    # Biological FPR: Events that have category NOISE (since BIO_PRECURSOR_ONLY are valid signals)
    # Actually user: "FP ONLY IF: No outbreak AND no valid biological precursor"
    # So category == NOISE is the definition of FP.
    
    # 6. Seasonal Breakdown
    summary_df["season"] = summary_df["month"].apply(lambda x: "Monsoon" if 6 <= x <= 10 else ("Pre-Monsoon" if 3 <= x <= 5 else "Post-Monsoon"))
    seasonal = summary_df.groupby("season")["category"].value_counts(normalize=True).unstack(fill_value=0)
    
    # 7. Print Results
    print("\n--- V6 AUDIT REPORT ---")
    print(f"Total Alert Days: {total_alert_days}")
    print(f"Total Risk Events (Clustered): {total_events}")
    print(f"Strict FPR (Naive): {strict_fpr:.2%}")
    print(f"Operational/Biological FPR: {operational_fpr:.2%}")
    print("\nEvent Categories:")
    print(summary_df["category"].value_counts())
    print("\nSeasonal Breakdown (Operational):")
    print(seasonal)
    
    # Final Verdict
    if operational_fpr < 0.20:
        verdict = "TRUE FP REDUCTION (system improved)"
    elif operational_fpr < 0.40:
        verdict = "FP REDUCTION WITH EARLY ALERT TRADEOFF (acceptable)"
    else:
        verdict = "OVER-SUPPRESSION RISK (system too strict)"
        
    print(f"\nFINAL VERDICT: {verdict}")
    
    # Save audit report
    summary_df.to_csv(os.path.join(config.OUTPUTS_DIR, "v6_audit_summary.csv"), index=False)

if __name__ == "__main__":
    run_v6_audit()
