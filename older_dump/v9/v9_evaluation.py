import os
import torch
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import model from existing source
import sys
sys.path.append(os.path.join(os.getcwd(), "v9", "src"))
from model import V9FusionModel

# Paths
BASE_DIR = os.getcwd()
FEATURES_PATH = os.path.join(BASE_DIR, "v9", "data", "processed", "features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "v9", "models", "v9_fusion_model.pth")
GT_PATH = os.path.join(BASE_DIR, "research_comp", "evidence_base", "outbreak_events", "sangli_synthetic_gt.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "v9", "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

WEATHER_FEATURES = [
    'RH2M', 'T2M', 'T2M_MAX', 'T2M_MIN', 
    'RH2M_mean_7', 'RH2M_mean_28', 'T2M_mean_28',
    'rainfall_sum_7', 'rainfall_sum_28',
    'T2M_MIN_lag_15', 'RH2M_lag_15',
    'RH2M_diff_1', 'RH2M_accel'
]
AGRO_FEATURES = [
    'NDVI', 'NDVI_trend_7', 'variety_susceptibility', 
    'ratoon_flag', 'sanitation_score'
]

def run_evaluation():
    print("Starting V9 System Evaluation - Sangli Ground Truth")
    
    # 1. Load Data
    df = pd.read_csv(FEATURES_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    gt_df = pd.read_csv(GT_PATH)
    gt_df['peak_start'] = pd.to_datetime(gt_df['peak_start'])
    
    # 2. Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V9FusionModel(len(WEATHER_FEATURES), len(AGRO_FEATURES)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    
    # Scalers
    scaler_w = StandardScaler().fit(df[WEATHER_FEATURES])
    scaler_a = StandardScaler().fit(df[AGRO_FEATURES])
    
    # 3. Perform Inference
    print("Performing inference...")
    seq_len = 14
    df = df.sort_values('date').reset_index(drop=True)
    risk_scores = {}
    
    with torch.no_grad():
        for i in range(seq_len - 1, len(df)):
            date = df.loc[i, 'date']
            weather_slice = df.iloc[i-seq_len+1:i+1][WEATHER_FEATURES]
            agro_row = df.iloc[i][AGRO_FEATURES]
            weather_t = torch.FloatTensor(scaler_w.transform(weather_slice)).unsqueeze(0).to(device)
            agro_t = torch.FloatTensor(scaler_a.transform(agro_row.values.reshape(1, -1))).to(device)
            logits, _ = model(weather_t, agro_t)
            prob = torch.sigmoid(logits).item()
            risk_scores[date] = prob * 100

    # 4. Threshold Sweep
    thresholds = [40, 50, 60, 70]
    eval_results = []
    
    def has_outbreak_in_next_14_days(current_date):
        future_window = [current_date, current_date + timedelta(days=14)]
        mask = (gt_df['peak_start'] >= future_window[0]) & (gt_df['peak_start'] <= future_window[1])
        return mask.any()

    print("Running threshold sweep...")
    for thresh in thresholds:
        tp, fn, fp, tn = 0, 0, 0, 0
        preds = {d: (1 if s >= thresh else 0) for d, s in risk_scores.items()}
        
        event_summaries = []
        for _, event in gt_df.iterrows():
            estart = event['peak_start']
            dw_start, dw_end = estart - timedelta(days=7), estart + timedelta(days=3)
            detected, first_det_date, max_score = False, None, 0
            
            curr = dw_start
            while curr <= dw_end:
                if curr in risk_scores:
                    max_score = max(max_score, risk_scores[curr])
                    if preds.get(curr, 0) == 1:
                        detected = True
                        if first_det_date is None: first_det_date = curr
                curr += timedelta(days=1)
            
            if detected: tp += 1
            else: fn += 1
            
            event_summaries.append({
                "event_start_date": estart.strftime('%Y-%m-%d'),
                "detected": "Yes" if detected else "No",
                "first_detection_date": first_det_date.strftime('%Y-%m-%d') if first_det_date else "N/A",
                "max_risk_score": round(max_score, 2)
            })
            
        for date, score in risk_scores.items():
            if not has_outbreak_in_next_14_days(date):
                if score >= thresh: fp += 1
                else: tn += 1
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        eval_results.append({
            "Threshold": thresh, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "Recall": round(recall, 4), "FPR": round(fpr, 4),
            "event_summaries": event_summaries
        })

    # 5. Output Generation and Storage
    metrics_df = pd.DataFrame(eval_results).drop(columns=['event_summaries'])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "sangli_metrics.csv"), index=False)
    
    candidates = metrics_df[metrics_df['Recall'] >= 0.80]
    best_row = candidates.loc[candidates['FPR'].idxmin()] if not candidates.empty else metrics_df.loc[metrics_df['Recall'].idxmax()]
    
    best_thresh_idx = metrics_df[metrics_df['Threshold'] == best_row['Threshold']].index[0]
    event_summary_df = pd.DataFrame(eval_results[best_thresh_idx]['event_summaries'])
    event_summary_df.to_csv(os.path.join(OUTPUT_DIR, "sangli_event_summary.csv"), index=False)
    
    recall_val, fpr_val = best_row['Recall'], best_row['FPR']
    classification = "NOT READY"
    if recall_val >= 0.80:
        classification = "DEPLOYABLE" if fpr_val <= 0.20 else "NEEDS OPTIMIZATION"
    elif recall_val >= 0.60:
        classification = "NEEDS OPTIMIZATION"

    # Save Text Report
    report_path = os.path.join(OUTPUT_DIR, "sangli_evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("V9 SYSTEM EVALUATION REPORT - SANGLI GROUND TRUTH\n")
        f.write("="*50 + "\n\n")
        f.write("10.1 Metrics Table\n")
        f.write(metrics_df.to_string(index=False) + "\n\n")
        f.write("10.2 Best Operating Threshold\n")
        f.write(f"Optimal Threshold: {best_row['Threshold']}\n")
        f.write(f"Recall: {best_row['Recall']}\n")
        f.write(f"FPR: {best_row['FPR']}\n\n")
        f.write("10.3 Event-Level Summary (at Optimal Threshold)\n")
        f.write(event_summary_df.to_string(index=False) + "\n\n")
        f.write("11. Final System Classification\n")
        f.write(f"Status: {classification}\n")
        f.write(f"Reason: Recall={recall_val}, FPR={fpr_val}\n")

    print(f"Evaluation complete. Results stored in {OUTPUT_DIR}")
    print(f"Report: {report_path}")

if __name__ == "__main__":
    run_evaluation()
