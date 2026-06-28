import os
import torch
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from model_tcn import V10TCNModel
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "v10_tcn_model.pth")
GT_PATH = os.path.join(os.path.dirname(BASE_DIR), "research_comp", "evidence_base", "outbreak_events", "sangli_synthetic_gt.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WEATHER_FEATURES = ['RH_Contrast', 'T_Contrast', 'RH_Contrast_Accum_7', 'Trigger_Pulse', 'RH_Slope', 'RH2M', 'T2M']
AGRO_FEATURES = ['ratoon_flag', 'sanitation_score']

def load_threshold():
    try:
        with open(os.path.join(BASE_DIR, "models", "v10_metadata.txt"), "r") as f:
            for line in f:
                if line.startswith("optimal_threshold"):
                    return float(line.split("=")[1].strip())
    except:
        pass
    return 0.5

def infer_on_df(model, df_w, df_a, scaler_w, scaler_a, device, dates, seq_len=14):
    model.eval()
    risk_scores = {}
    w_scaled = scaler_w.transform(df_w)
    a_scaled = scaler_a.transform(df_a)
    with torch.no_grad():
        for i in range(seq_len - 1, len(dates)):
            date = dates[i]
            w_t = torch.FloatTensor(w_scaled[i-seq_len+1:i+1]).unsqueeze(0).to(device)
            a_t = torch.FloatTensor(a_scaled[i]).reshape(1, -1).to(device)
            logits, _ = model(w_t, a_t)
            risk_scores[date] = torch.sigmoid(logits).item()
    return risk_scores

def recall_at_threshold(risk_scores, gt_df, threshold):
    tp = fn = 0
    # In V10.5, positive window is t-10 to t-2
    for _, event in gt_df.iterrows():
        estart = event['peak_start']
        dw_start = estart - timedelta(days=7)
        dw_end   = estart - timedelta(days=3)
        detected = any(risk_scores.get(d, 0) >= threshold for d in pd.date_range(dw_start, dw_end))
        if detected: tp += 1
        else:        fn += 1
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def main():
    print("=" * 62)
    print("  V10 MULTI-SHIFT CAUSALITY AUDIT")
    print("=" * 62)

    df = pd.read_csv(FEATURES_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    # Train Scalers on 2005-2016
    train_mask = (df['date'].dt.year >= 2005) & (df['date'].dt.year <= 2016)
    scaler_w = StandardScaler().fit(df.loc[train_mask, WEATHER_FEATURES])
    scaler_a = StandardScaler().fit(df.loc[train_mask, AGRO_FEATURES])
    
    # Evaluate ONLY on Test Set (2019-2024)
    df_test = df[df['date'].dt.year >= 2019].reset_index(drop=True)
    gt_df = pd.read_csv(GT_PATH)
    gt_df['peak_start'] = pd.to_datetime(gt_df['peak_start'])
    gt_test = gt_df[gt_df['peak_start'].dt.year >= 2019].reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V10TCNModel(len(WEATHER_FEATURES), len(AGRO_FEATURES)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    threshold = load_threshold()

    dates_ts = [pd.Timestamp(d) for d in df_test['date'].values]
    
    print(f"\n[Task 0] Baseline Test Set Recall (Threshold={threshold:.4f})")
    baseline_scores = infer_on_df(model, df_test[WEATHER_FEATURES].values, df_test[AGRO_FEATURES].values, scaler_w, scaler_a, device, dates_ts)
    base_recall = recall_at_threshold(baseline_scores, gt_test, threshold)
    base_fpr = sum(1 for s in baseline_scores.values() if s >= threshold) / len(baseline_scores)
    
    print(f"  Baseline Recall: {base_recall:.4f}")
    print(f"  Baseline FPR:    {base_fpr:.4f}")

    if base_recall < 0.2:
        print("\n[GUARDRAIL] Model is not sufficiently predictive (Recall < 0.2).")
        print("  Shift test audit is INVALID as there's no signal to degrade.")
        return

    print("\n[Task 2] Multi-Shift Degradation Test")
    print("  Testing if forward-shifting data (simulating future info) degrades performance.")
    print("  Expected: Recall should drop progressively as temporal alignment breaks.")

    for shift_days in [1, 3, 5]:
        df_shifted = df_test.copy()
        # Shift weather forward (simulating late reporting/lost causality)
        df_shifted[WEATHER_FEATURES] = df_test[WEATHER_FEATURES].shift(shift_days).bfill()
        
        shifted_scores = infer_on_df(model, df_shifted[WEATHER_FEATURES].values, df_shifted[AGRO_FEATURES].values, scaler_w, scaler_a, device, dates_ts)
        s_recall = recall_at_threshold(shifted_scores, gt_test, threshold)
        delta = s_recall - base_recall
        
        status = "PASSED (Dropped)" if delta < 0 else "SUSPICIOUS (Maintained)"
        if delta < -0.2: status = "PASSED (Strong Drop)"
        
        print(f"  Shift +{shift_days}d | Recall: {s_recall:.4f} | Delta: {delta:+.4f} | {status}")

if __name__ == "__main__":
    main()
