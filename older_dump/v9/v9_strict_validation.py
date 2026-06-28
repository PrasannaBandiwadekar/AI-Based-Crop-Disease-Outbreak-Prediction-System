import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Paths
BASE_DIR = os.getcwd()
FEATURES_PATH  = os.path.join(BASE_DIR, "v9", "data", "processed", "features.csv")
MODEL_PATH     = os.path.join(BASE_DIR, "v9", "models", "v9_tcn_corrected.pth")
GT_PATH        = os.path.join(BASE_DIR, "research_comp", "evidence_base", "outbreak_events", "sangli_synthetic_gt.csv")
OUTPUT_DIR     = os.path.join(BASE_DIR, "v9", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WEATHER_FEATURES = [
    'RH2M', 'T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR',
    'RH2M_mean_14', 'RH2M_mean_28', 'T2M_mean_14', 'T2M_mean_28',
    'humidity_streak', 'temp_streak', 'rainfall_streak', 'rainfall_sum_3',
    'T2M_MIN_lag_15', 'RH2M_lag_15', 'RH2M_diff_1', 'RH2M_accel'
]
AGRO_FEATURES = [
    'NDVI', 'NDVI_trend_7', 'variety_susceptibility',
    'ratoon_flag', 'sanitation_score'
]

# ─── Exact architecture used in v9_correction.py ───

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Sequential(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        L = x.size(2)
        out = self.conv(x)[:, :, :L]
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=128, kernel_size=3, dropout=0.2):
        super().__init__()
        self.blocks = nn.Sequential(
            TemporalBlock(input_dim,   hidden_dim,  kernel_size, dilation=1, dropout=dropout),
            TemporalBlock(hidden_dim,  hidden_dim,  kernel_size, dilation=2, dropout=dropout),
            TemporalBlock(hidden_dim,  output_dim,  kernel_size, dilation=4, dropout=dropout),
        )
        self.out_proj = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.blocks(x).transpose(1, 2)
        return self.out_proj(x)

class V9TCNModel(nn.Module):
    def __init__(self, weather_dim, agro_dim, gru_units=64):
        super().__init__()
        self.tcn = TCNEncoder(weather_dim, hidden_dim=gru_units, output_dim=gru_units * 2)
        self.attention_w = nn.Linear(gru_units * 2, 1)
        self.agronomic_mlp = nn.Sequential(
            nn.Linear(agro_dim, 16), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16, 8), nn.ReLU()
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(gru_units * 2 + 8, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, weather_seq, agronomic_state):
        tcn_out     = self.tcn(weather_seq)
        attn_scores = self.attention_w(tcn_out)
        attn_weights = F.softmax(attn_scores, dim=1)
        weather_ctx = torch.sum(attn_weights * tcn_out, dim=1)
        agro_ctx    = self.agronomic_mlp(agronomic_state)
        combined    = torch.cat([weather_ctx, agro_ctx], dim=1)
        logits      = self.fusion_layer(combined)
        return logits, attn_weights

# ─── Inference (Black-Box) ───

def run_inference(model, df, scaler_w, scaler_a, device, seq_len=14):
    model.eval()
    risk_scores = {}
    with torch.no_grad():
        for i in range(seq_len - 1, len(df)):
            date  = df.loc[i, 'date']
            w_t   = torch.FloatTensor(
                scaler_w.transform(df.iloc[i-seq_len+1:i+1][WEATHER_FEATURES])
            ).unsqueeze(0).to(device)
            a_t   = torch.FloatTensor(
                scaler_a.transform(df.iloc[i][AGRO_FEATURES].values.reshape(1, -1))
            ).to(device)
            logits, _ = model(w_t, a_t)
            risk_scores[date] = torch.sigmoid(logits).item() * 100
    return risk_scores

# ─── Strict Early Warning Evaluation ───

def strict_evaluate(risk_scores, gt_df, threshold):
    tp = fn = fp = tn = 0
    event_rows = []

    # Event-level: Detection window = [peak_start-7 → peak_start-1] STRICTLY PRE-ONSET
    for _, event in gt_df.iterrows():
        estart   = event['peak_start']
        dw_start = estart - timedelta(days=7)
        dw_end   = estart - timedelta(days=1)          # strict: before onset only

        detected       = False
        first_det_date = None
        max_score      = 0.0

        curr = dw_start
        while curr <= dw_end:
            score = risk_scores.get(curr, 0)
            if score > max_score:
                max_score = score
            if score >= threshold:
                detected = True
                if first_det_date is None:
                    first_det_date = curr
            curr += timedelta(days=1)

        if detected:
            tp += 1
            lead_time = (estart - first_det_date).days
        else:
            fn += 1
            lead_time = None

        event_rows.append({
            "event_date":      estart.strftime('%Y-%m-%d'),
            "detected":        "Yes" if detected else "No",
            "first_detection": first_det_date.strftime('%Y-%m-%d') if first_det_date else "N/A",
            "lead_time_days":  lead_time if lead_time is not None else "N/A",
            "max_score":       round(max_score, 2),
        })

    # Daily-level FP / TN
    # Valid negative: no outbreak in next 14 days AND no outbreak in previous 3 days
    for date, score in risk_scores.items():
        next_14  = any(
            (gt_df['peak_start'] >= date) &
            (gt_df['peak_start'] <= date + timedelta(days=14))
        )
        prev_3 = any(
            (gt_df['peak_start'] >= date - timedelta(days=3)) &
            (gt_df['peak_start'] < date)
        )
        if not next_14 and not prev_3:
            if score >= threshold:
                fp += 1
            else:
                tn += 1

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr    = fp / (fp + tn) if (fp + tn) > 0 else 0
    return recall, fpr, tp, fn, fp, tn, event_rows

# ─── Main ───

def main():
    print("=" * 58)
    print("  V9 TCN STRICT BLACK-BOX VALIDATION — SANGLI GT")
    print("=" * 58)

    # Load data
    df = pd.read_csv(FEATURES_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    gt_df = pd.read_csv(GT_PATH)
    gt_df['peak_start'] = pd.to_datetime(gt_df['peak_start'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Scalers (fit on features only — no model modification)
    scaler_w = StandardScaler().fit(df[WEATHER_FEATURES])
    scaler_a = StandardScaler().fit(df[AGRO_FEATURES])

    # Load black-box model — no modifications
    model = V9TCNModel(len(WEATHER_FEATURES), len(AGRO_FEATURES)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded: v9/models/v9_tcn_corrected.pth")

    # Inference
    print("Running inference (black-box, no modifications)...")
    risk_scores = run_inference(model, df, scaler_w, scaler_a, device)
    print(f"Inference complete: {len(risk_scores)} daily risk scores generated.")

    # Threshold sweep
    thresholds = [30, 40, 50, 60, 70]
    metric_rows = []
    best_event_rows = None
    print("\n10.1 Metrics Table (Strict Pre-Onset Detection Window)")
    print(f"{'Threshold':>10} | {'TP':>4} | {'FP':>6} | {'TN':>6} | {'FN':>4} | {'Recall':>7} | {'FPR':>7}")
    print("-" * 62)

    for thresh in thresholds:
        recall, fpr, tp, fn, fp, tn, event_rows = strict_evaluate(risk_scores, gt_df, thresh)
        metric_rows.append(dict(Threshold=thresh, TP=tp, FP=fp, TN=tn, FN=fn,
                                Recall=round(recall, 4), FPR=round(fpr, 4)))
        print(f"{thresh:>10} | {tp:>4} | {fp:>6} | {tn:>6} | {fn:>4} | {recall:>7.4f} | {fpr:>7.4f}")
        if best_event_rows is None:
            best_event_rows = (thresh, event_rows, recall, fpr)

    metrics_df = pd.DataFrame(metric_rows)

    # Best threshold: Recall >= 0.60, min FPR
    candidates = metrics_df[metrics_df['Recall'] >= 0.60]
    best_row = candidates.loc[candidates['FPR'].idxmin()] if not candidates.empty \
               else metrics_df.loc[metrics_df['Recall'].idxmax()]

    r, f, tp2, fn2, fp2, tn2, best_event_rows_final = strict_evaluate(
        risk_scores, gt_df, int(best_row['Threshold'])
    )

    event_df = pd.DataFrame(best_event_rows_final)

    # Lead Time Analysis
    lead_times = [r['lead_time_days'] for r in best_event_rows_final
                  if r['lead_time_days'] != 'N/A']
    avg_lead = round(np.mean(lead_times), 2) if lead_times else 0
    min_lead = min(lead_times) if lead_times else 0

    print(f"\n10.2 Event-Level Table (Threshold = {int(best_row['Threshold'])})")
    print(f"{'Event Date':<13} | {'Det':>3} | {'First Detection':<16} | {'Lead (days)':>10} | {'Max Score':>9}")
    print("-" * 62)
    for r in best_event_rows_final:
        print(f"{r['event_date']:<13} | {r['detected']:>3} | {r['first_detection']:<16} | "
              f"{str(r['lead_time_days']):>10} | {r['max_score']:>9.2f}")

    print(f"\n10.3 Summary")
    print(f"Best Threshold:   {int(best_row['Threshold'])}")
    print(f"Recall:           {best_row['Recall']}")
    print(f"FPR:              {best_row['FPR']}")
    print(f"Avg Lead Time:    {avg_lead} days")
    print(f"Min Lead Time:    {min_lead} days")
    print(f"Events Detected:  {tp2}/{tp2+fn2}")

    # Validation criteria
    recall_ok   = best_row['Recall'] >= 0.60
    fpr_ok      = best_row['FPR'] <= 0.15
    lead_ok     = avg_lead >= 2
    pre_onset_ok = all(r['lead_time_days'] != 'N/A' and r['lead_time_days'] >= 1
                       for r in best_event_rows_final if r['detected'] == 'Yes')

    status = "PASS" if (recall_ok and fpr_ok and lead_ok and pre_onset_ok) else "FAIL"

    print(f"\n{'='*58}")
    print(f"FINAL STATUS: {status}")
    print(f"  Recall >= 0.60 : {'PASS' if recall_ok else 'FAIL'} ({best_row['Recall']})")
    print(f"  FPR    <= 0.15 : {'PASS' if fpr_ok else 'FAIL'} ({best_row['FPR']})")
    print(f"  AvgLead >= 2d  : {'PASS' if lead_ok else 'FAIL'} ({avg_lead} days)")
    print(f"  Pre-Onset Only : {'PASS' if pre_onset_ok else 'FAIL'}")
    print(f"{'='*58}")

    # Save outputs
    report_path = os.path.join(OUTPUT_DIR, "sangli_strict_validation_report.txt")
    with open(report_path, "w") as f:
        f.write("V9 TCN STRICT BLACK-BOX VALIDATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write("10.1 Metrics Table (Strict Pre-Onset Window)\n")
        f.write(metrics_df.to_string(index=False) + "\n\n")
        f.write(f"10.2 Event-Level Table (Best Threshold: {int(best_row['Threshold'])})\n")
        f.write(event_df.to_string(index=False) + "\n\n")
        f.write("10.3 Summary\n")
        f.write(f"  Best Threshold : {int(best_row['Threshold'])}\n")
        f.write(f"  Recall         : {best_row['Recall']}\n")
        f.write(f"  FPR            : {best_row['FPR']}\n")
        f.write(f"  Avg Lead Time  : {avg_lead} days\n")
        f.write(f"  Min Lead Time  : {min_lead} days\n")
        f.write(f"  Events Detected: {tp2}/{tp2+fn2}\n\n")
        f.write(f"FINAL STATUS: {status}\n")

    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "sangli_strict_metrics.csv"), index=False)
    event_df.to_csv(os.path.join(OUTPUT_DIR, "sangli_strict_events.csv"), index=False)
    print(f"\nOutputs saved to v9/outputs/")

if __name__ == "__main__":
    main()
