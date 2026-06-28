"""
V9 STRICT DATA LEAKAGE DETECTION AUDIT
- No model modifications, no feature modifications
- Black-box inference only
"""
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
BASE_DIR      = os.getcwd()
FEATURES_PATH = os.path.join(BASE_DIR, "v9", "data", "processed", "features.csv")
MODEL_PATH    = os.path.join(BASE_DIR, "v9", "models", "v9_tcn_corrected.pth")
GT_PATH       = os.path.join(BASE_DIR, "research_comp", "evidence_base", "outbreak_events", "sangli_synthetic_gt.csv")
OUTPUT_DIR    = os.path.join(BASE_DIR, "v9", "outputs")
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

# Feature metadata for Task 1 (static analysis)
FEATURE_METADATA = {
    'RH2M':          {'type': 'instantaneous', 'window': None},
    'T2M':           {'type': 'instantaneous', 'window': None},
    'T2M_MAX':       {'type': 'instantaneous', 'window': None},
    'T2M_MIN':       {'type': 'instantaneous', 'window': None},
    'RH2M_mean_7':   {'type': 'rolling_window', 'window': 7},
    'RH2M_mean_28':  {'type': 'rolling_window', 'window': 28},
    'T2M_mean_28':   {'type': 'rolling_window', 'window': 28},
    'rainfall_sum_7': {'type': 'rolling_window', 'window': 7},
    'rainfall_sum_28':{'type': 'rolling_window', 'window': 28},
    'T2M_MIN_lag_15': {'type': 'lag_based',     'window': 15},
    'RH2M_lag_15':    {'type': 'lag_based',     'window': 15},
    'RH2M_diff_1':    {'type': 'derived',       'window': 1},
    'RH2M_accel':     {'type': 'derived',       'window': 2},
    'NDVI':           {'type': 'instantaneous', 'window': None},
    'NDVI_trend_7':   {'type': 'rolling_window', 'window': 7},
    'variety_susceptibility': {'type': 'static', 'window': None},
    'ratoon_flag':    {'type': 'static',         'window': None},
    'sanitation_score': {'type': 'static',       'window': None},
}

# ─── Model (exact architecture from v9_correction.py) ───

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
            TemporalBlock(input_dim,  hidden_dim,  kernel_size, dilation=1, dropout=dropout),
            TemporalBlock(hidden_dim, hidden_dim,  kernel_size, dilation=2, dropout=dropout),
            TemporalBlock(hidden_dim, output_dim,  kernel_size, dilation=4, dropout=dropout),
        )
        self.out_proj = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.out_proj(self.blocks(x).transpose(1, 2))

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
        tcn_out = self.tcn(weather_seq)
        attn_weights = F.softmax(self.attention_w(tcn_out), dim=1)
        weather_ctx = torch.sum(attn_weights * tcn_out, dim=1)
        agro_ctx = self.agronomic_mlp(agronomic_state)
        return self.fusion_layer(torch.cat([weather_ctx, agro_ctx], dim=1)), attn_weights

# ─── Inference helpers ───

def infer_on_df(model, df_w, df_a, scaler_w, scaler_a, device, dates, seq_len=14):
    """Run inference using provided weather (df_w) and agro (df_a) arrays."""
    model.eval()
    risk_scores = {}
    w_scaled = scaler_w.transform(df_w)
    a_scaled = scaler_a.transform(df_a)
    with torch.no_grad():
        for i in range(seq_len - 1, len(dates)):
            date = dates[i]
            w_t = torch.FloatTensor(w_scaled[i-seq_len+1:i+1]).unsqueeze(0).to(device)
            a_t = torch.FloatTensor(a_scaled[i]).reshape(1, -1)
            a_t = torch.FloatTensor(a_scaled[i:i+1]).to(device)
            logits, _ = model(w_t, a_t)
            risk_scores[date] = torch.sigmoid(logits).item() * 100
    return risk_scores

def recall_at_threshold(risk_scores, gt_df, threshold=60.0, strict=True):
    """
    strict=True: detection window = [peak_start-7, peak_start-1]
    strict=False: detection window = [peak_start-7, peak_start+3]
    """
    tp = fn = 0
    for _, event in gt_df.iterrows():
        estart = event['peak_start']
        dw_start = estart - timedelta(days=7)
        dw_end   = estart - timedelta(days=1) if strict else estart + timedelta(days=3)
        detected = any(risk_scores.get(d, 0) >= threshold
                       for d in pd.date_range(dw_start, dw_end))
        if detected: tp += 1
        else:        fn += 1
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def fpr_at_threshold(risk_scores, gt_df, threshold=60.0):
    fp = tn = 0
    for date, score in risk_scores.items():
        no_next14 = not ((gt_df['peak_start'] >= date) &
                         (gt_df['peak_start'] <= date + timedelta(days=14))).any()
        no_prev3  = not ((gt_df['peak_start'] >= date - timedelta(days=3)) &
                         (gt_df['peak_start'] < date)).any()
        if no_next14 and no_prev3:
            if score >= threshold: fp += 1
            else:                  tn += 1
    return fp / (fp + tn) if (fp + tn) > 0 else 0

# ─── Main audit ───

def main():
    print("=" * 62)
    print("  V9 TCN DATA LEAKAGE DETECTION AUDIT")
    print("=" * 62)

    df = pd.read_csv(FEATURES_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    gt_df = pd.read_csv(GT_PATH)
    gt_df['peak_start'] = pd.to_datetime(gt_df['peak_start'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler_w = StandardScaler().fit(df[WEATHER_FEATURES])
    scaler_a = StandardScaler().fit(df[AGRO_FEATURES])

    model = V9TCNModel(len(WEATHER_FEATURES), len(AGRO_FEATURES)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    dates_arr = df['date'].values
    dates_ts  = [pd.Timestamp(d) for d in dates_arr]

    # Baseline
    print("\n[Task 0] Baseline recall (threshold=60, strict pre-onset window)...")
    base_scores = infer_on_df(model, df[WEATHER_FEATURES].values,
                              df[AGRO_FEATURES].values,
                              scaler_w, scaler_a, device, dates_ts)
    base_recall = recall_at_threshold(base_scores, gt_df, threshold=60)
    base_fpr    = fpr_at_threshold(base_scores, gt_df, threshold=60)
    print(f"  Baseline: Recall={base_recall:.4f}, FPR={base_fpr:.4f}")

    # ─── Task 1: Feature Temporal Dependency Audit ───
    print("\n[Task 1] Feature Temporal Dependency Audit (Static)")
    task1_rows = []
    for feat in WEATHER_FEATURES + AGRO_FEATURES:
        meta = FEATURE_METADATA.get(feat, {'type': 'unknown', 'window': None})
        # Leakage risk assessment
        if meta['type'] == 'static':
            leakage_risk = 'LOW'
        elif meta['type'] == 'lag_based':
            leakage_risk = 'LOW'  # Lags reference PAST data by definition
        elif meta['type'] == 'instantaneous':
            leakage_risk = 'LOW'
        elif meta['type'] == 'rolling_window':
            # Rolling windows that are longer than the prediction horizon are suspicious
            leakage_risk = 'MEDIUM' if meta['window'] and meta['window'] <= 7 else 'MEDIUM-HIGH'
        elif meta['type'] == 'derived':
            leakage_risk = 'LOW'
        else:
            leakage_risk = 'UNKNOWN'
        task1_rows.append({'feature': feat, 'type': meta['type'],
                           'window_days': meta['window'], 'leakage_risk': leakage_risk})

    task1_df = pd.DataFrame(task1_rows)
    print(f"  {'Feature':<22} | {'Type':<16} | {'Window':>6} | {'Leakage Risk'}")
    print("  " + "-" * 65)
    for _, r in task1_df.iterrows():
        print(f"  {r['feature']:<22} | {r['type']:<16} | {str(r['window_days']):>6} | {r['leakage_risk']}")

    # ─── Task 2: Shift-Based Leakage Test ───
    print("\n[Task 2] Shift-Based Leakage Test (Black-box Temporal Shift)")
    shift_results = []
    for shift_days in [1, 3, 7]:
        # Shift features forward by shift_days (simulate feeding "future" data to model)
        df_shifted = df.copy()
        df_shifted[WEATHER_FEATURES] = df[WEATHER_FEATURES].shift(shift_days).bfill()
        shifted_scores = infer_on_df(model,
                                     df_shifted[WEATHER_FEATURES].values,
                                     df[AGRO_FEATURES].values,
                                     scaler_w, scaler_a, device, dates_ts)
        recall_shifted = recall_at_threshold(shifted_scores, gt_df, threshold=60)
        fpr_shifted    = fpr_at_threshold(shifted_scores, gt_df, threshold=60)
        delta_recall   = recall_shifted - base_recall
        # If forward-shift doesn't hurt (delta ≥ 0) → potential leakage
        verdict = "LEAKAGE SUSPECTED" if delta_recall >= -0.05 else "CAUSAL"
        shift_results.append({'shift_days': shift_days,
                              'recall': round(recall_shifted, 4),
                              'fpr': round(fpr_shifted, 4),
                              'delta_recall': round(delta_recall, 4),
                              'verdict': verdict})
        print(f"  Shift +{shift_days}d: Recall={recall_shifted:.4f} (delta={delta_recall:+.4f}), "
              f"FPR={fpr_shifted:.4f} -> {verdict}")

    # ─── Task 3: Feature Isolation Influence Test ───
    print("\n[Task 3] Feature Isolation Influence Test (Zero-Masking)")
    influence_rows = []
    for feat_idx, feat in enumerate(WEATHER_FEATURES):
        w_vals = df[WEATHER_FEATURES].values.copy()
        w_vals[:, feat_idx] = 0.0  # Zero out in raw space before scaling
        masked_scores  = infer_on_df(model, w_vals, df[AGRO_FEATURES].values,
                                     scaler_w, scaler_a, device, dates_ts)
        masked_recall  = recall_at_threshold(masked_scores, gt_df, threshold=60)
        masked_fpr     = fpr_at_threshold(masked_scores, gt_df, threshold=60)
        delta_r = masked_recall - base_recall
        delta_f = masked_fpr - base_fpr
        influence = "HIGH" if abs(delta_r) > 0.1 else ("MEDIUM" if abs(delta_r) > 0.02 else "LOW")
        # Flag if highly influential AND rolling/derived
        meta = FEATURE_METADATA.get(feat, {})
        suspicious = influence == "HIGH" and meta.get('type') in ['rolling_window', 'derived']
        influence_rows.append({
            'feature': feat, 'recall_masked': round(masked_recall, 4),
            'fpr_masked': round(masked_fpr, 4),
            'delta_recall': round(delta_r, 4), 'influence': influence,
            'leakage_candidate': 'YES' if suspicious else 'no'
        })
        print(f"  {feat:<22}: Recall={masked_recall:.4f} (delta={delta_r:+.4f}), "
              f"Influence={influence}, LeakageCandidate={suspicious}")

    influence_df = pd.DataFrame(influence_rows)

    # ─── Task 4: Temporal Causality Check ───
    print("\n[Task 4] Temporal Causality Check — Feature Trajectories Around Events")
    causality_rows = []
    key_features = ['RH2M', 'rainfall_sum_3', 'RH2M_mean_14', 'RH2M_diff_1']
    for _, event in gt_df.iterrows():
        estart = event['peak_start']
        window = pd.date_range(estart - timedelta(days=14), estart + timedelta(days=3))
        sub = df[df['date'].isin(window)][['date'] + key_features].copy()
        if len(sub) < 5:
            continue
        pre  = sub[sub['date'] < estart]
        post = sub[sub['date'] >= estart]
        for feat in key_features:
            if feat not in sub.columns:
                continue
            pre_mean  = pre[feat].mean()
            post_mean = post[feat].mean()
            # Feature spikes BEFORE outbreak = causal
            direction = 'BEFORE (causal)' if pre_mean >= post_mean * 0.9 else 'AFTER (suspicious)'
            causality_rows.append({
                'event': estart.strftime('%Y-%m-%d'),
                'feature': feat,
                'pre_mean': round(pre_mean, 3),
                'post_mean': round(post_mean, 3),
                'direction': direction
            })

    causality_df = pd.DataFrame(causality_rows)
    print(f"\n  {'Event':<12} | {'Feature':<18} | {'Pre':<8} | {'Post':<8} | Direction")
    print("  " + "-" * 72)
    for _, r in causality_df.iterrows():
        print(f"  {r['event']:<12} | {r['feature']:<18} | {r['pre_mean']:<8} | "
              f"{r['post_mean']:<8} | {r['direction']}")

    # ─── Final Verdict ───
    shift_leak   = any(r['verdict'] == 'LEAKAGE SUSPECTED' for r in shift_results)
    feature_leak = any(r['leakage_candidate'] == 'YES' for _, r in influence_df.iterrows())
    causal_after = any('suspicious' in r['direction'] for _, r in causality_df.iterrows())

    if shift_leak and feature_leak:
        verdict = "LEAKY"
    elif shift_leak or (feature_leak and causal_after):
        verdict = "SUSPICIOUS"
    else:
        verdict = "CLEAN"

    print(f"\n{'='*62}")
    print(f"LEAKAGE STATUS: {verdict}")
    print(f"  Shift Test:           {'FAILED (leakage)' if shift_leak else 'PASSED'}")
    print(f"  Feature Influence:    {'Suspicious candidates found' if feature_leak else 'No high-influence suspicious features'}")
    print(f"  Temporal Causality:   {'Post-onset spikes found' if causal_after else 'All pre-onset'}")
    print(f"{'='*62}")

    # ─── Save Report ───
    report_path = os.path.join(OUTPUT_DIR, "sangli_leakage_audit.txt")
    with open(report_path, "w") as f:
        f.write("V9 TCN DATA LEAKAGE AUDIT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Baseline: Recall={base_recall:.4f}, FPR={base_fpr:.4f}\n\n")
        f.write("Task 1: Feature Dependency Audit\n")
        f.write(task1_df.to_string(index=False) + "\n\n")
        f.write("Task 2: Shift-Based Leakage Test\n")
        f.write(pd.DataFrame(shift_results).to_string(index=False) + "\n\n")
        f.write("Task 3: Feature Isolation Influence\n")
        f.write(influence_df.to_string(index=False) + "\n\n")
        f.write("Task 4: Temporal Causality\n")
        f.write(causality_df.to_string(index=False) + "\n\n")
        f.write(f"LEAKAGE STATUS: {verdict}\n")

    print(f"\nFull audit saved to: {report_path}")

if __name__ == "__main__":
    main()
