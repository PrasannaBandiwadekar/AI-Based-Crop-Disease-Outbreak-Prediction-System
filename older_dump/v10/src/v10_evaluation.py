"""
V10.6 COMPREHENSIVE EVALUATION
===============================

Reports:
  1. Standard classification metrics (Recall, Precision, FPR, AUC-PR)
  2. Temporal metrics (Detection Lead Time, first-hit accuracy per outbreak)
  3. Causality audit (Shift +1, +3, +5 degradation test)
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import timedelta

import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, roc_auc_score
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_tcn import V10TCNModel

import warnings
warnings.filterwarnings("ignore")

# ── PATHS ──────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "features.csv")
MODEL_PATH    = os.path.join(BASE_DIR, "models", "v10_tcn_model.pth")
META_PATH     = os.path.join(BASE_DIR, "models", "v10_metadata.txt")
GT_PATH       = os.path.join(
    os.path.dirname(BASE_DIR),
    "research_comp", "evidence_base", "outbreak_events", "sangli_synthetic_gt.csv"
)
OUTPUT_DIR    = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LEN = 14

WEATHER_FEATURES = [
    'RH_z90',
    'Trigger_3d_sum',
    'RH_persist',
    'RH_season_z',
    'Rain_sum_7',
]
AGRO_FEATURES = ['ratoon_flag', 'sanitation_score']


# ─────────────────────────────────────────────────────────────────────────────
def load_threshold():
    try:
        with open(META_PATH) as f:
            for line in f:
                if line.startswith("optimal_threshold"):
                    return float(line.split("=")[1].strip())
    except Exception:
        pass
    return 0.5


def load_model(device):
    model = V10TCNModel(len(WEATHER_FEATURES), len(AGRO_FEATURES)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    return model


def run_inference(model, df_w, df_a, scaler_w, scaler_a, dates, device):
    """Returns dict {Timestamp: risk_prob}."""
    w_sc = scaler_w.transform(df_w)
    a_sc = scaler_a.transform(df_a)
    scores = {}
    with torch.no_grad():
        for i in range(SEQ_LEN - 1, len(dates)):
            w_t = torch.FloatTensor(w_sc[i - SEQ_LEN + 1: i + 1]).unsqueeze(0).to(device)
            a_t = torch.FloatTensor(a_sc[i]).reshape(1, -1).to(device)
            logits, _ = model(w_t, a_t)
            scores[dates[i]] = torch.sigmoid(logits).item()
    return scores


# ─────────────────────────────────────────────────────────────────────────────
def compute_flat_metrics(scores, y_true_map, threshold):
    """Day-level Recall, Precision, FPR at given threshold."""
    dates = list(scores.keys())
    probs = np.array([scores[d] for d in dates])
    yt    = np.array([y_true_map.get(d, 0) for d in dates])

    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (yt == 1)).sum())
    fp = int(((preds == 1) & (yt == 0)).sum())
    fn = int(((preds == 0) & (yt == 1)).sum())
    tn = int(((preds == 0) & (yt == 0)).sum())

    recall    = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    fpr       = fp / max(fp + tn, 1)
    ap        = average_precision_score(yt, probs) if yt.sum() > 0 else 0.0

    return recall, precision, fpr, ap, tp, fp, fn, tn


def compute_temporal_metrics(scores, gt_df, threshold):
    """
    Event-level detection.  An outbreak is 'detected' if at least one day in
    its [peak-7, peak-3] early-warning window exceeds threshold.
    Returns (success_rate%, avg_lead_time_days, per_event_details).
    """
    detected   = 0
    lead_times = []
    details    = []

    for _, event in gt_df.iterrows():
        peak   = event['peak_start']
        win_s  = peak - timedelta(days=7)
        win_e  = peak - timedelta(days=3)

        hit     = False
        lead_d  = None
        for d in pd.date_range(win_s, win_e):
            if scores.get(d, 0) >= threshold:
                hit    = True
                lead_d = (peak - d).days
                break

        if hit:
            detected += 1
            lead_times.append(lead_d)

        details.append({
            'peak_start': peak.date(),
            'detected': hit,
            'lead_days': lead_d,
        })

    n = len(gt_df)
    success_rate   = 100.0 * detected / max(n, 1)
    avg_lead_time  = float(np.mean(lead_times)) if lead_times else 0.0

    return success_rate, avg_lead_time, detected, n, pd.DataFrame(details)


def shift_recall(model, df_feat, scaler_w, scaler_a, dates_ts, gt_df, threshold, shift_days, device):
    """
    Causality test: shift weather features forward by `shift_days` rows,
    simulating use of future data.  Recall should drop.
    """
    df_sh = df_feat.copy()
    df_sh[WEATHER_FEATURES] = df_feat[WEATHER_FEATURES].shift(shift_days).bfill()

    s_scores = run_inference(
        model,
        df_sh[WEATHER_FEATURES].values,
        df_sh[AGRO_FEATURES].values,
        scaler_w, scaler_a, dates_ts, device
    )
    _, _, s_det, s_tot, _ = compute_temporal_metrics(s_scores, gt_df, threshold)
    return s_det / max(s_tot, 1)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 66)
    print("  V10.6 FULL EVALUATION — TEST SET + CAUSALITY AUDIT")
    print("=" * 66)

    df = pd.read_csv(FEATURES_PATH)
    df['date'] = pd.to_datetime(df['date'])

    gt_df = pd.read_csv(GT_PATH)
    gt_df['peak_start'] = pd.to_datetime(gt_df['peak_start'])

    # Scalers fitted on training data only (same boundary as train.py: 2005-2015)
    train_mask = (df['date'].dt.year >= 2005) & (df['date'].dt.year <= 2015)
    scaler_w   = StandardScaler().fit(df.loc[train_mask, WEATHER_FEATURES])
    scaler_a   = StandardScaler().fit(df.loc[train_mask, AGRO_FEATURES])

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = load_model(device)
    threshold = load_threshold()

    print(f"\n  Model loaded | Threshold = {threshold:.4f}")

    # ── TEST SET (2019–2024) ──────────────────────────────────────────────
    df_test  = df[df['date'].dt.year >= 2019].reset_index(drop=True)
    gt_test  = gt_df[gt_df['peak_start'].dt.year >= 2019].reset_index(drop=True)
    dates_ts = [pd.Timestamp(d) for d in df_test['date'].values]

    print(f"\n  Test period: {df_test['date'].min().date()} -> {df_test['date'].max().date()}")
    print(f"  Test outbreaks: {len(gt_test)}")

    test_scores = run_inference(
        model,
        df_test[WEATHER_FEATURES].values,
        df_test[AGRO_FEATURES].values,
        scaler_w, scaler_a, dates_ts, device
    )

    # Build day-level ground truth map for test set
    y_true_map = {}
    for idx, row in df_test.iterrows():
        d = pd.Timestamp(row['date'])
        y_true_map[d] = int(row['risk_label'])

    recall, precision, fpr, ap, tp, fp, fn, tn = compute_flat_metrics(
        test_scores, y_true_map, threshold
    )

    print("\n-- DAY-LEVEL METRICS (Test Set 2019-2024) --")
    print(f"  Recall    : {recall:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  FPR       : {fpr:.4f}")
    print(f"  AUC-PR    : {ap:.4f}")
    print(f"  TP={tp} | FP={fp} | FN={fn} | TN={tn}")

    # Success criteria check
    print("\n-- SUCCESS CRITERIA CHECK --")
    print(f"  Recall >= 0.70 :  {'PASS' if recall >= 0.70 else 'FAIL'}  ({recall:.4f})")
    print(f"  FPR    <= 0.05 :  {'PASS' if fpr    <= 0.05 else 'FAIL'}  ({fpr:.4f})")
    print(f"  FPR    > 0.02 :  {'PASS' if fpr    >  0.02 else 'FAIL'}  ({fpr:.4f})")

    # Temporal metrics
    sr, alt, det, tot, detail_df = compute_temporal_metrics(
        test_scores, gt_test, threshold
    )

    print("\n-- EVENT-LEVEL TEMPORAL METRICS --")
    print(f"  Total test outbreaks       : {tot}")
    print(f"  Detected (>=3 days early)  : {det}")
    print(f"  Event Detection Rate       : {sr:.1f}%")
    print(f"  Median/Avg Lead Time       : {alt:.2f} days")

    print("\n  Per-outbreak results:")
    for _, r in detail_df.iterrows():
        status = f"DETECTED  (lead={r['lead_days']}d)" if r['detected'] else "MISSED"
        print(f"    {r['peak_start']}  -> {status}")

    early_hit = det >= 1
    print(f"\n  >=1 early detection : {'PASS' if early_hit else 'FAIL'}")

    # ── CAUSALITY AUDIT ───────────────────────────────────────────────────
    print("\n-- CAUSALITY AUDIT (Temporal Shift Degradation) --")
    print("  Expected: Recall drops as forward shift increases")

    base_recall = det / max(tot, 1)

    if base_recall < 0.15:
        print("  [SKIP] Baseline recall too low for meaningful shift test.")
    else:
        for shift in [1, 3, 5]:
            s_recall = shift_recall(
                model, df_test, scaler_w, scaler_a,
                dates_ts, gt_test, threshold, shift, device
            )
            delta  = s_recall - base_recall
            status = "PASSED" if delta < 0 else ("SUSPECT" if delta >= 0 else "")
            if delta < -0.25:
                status = "PASSED (strong)"
            print(f"    Shift +{shift}d | Recall={s_recall:.4f} | Delta={delta:+.4f} | {status}")

    # Save scores
    scores_df = pd.DataFrame([
        {'date': d, 'risk_score': s, 'alert': int(s >= threshold)}
        for d, s in sorted(test_scores.items())
    ])
    scores_df.to_csv(os.path.join(OUTPUT_DIR, "v10_test_scores.csv"), index=False)
    print(f"\n  Scores saved -> outputs/v10_test_scores.csv")
    print("=" * 66)


if __name__ == "__main__":
    main()
