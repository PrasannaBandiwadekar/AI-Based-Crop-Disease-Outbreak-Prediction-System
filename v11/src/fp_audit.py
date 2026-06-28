"""
FP Diagnostic — understand where the 1024 false alarm days come from.
Run from the src/ directory alongside validate_pipeline.py.
Outputs: fp_audit_report.txt and fp_score_distribution.csv
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
import joblib

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
GT_PATH   = os.path.join(
    BASE_DIR, "research_comp", "evidence_base", "outbreak_events", "sangli_gt_v2.csv"
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import KGCTCN
from assign_causal_labels_v2 import assign_labels

MEDIUM_THRESHOLD  = 0.20
KG_RH_PERSIST_MIN = 2.0
KG_RAIN_SUM_MIN   = 5.0
KG_GATE_CAP       = 0.15
BATCH_SIZE        = 512

def main():
    meta = json.load(open(os.path.join(MODEL_DIR, "v11_metadata.json")))
    weather_features = meta["weather_features"]
    agro_features    = meta["agro_features"]
    seq_len          = int(meta["seq_len"])

    a_sc = joblib.load(os.path.join(MODEL_DIR, "agro_scaler.pkl"))
    T    = joblib.load(os.path.join(MODEL_DIR, "temperature.pkl"))

    df = pd.read_csv(os.path.join(DATA_DIR, "v11_features.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["warmup_mask"] == 0].reset_index(drop=True)
    df = assign_labels(df, gt_path=GT_PATH)

    device = torch.device("cpu")
    model  = KGCTCN(len(weather_features), len(agro_features)).to(device)
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "v11_kg_ctcn.pth"),
        map_location=device, weights_only=True
    ))
    model.eval()

    w_vals = df[weather_features].values.astype(np.float32)
    a_vals = a_sc.transform(df[agro_features].values.astype(np.float32))
    labels = df["risk_label"].values.astype(np.float32)

    X_w_list, X_a_list, label_list, dates = [], [], [], []
    for i in range(seq_len, len(df)):
        X_w_list.append(w_vals[i - seq_len + 1 : i + 1])
        X_a_list.append(a_vals[i])
        label_list.append(labels[i])
        dates.append(df["date"].iloc[i])

    dates     = pd.to_datetime(dates)
    label_arr = np.array(label_list, dtype=np.float32)
    X_w_np    = np.array(X_w_list, dtype=np.float32)
    X_a_np    = np.array(X_a_list, dtype=np.float32)

    raw_probs = []
    with torch.no_grad():
        for s in range(0, len(X_w_np), BATCH_SIZE):
            bw = torch.FloatTensor(X_w_np[s:s+BATCH_SIZE])
            ba = torch.FloatTensor(X_a_np[s:s+BATCH_SIZE])
            logits, _, _ = model(bw, ba)
            raw_probs.extend(torch.sigmoid(logits / T).cpu().numpy().flatten())
    raw_probs = np.array(raw_probs, dtype=np.float32)

    rh_persist = df["RH_persist_7d"].values[seq_len:]
    rain_sum   = df["Rain_sum_7d"].values[seq_len:]

    # Apply gate
    gated_probs = raw_probs.copy()
    gate_open   = np.ones(len(raw_probs), dtype=bool)
    for i in range(len(raw_probs)):
        if rh_persist[i] < KG_RH_PERSIST_MIN and rain_sum[i] < KG_RAIN_SUM_MIN:
            gated_probs[i] = min(gated_probs[i], KG_GATE_CAP)
            gate_open[i]   = False

    # FP mask
    fp_mask = (gated_probs >= MEDIUM_THRESHOLD) & (label_arr == 0)
    fp_dates  = dates[fp_mask]
    fp_scores = gated_probs[fp_mask]
    fp_rh     = rh_persist[fp_mask]
    fp_rain   = rain_sum[fp_mask]
    fp_years  = fp_dates.year

    # Build summary df
    fp_df = pd.DataFrame({
        "date":         fp_dates,
        "year":         fp_years,
        "score":        fp_scores,
        "RH_persist_7d": fp_rh,
        "Rain_sum_7d":   fp_rain,
        "month":        fp_dates.month,
    }).sort_values("score", ascending=False)

    out_csv = os.path.join(BASE_DIR, "data", "synthetic", "fp_score_distribution.csv")
    fp_df.to_csv(out_csv, index=False)

    # ── Report ────────────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 60)
    lines.append(" FALSE POSITIVE DIAGNOSTIC REPORT")
    lines.append("=" * 60)
    lines.append(f"Total FP days: {fp_mask.sum()}")
    lines.append("")

    lines.append("[ SCORE DISTRIBUTION OF FP DAYS ]")
    for thr in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]:
        n = int((fp_scores >= thr).sum())
        lines.append(f"  score >= {thr:.2f}: {n:4d} days  ({n/len(fp_scores)*100:.1f}%)")
    lines.append("")

    lines.append("[ FP DAYS BY YEAR ]")
    for yr, cnt in sorted(fp_df["year"].value_counts().items()):
        lines.append(f"  {yr}: {cnt:4d} days")
    lines.append("")

    lines.append("[ FP DAYS BY MONTH (across all years) ]")
    for mo, cnt in sorted(fp_df["month"].value_counts().items()):
        lines.append(f"  Month {mo:02d}: {cnt:4d} days")
    lines.append("")

    lines.append("[ TOP 20 FP DAYS (highest scores) ]")
    lines.append(f"  {'Date':<14} {'Score':>8}  {'RH_persist':>10}  {'Rain_sum':>10}")
    lines.append("  " + "-" * 50)
    for _, r in fp_df.head(20).iterrows():
        lines.append(f"  {str(r['date'].date()):<14} {r['score']:8.4f}  {r['RH_persist_7d']:10.2f}  {r['Rain_sum_7d']:10.2f}")
    lines.append("")

    lines.append("[ WHAT THRESHOLD WOULD HIT 5% FPR? ]")
    neg_count = int((label_arr == 0).sum())
    for thr in np.arange(0.20, 0.95, 0.01):
        fp_at_thr = int(((gated_probs >= thr) & (label_arr == 0)).sum())
        fpr_at_thr = fp_at_thr / neg_count
        # also check how many val events still detected at this threshold
        if fpr_at_thr <= 0.05:
            lines.append(f"  Threshold {thr:.2f} → FPR {fpr_at_thr*100:.2f}%  (FP days: {fp_at_thr})")
            lines.append(f"  (First threshold at or below 5% target)")
            break
    else:
        lines.append("  No threshold in [0.20, 0.95] achieves FPR <= 5%")
        lines.append("  Model score distribution is too flat — retraining needed.")
    lines.append("")

    lines.append("[ RECALL AT 5%-FPR THRESHOLD ]")
    for thr in np.arange(0.20, 0.95, 0.01):
        fp_at_thr  = int(((gated_probs >= thr) & (label_arr == 0)).sum())
        fpr_at_thr = fp_at_thr / neg_count
        if fpr_at_thr <= 0.05:
            tp = int(((gated_probs >= thr) & (label_arr == 1)).sum())
            fn = int(((gated_probs <  thr) & (label_arr == 1)).sum())
            recall = tp / max(1, tp + fn)
            lines.append(f"  Threshold {thr:.2f}: recall {recall:.3f}  ({tp} TP, {fn} FN)")
            break
    lines.append("=" * 60)

    report_str = "\n".join(lines)
    print(report_str)

    out_txt = os.path.join(BASE_DIR, "data", "synthetic", "fp_audit_report.txt")
    with open(out_txt, "w") as f:
        f.write(report_str)
    print(f"\nSaved: {out_txt}")
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    main()