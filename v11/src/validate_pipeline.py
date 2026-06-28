"""
V11 KG-CTCN Comprehensive Pipeline Validation
--------------------------------------------------
Validates the entire pipeline against the ground truth dataset:
research_comp/evidence_base/outbreak_events/sangli_synthetic_gt.csv

Produces a comprehensive report for all events across all years.

CHANGELOG (v11.4):
  - [CRITICAL]  FPR audit now uses the SAME decision logic as
                inference_engine.py: temperature-calibrated scores +
                KG biological gate + operational MEDIUM_THRESHOLD (0.20).
                Previously, validate_pipeline.py bypassed inference_engine.py
                entirely — it applied sigmoid(logits/T) then threshold'd at
                the F2-optimal value (0.0861 from val set). The KG gate in
                inference_engine.py never ran, so the reported 19.2% FPR was
                measuring a different decision function than production uses.
  - [DESIGN]    KG_APPLY_GATE flag (default True) allows gate to be disabled
                for ablation studies without editing production code.
  - [DESIGN]    FPR audit now reports gated vs ungated day counts so the
                contribution of the biological gate is visible.

CHANGELOG (v11.3):
  - [BUG]       Detection window was hardcoded as [peak-14, peak-7] but
                assign_causal_labels_v2 labels [peak-10, peak-7]. Window now
                imported from assign_causal_labels_v2 as LABEL_WINDOW_FAR /
                LABEL_WINDOW_NEAR — single source of truth.

CHANGELOG (v11.2):
  - [CRITICAL]  GT_PATH updated to sangli_gt_v2.csv.
  - [CRITICAL]  assign_causal_labels_v2.assign_labels() called at runtime.
  - [CRITICAL]  FileNotFoundError guard added for GT_PATH.

CHANGELOG (v11.1):
  - [CRITICAL]  load_metadata() switched to json.load().
  - [CRITICAL]  Removed weather StandardScaler.
  - [CRITICAL]  warmup_mask filter applied before sequence building.
  - [BUG]       Threshold search reversed to descending.
  - [BUG]       Val / FP label alignment fixed.
  - [DESIGN]    Full dataset no longer pre-allocated on device.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import timedelta
import torch
import joblib
from sklearn.metrics import precision_recall_curve, average_precision_score

from model import KGCTCN
from assign_causal_labels_v2 import assign_labels, LABEL_WINDOW_FAR, LABEL_WINDOW_NEAR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

GT_PATH = os.path.join(
    BASE_DIR,
    "research_comp", "evidence_base", "outbreak_events", "sangli_gt_v2.csv",
)

BATCH_SIZE = 512

# ---------------------------------------------------------------------------
# Decision thresholds — must match inference_engine.py exactly
# ---------------------------------------------------------------------------

MEDIUM_THRESHOLD = 0.20   # operational alert threshold (not F2-optimal)
HIGH_THRESHOLD   = 0.55

# KG biological gate — same constants as inference_engine.py
KG_RH_PERSIST_MIN = 2.0   # days of high RH required for sporangium dispersal
KG_RAIN_SUM_MIN   = 5.0   # mm over 7 days required for surface wetness
KG_GATE_CAP       = 0.15  # max score when gate is closed (dry week)
KG_APPLY_GATE     = True  # set False to ablate gate contribution


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def load_metadata() -> dict:
    with open(os.path.join(MODEL_DIR, "v11_metadata.json"), "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# KG biological gate  (mirrors inference_engine.py apply_kg_gate)
# ---------------------------------------------------------------------------

def apply_kg_gate(score: float, rh_persist_7d: float, rain_sum_7d: float) -> tuple:
    """
    Returns (gated_score, gate_open).
    gate_open=False: biological conditions for outbreak are absent;
    score is capped at KG_GATE_CAP regardless of model output.
    """
    if not KG_APPLY_GATE:
        return score, True
    dry_week = (rh_persist_7d < KG_RH_PERSIST_MIN) and (rain_sum_7d < KG_RAIN_SUM_MIN)
    if dry_week:
        return min(score, KG_GATE_CAP), False
    return score, True


# ---------------------------------------------------------------------------
# Threshold calibration (kept for informational Val AP reporting only)
# ---------------------------------------------------------------------------

def find_optimal_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f2_scores = (5 * precision * recall) / (4 * precision + recall + 1e-8)
    best_idx = np.argmax(f2_scores)
    return float(thresholds[best_idx])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 68)
    print(" V11 KG-CTCN COMPREHENSIVE PIPELINE VALIDATION REPORT")
    print("=" * 68)

    # ── Metadata & scalers ───────────────────────────────────────────────────
    meta             = load_metadata()
    weather_features = meta["weather_features"]
    agro_features    = meta["agro_features"]
    seq_len          = int(meta["seq_len"])

    a_sc = joblib.load(os.path.join(MODEL_DIR, "agro_scaler.pkl"))
    T    = joblib.load(os.path.join(MODEL_DIR, "temperature.pkl"))
    print(f"Loaded temperature T = {T:.4f}")
    print(f"Decision threshold   : {MEDIUM_THRESHOLD:.4f}  (operational, matches inference_engine.py)")
    print(f"KG biological gate   : {'ENABLED' if KG_APPLY_GATE else 'DISABLED (ablation mode)'}")

    # ── Load dataset ─────────────────────────────────────────────────────────
    df = pd.read_csv(os.path.join(DATA_DIR, "v11_features.csv"))
    df["date"] = pd.to_datetime(df["date"])

    n_before = len(df)
    df = df[df["warmup_mask"] == 0].reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} warm-up rows. Remaining: {len(df)}")

    # ── Assign labels ────────────────────────────────────────────────────────
    if not os.path.exists(GT_PATH):
        raise FileNotFoundError(
            f"GT file not found: {GT_PATH}\n"
            "Check that sangli_gt_v2.csv exists in the outbreak_events directory."
        )
    df = assign_labels(df, gt_path=GT_PATH)

    gt_df = pd.read_csv(GT_PATH)
    gt_df["peak_start"] = pd.to_datetime(gt_df["peak_start"])

    # ── Load model ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = KGCTCN(len(weather_features), len(agro_features)).to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, "v11_kg_ctcn.pth"),
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    # ── Build sequences ───────────────────────────────────────────────────────
    # Weather: no scaler — pipeline already Z-scored.
    # Agro: transform with the saved scaler (do NOT refit).
    w_vals = df[weather_features].values.astype(np.float32)
    a_vals = a_sc.transform(df[agro_features].values.astype(np.float32))
    labels = df["risk_label"].values.astype(np.float32)
    dates  = []

    X_w_list, X_a_list, label_list = [], [], []
    for i in range(seq_len, len(df)):
        X_w_list.append(w_vals[i - seq_len + 1 : i + 1])
        X_a_list.append(a_vals[i])
        label_list.append(labels[i])
        dates.append(df["date"].iloc[i])

    dates     = pd.to_datetime(dates)
    label_arr = np.array(label_list, dtype=np.float32)  # aligned with dates

    X_w_np = np.array(X_w_list, dtype=np.float32)
    X_a_np = np.array(X_a_list, dtype=np.float32)

    # ── Batch inference — raw calibrated probabilities ────────────────────────
    # Tensors moved to device inside loop — avoids pre-allocating full dataset
    # on VRAM, which fails on low-memory GPUs.
    print("Running full-dataset inference...")
    raw_probs = []
    with torch.no_grad():
        for start in range(0, len(X_w_np), BATCH_SIZE):
            bw = torch.FloatTensor(X_w_np[start : start + BATCH_SIZE]).to(device)
            ba = torch.FloatTensor(X_a_np[start : start + BATCH_SIZE]).to(device)
            logits, _, _ = model(bw, ba)
            probs = torch.sigmoid(logits / T)
            raw_probs.extend(probs.cpu().numpy().flatten())

    raw_probs = np.array(raw_probs, dtype=np.float32)

    # ── Apply KG biological gate ──────────────────────────────────────────────
    # RH_persist_7d and Rain_sum_7d are read from df rows that correspond to
    # the sequence endpoints (indices seq_len..len(df)-1), which is exactly the
    # same alignment used when building sequences above.
    rh_persist_vals = df["RH_persist_7d"].values[seq_len:]
    rain_sum_vals   = df["Rain_sum_7d"].values[seq_len:]

    gated_probs   = np.empty_like(raw_probs)
    gate_open_arr = np.ones(len(raw_probs), dtype=bool)

    for idx in range(len(raw_probs)):
        gs, go = apply_kg_gate(
            float(raw_probs[idx]),
            float(rh_persist_vals[idx]),
            float(rain_sum_vals[idx]),
        )
        gated_probs[idx]   = gs
        gate_open_arr[idx] = go

    n_gated = int((~gate_open_arr).sum())

    # ── scores_dict for event-level detection ─────────────────────────────────
    # Uses GATED scores + operational threshold — same as inference_engine.py.
    scores_dict = dict(zip(dates, gated_probs))

    # ── Val AP (informational — uses raw probs to measure model quality) ──────
    val_mask   = np.array([2019 <= d.year <= 2021 for d in dates])
    val_probs  = raw_probs[val_mask]
    val_labels = label_arr[val_mask]
    val_ap     = (
        average_precision_score(val_labels, val_probs)
        if val_labels.sum() > 0
        else float("nan")
    )
    opt_thr = find_optimal_threshold(val_probs, val_labels)

    print(f"\n[ PERFORMANCE ]")
    print(f"  Val AP (2019-2021)         : {val_ap:.4f}")
    print(f"  F2-optimal threshold (info): {opt_thr:.4f}  (not used for FPR audit)")
    print(f"  Operational threshold used : {MEDIUM_THRESHOLD:.4f}")

    # ── Event-level validation ────────────────────────────────────────────────
    print("\n[ EVENT-LEVEL VALIDATION - ALL YEARS ]")
    print(
        "  NOTE: Train detections are IN-SAMPLE (memorisation check).\n"
        "        Val detections are OUT-OF-SAMPLE (generalisation metric).\n"
    )
    header = f"{'Peak Date':<16}  {'Split':<8}  {'Status':<12}  {'Lead Time'}"
    print(header)
    print("-" * 55)

    counts = {"Train": [0, 0], "Val": [0, 0], "Test": [0, 0]}

    for _, row in gt_df.sort_values("peak_start").iterrows():
        peak  = row["peak_start"]
        yr    = peak.year
        split = "Train" if yr <= 2018 else ("Val" if yr <= 2021 else "Test")
        counts[split][1] += 1

        # Detection window: exactly the actionable label window from
        # assign_causal_labels_v2 — [peak - FAR, peak - NEAR].
        window_start = peak - timedelta(days=LABEL_WINDOW_FAR)
        window_end   = peak - timedelta(days=LABEL_WINDOW_NEAR)

        earliest_alert = None
        for d in pd.date_range(window_start, window_end):
            score = scores_dict.get(d, 0.0)
            if score >= MEDIUM_THRESHOLD:
                earliest_alert = d
                break

        if earliest_alert is not None:
            lead = (peak - earliest_alert).days
            counts[split][0] += 1
            status   = "DETECTED"
            lead_str = f"{lead} days"
        else:
            status   = "MISSED"
            lead_str = "N/A"

        print(f"  {peak.date().isoformat():<16}{split:<8}  {status:<12}  {lead_str}")

    print("-" * 55)
    print("\n[ SUMMARY REPORT ]")
    print(f"  Total GT events: {len(gt_df)}")

    for split, (det, tot) in counts.items():
        if tot == 0:
            label = f"  {split:<8}: N/A (0 events in GT)"
        else:
            rate  = det / tot * 100
            note  = " <- in-sample" if split == "Train" else ""
            label = f"  {split:<8}: {rate:.1f}%  ({det}/{tot}){note}"
        print(label)

    # ── False positive audit ──────────────────────────────────────────────────
    # GATED scores + operational threshold — matches production inference path.
    # Counterfactual (raw, ungated) shown for comparison so you can see
    # how much the KG gate alone is contributing to FPR reduction.
    all_preds = (gated_probs >= MEDIUM_THRESHOLD).astype(int)

    fp_total  = int(((all_preds == 1) & (label_arr == 0)).sum())
    tn_total  = int(((all_preds == 0) & (label_arr == 0)).sum())
    fpr_total = fp_total / max(1, fp_total + tn_total)

    all_preds_ungated = (raw_probs >= MEDIUM_THRESHOLD).astype(int)
    fp_ungated  = int(((all_preds_ungated == 1) & (label_arr == 0)).sum())
    fpr_ungated = fp_ungated / max(1, fp_ungated + tn_total)

    print(f"\n[ FALSE POSITIVE AUDIT ]")
    print(f"  Days capped by KG gate (dry weeks): {n_gated}  ({n_gated / len(raw_probs) * 100:.1f}% of all days)")
    print(f"  FPR without KG gate (informational): {fpr_ungated * 100:.2f}%")
    print(f"  FPR with KG gate    (production)   : {fpr_total * 100:.2f}%  (target: < 5%)")
    print(f"  False alarm days after gating      : {fp_total}")

    status_str = (
        "VALIDATED  — FPR constraint met, causality intact"
        if fpr_total <= 0.05
        else "WARNING    — FPR exceeds 5% target"
    )
    print(f"\n  >> STATUS: {status_str}")
    print("=" * 68)


if __name__ == "__main__":
    main()