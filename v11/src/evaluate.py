"""
V11 KG-CTCN Evaluation Engine
--------------------------------------------------
Quantifies the strictly causal Early Warning capability of the system.
Tests generalisation to the unseen Test period (2022-2024).

CHANGELOG (v11.1):
  - [CRITICAL]  load_metadata() switched from .txt parser to json.load().
                train.py now writes v11_metadata.json.
  - [CRITICAL]  Removed weather StandardScaler. Weather features are already
                90-day rolling Z-scores from the pipeline. Agro scaler is
                loaded from agro_scaler.pkl (fitted on train split in train.py)
                rather than refitted here — refitting on a different state
                breaks exact reproducibility with the saved model.
  - [CRITICAL]  warmup_mask filter applied before any split or inference.
                Sequences built from unfiltered data include windows that
                overlap NaN warm-up rows, producing NaN model outputs.
  - [BUG]       Threshold search direction reversed. FPR decreases as
                threshold increases; searching ascending found the lowest
                threshold satisfying FPR ≤ 0.05 (nearly always ~0.01),
                producing a very low-precision operating point. Descending
                search now finds the most conservative (highest) threshold
                that still meets the FPR budget.
  - [BUG]       Val-as-fallback now uses a fixed threshold (0.5) instead of
                optimising threshold on the same split being evaluated.
                Optimising and evaluating on the same split produces optimistic
                bias and is flagged with an explicit warning.
  - [BUG]       Lead time semantics documented explicitly. The window search
                iterates earliest→latest and breaks on first hit, so reported
                lead time is the EARLIEST alert within the 3-7 day window
                (maximum possible lead). This is the correct metric for an
                early warning system and is now stated clearly.
  - [DESIGN]    run_inference() refactored from a closure to an explicit-
                parameter function. Scaler dependency is now visible at the
                call site; function is independently testable.
  - [DESIGN]    'fpr' variable aliased to avoid two meanings in the same
                scope (threshold search target vs. confusion matrix metric).
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import timedelta
import torch
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score

from model import KGCTCN
from assign_causal_labels import assign_labels

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

GT_PATH = os.path.join(BASE_DIR, "research_comp", "evidence_base", "outbreak_events", "sangli_gt_v2.csv")

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def load_metadata() -> dict:
    """Load model metadata from v11_metadata.json (written by train.py v11.1)."""
    with open(os.path.join(MODEL_DIR, "v11_metadata.json"), "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    df_subset: pd.DataFrame,
    weather_features: list,
    agro_features: list,
    seq_len: int,
    a_sc,           # fitted StandardScaler for agro features
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Run model inference on a contiguous dataframe subset.

    Weather features are passed directly (already Z-scored by the pipeline).
    Agro features are transformed with the provided fitted scaler.

    Args:
        df_subset        : filtered, reset-index dataframe (warmup rows excluded)
        weather_features : ordered list of weather feature column names
        agro_features    : ordered list of agro feature column names
        seq_len          : TCN receptive window length
        a_sc             : fitted StandardScaler for agro features
        model            : loaded KGCTCN model in eval mode
        device           : torch device

    Returns:
        probs      : (N,) float32 sigmoid risk probabilities
        y_true     : (N,) float32 ground truth labels
        dates_out  : (N,) DatetimeIndex of prediction dates
    """
    # Weather: no scaler — pipeline already produces Z-scored values
    w_vals = df_subset[weather_features].values.astype(np.float32)
    a_vals = a_sc.transform(df_subset[agro_features].values.astype(np.float32))
    labels = df_subset["risk_label"].values
    dates  = df_subset["date"].values

    X_w, X_a, y_arr, dates_arr = [], [], [], []
    for i in range(seq_len, len(df_subset)):
        X_w.append(w_vals[i - seq_len + 1 : i + 1])
        X_a.append(a_vals[i])
        y_arr.append(labels[i])
        dates_arr.append(dates[i])

    X_w_t = torch.FloatTensor(np.array(X_w, dtype=np.float32)).to(device)
    X_a_t = torch.FloatTensor(np.array(X_a, dtype=np.float32)).to(device)

    with torch.no_grad():
        _, probs_t, _ = model(X_w_t, X_a_t)

    return (
        probs_t.cpu().numpy().flatten(),
        np.array(y_arr, dtype=np.float32),
        pd.to_datetime(np.array(dates_arr)),
    )


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def find_threshold_at_fpr(
    probs: np.ndarray,
    y_true: np.ndarray,
    max_fpr: float = 0.05,
    fallback: float = 0.5,
) -> float:
    """
    Return the highest threshold t such that FPR(t) ≤ max_fpr.

    Searching in descending order ensures we find the most conservative
    (highest) threshold that still satisfies the FPR budget — maximising
    precision while respecting the recall constraint.

    FPR = FP / (FP + TN).  As threshold increases, FPR decreases.
    Ascending search (0.01→0.99) finds the FIRST threshold where FPR drops
    below the budget, which is often near 0.01 and produces near-zero
    precision. Descending search finds the LAST (highest) such threshold.

    Returns `fallback` if no threshold in the search grid satisfies the
    constraint (all thresholds produce FPR > max_fpr).
    """
    n_neg = (y_true == 0).sum()
    if n_neg == 0:
        return fallback

    for t in np.linspace(0.99, 0.01, 99):   # descending: 0.99 → 0.01
        fpr_t = ((probs >= t) & (y_true == 0)).sum() / n_neg
        if fpr_t <= max_fpr:
            return float(round(t, 4))

    return fallback


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate():
    print("=" * 60)
    print(" V11 KG-CTCN EVALUATION (EARLY WARNING METRICS)")
    print("=" * 60)

    # ── Load metadata and scalers ────────────────────────────────────────────
    meta = load_metadata()
    weather_features = meta["weather_features"]
    agro_features    = meta["agro_features"]
    seq_len          = int(meta["seq_len"])

    # Agro scaler: load the exact instance fitted on the training split in
    # train.py. Do NOT refit here — a different random state or inclusion of
    # warmup rows produces a different scaler and breaks model calibration.
    a_sc = joblib.load(os.path.join(MODEL_DIR, "agro_scaler.pkl"))

    # ── Load dataset and apply warmup filter ─────────────────────────────────
    df = pd.read_csv(os.path.join(DATA_DIR, "v11_features.csv"))
    df["date"] = pd.to_datetime(df["date"])

    n_before = len(df)
    df = df[df["warmup_mask"] == 0].reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} warm-up rows. Remaining: {len(df)}")

    # ── Assign Labels (In-Memory) ──────────────────────────────────────────
    df = assign_labels(df, GT_PATH)

    # ── Chronological splits ─────────────────────────────────────────────────
    train_mask = (df["date"].dt.year >= 2005) & (df["date"].dt.year <= 2018)
    val_mask   = (df["date"].dt.year >= 2019) & (df["date"].dt.year <= 2021)
    test_mask  = (df["date"].dt.year >= 2022) & (df["date"].dt.year <= 2024)

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

    # Shared kwargs for run_inference calls
    infer_kwargs = dict(
        weather_features=weather_features,
        agro_features=agro_features,
        seq_len=seq_len,
        a_sc=a_sc,
        model=model,
        device=device,
    )

    # ── Val set: threshold calibration ───────────────────────────────────────
    df_val    = df[val_mask].reset_index(drop=True)
    v_probs, v_y, _ = run_inference(df_val, **infer_kwargs)
    opt_thr   = find_threshold_at_fpr(v_probs, v_y, max_fpr=0.05, fallback=0.5)
    print(f"\nThreshold calibrated on val set (FPR ≤ 5%): {opt_thr:.4f}")

    # ── Test set: inference ───────────────────────────────────────────────────
    df_test   = df[test_mask].reset_index(drop=True)
    using_val_as_test = False

    if len(df_test) < seq_len or df_test["risk_label"].sum() == 0:
        print(
            "\nWARNING: Test set (2022-2024) has no positive labels or is too small.\n"
            "  Falling back to Validation Set (2019-2021) for event-level reporting.\n"
            "  IMPORTANT: Threshold was optimised on this same split — day-level\n"
            "  metrics will be optimistically biased. Use a fixed threshold of 0.5\n"
            "  for any published comparison."
        )
        df_test          = df_val
        probs, y_true, dates_test = v_probs, v_y, _
        # Override threshold with fixed value to avoid eval-on-train-split bias
        opt_thr          = 0.5
        using_val_as_test = True
    else:
        probs, y_true, dates_test = run_inference(df_test, **infer_kwargs)

    # ── Ground truth events in the evaluation window ─────────────────────────
    gt_df = pd.read_csv(GT_PATH)
    gt_df["peak_start"] = pd.to_datetime(gt_df["peak_start"])

    eval_year_min = df_test["date"].dt.year.min()
    eval_year_max = df_test["date"].dt.year.max()
    eval_peaks = gt_df[
        (gt_df["peak_start"].dt.year >= eval_year_min) &
        (gt_df["peak_start"].dt.year <= eval_year_max)
    ]

    # ── Day-level metrics ────────────────────────────────────────────────────
    preds = (probs >= opt_thr).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())

    recall    = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    # FPR = FP / (FP + TN) — named explicitly to distinguish from the
    # threshold-search variable 'fpr_t' in find_threshold_at_fpr()
    false_positive_rate = fp / max(1, fp + tn)

    auc    = roc_auc_score(y_true, probs) if y_true.sum() > 0 else float("nan")
    avg_pr = average_precision_score(y_true, probs) if y_true.sum() > 0 else float("nan")

    print("\n[ DAY-LEVEL METRICS ]")
    if using_val_as_test:
        print("  (!) Threshold = 0.5 fixed — eval split equals calibration split.")
    print(f"  Operating Threshold   : {opt_thr:.4f}")
    print(f"  Recall (Outbreaks)    : {recall:.4f}")
    print(f"  Precision             : {precision:.4f}")
    print(f"  False Positive Rate   : {false_positive_rate:.4f}")
    print(f"  ROC-AUC               : {auc:.4f}")
    print(f"  Avg Precision (PR-AUC): {avg_pr:.4f}")
    print(f"  TP={tp} | FP={fp} | TN={tn} | FN={fn}")

    # ── Event-level early warning metrics ────────────────────────────────────
    # For each known outbreak peak, check whether any alert fires in the
    # lead window [peak-10d, peak-7d].
    print("\n[ EVENT-LEVEL EARLY WARNING (7-10 DAY LEAD) ]")
    scores_dict = dict(zip(dates_test, probs))

    detected_events = 0
    lead_times      = []

    for _, row in eval_peaks.iterrows():
        peak         = row["peak_start"]
        window_start = peak - timedelta(days=10)
        window_end   = peak - timedelta(days=7)

        earliest_alert_date = None
        for d in pd.date_range(window_start, window_end):
            if scores_dict.get(d, 0.0) >= opt_thr:
                earliest_alert_date = d
                break   # earliest hit = maximum lead time

        if earliest_alert_date is not None:
            lead = (peak - earliest_alert_date).days
            detected_events += 1
            lead_times.append(lead)
            print(f"  Outbreak {peak.date()} → DETECTED  (earliest alert: "
                  f"{earliest_alert_date.date()}, lead = {lead} days)")
        else:
            print(f"  Outbreak {peak.date()} → MISSED")

    n_events = len(eval_peaks)
    edr      = (detected_events / max(n_events, 1)) * 100
    avg_lead = float(np.mean(lead_times)) if lead_times else 0.0

    print(f"\n  Event Detection Rate : {edr:.1f}%  ({detected_events}/{n_events})")
    print(f"  Average Lead Time    : {avg_lead:.1f} days  "
          f"(earliest alert per event; max possible = 10)")

    # ── Save predictions ─────────────────────────────────────────────────────
    out_df = pd.DataFrame({
        "date":       dates_test,
        "risk_score": probs,
        "alert":      preds,
        "true_label": y_true,
    })
    out_path = os.path.join(OUTPUT_DIR, "v11_evaluation_scores.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved predictions to {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()