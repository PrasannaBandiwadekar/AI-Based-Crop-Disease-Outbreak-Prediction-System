"""
V11 KG-CTCN Manual Performance Check
--------------------------------------------------
Interactive command-line script to inspect pipeline performance.

Usage:
    python check_performance.py                         # full report
    python check_performance.py --split val             # val only
    python check_performance.py --split train           # train only
    python check_performance.py --date 2019-06-19       # single date probe
    python check_performance.py --threshold 0.3         # override threshold
    python check_performance.py --no-temp               # skip temperature scaling
    python check_performance.py --top-fp 10             # show top-N false positives

Examples:
    python check_performance.py --split val --top-fp 5
    python check_performance.py --date 2019-09-10 --no-temp
"""

import argparse
import os
import sys
import json
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import joblib

# ── Path resolution ──────────────────────────────────────────────────────────
# Script lives in src/. BASE_DIR = v11/
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

DATA_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
GT_PATH   = os.path.join(BASE_DIR, "research_comp", "evidence_base",
                         "outbreak_events", "sangli_gt_v2.csv")

BATCH_SIZE = 512

# ANSI colours (disabled automatically on Windows if not supported)
try:
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(
        ctypes.windll.kernel32.GetStdHandle(-11), 7
    )
except Exception:
    pass

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def col(text, colour):
    return f"{colour}{text}{RESET}"


def sep(char="─", width=68):
    print(char * width)


def load_pipeline(use_temperature: bool):
    """Load metadata, scalers, temperature, and model. Return all as a dict."""
    print(col("Loading pipeline artefacts...", CYAN))

    meta_path = os.path.join(MODEL_DIR, "v11_metadata.json")
    if not os.path.exists(meta_path):
        sys.exit(col(f"ERROR: metadata not found at {meta_path}\n"
                     "Run train.py first.", RED))

    with open(meta_path) as f:
        meta = json.load(f)

    weather_features = meta["weather_features"]
    agro_features    = meta["agro_features"]
    seq_len          = int(meta["seq_len"])

    a_sc = joblib.load(os.path.join(MODEL_DIR, "agro_scaler.pkl"))

    T = 1.0
    if use_temperature:
        t_path = os.path.join(MODEL_DIR, "temperature.pkl")
        if os.path.exists(t_path):
            T = joblib.load(t_path)
            print(f"  Temperature T = {T:.4f}")
        else:
            print(col("  WARNING: temperature.pkl not found — using T=1.0", YELLOW))

    from model import KGCTCN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = KGCTCN(len(weather_features), len(agro_features)).to(device)
    ckpt   = os.path.join(MODEL_DIR, "v11_kg_ctcn.pth")
    if not os.path.exists(ckpt):
        sys.exit(col(f"ERROR: checkpoint not found at {ckpt}\n"
                     "Run train.py first.", RED))
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    print(f"  Device: {device}  |  Pipeline version: {meta.get('pipeline_version','?')}")

    return dict(
        meta=meta, weather_features=weather_features,
        agro_features=agro_features, seq_len=seq_len,
        a_sc=a_sc, T=T, model=model, device=device,
    )


def load_dataset(pipeline: dict):
    """Load CSV, apply warmup filter, assign labels. Return df."""
    df = pd.read_csv(os.path.join(DATA_DIR, "v11_features.csv"))
    df["date"] = pd.to_datetime(df["date"])

    n_before = len(df)
    df = df[df["warmup_mask"] == 0].reset_index(drop=True)
    print(f"  Warmup rows dropped: {n_before - len(df)}  |  Remaining: {len(df)}")

    if not os.path.exists(GT_PATH):
        sys.exit(col(f"ERROR: GT file not found at {GT_PATH}", RED))

    from assign_causal_labels_v2 import assign_labels
    df = assign_labels(df, gt_path=GT_PATH)
    return df


def run_inference(pipeline: dict, df: pd.DataFrame):
    """Build sequences and run batched inference. Return (dates, label_arr, all_probs)."""
    wf, af, sl = (pipeline["weather_features"],
                  pipeline["agro_features"],
                  pipeline["seq_len"])

    w_vals = df[wf].values.astype(np.float32)
    a_vals = pipeline["a_sc"].transform(df[af].values.astype(np.float32))
    labels = df["risk_label"].values.astype(np.float32)

    X_w_list, X_a_list, label_list, dates = [], [], [], []
    for i in range(sl, len(df)):
        X_w_list.append(w_vals[i - sl + 1 : i + 1])
        X_a_list.append(a_vals[i])
        label_list.append(labels[i])
        dates.append(df["date"].iloc[i])

    dates     = pd.to_datetime(dates)
    label_arr = np.array(label_list, dtype=np.float32)
    X_w_np    = np.array(X_w_list,  dtype=np.float32)
    X_a_np    = np.array(X_a_list,  dtype=np.float32)

    model, device, T = pipeline["model"], pipeline["device"], pipeline["T"]
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(X_w_np), BATCH_SIZE):
            bw = torch.FloatTensor(X_w_np[start:start+BATCH_SIZE]).to(device)
            ba = torch.FloatTensor(X_a_np[start:start+BATCH_SIZE]).to(device)
            logits, _, _ = model(bw, ba)
            probs = torch.sigmoid(logits / T)
            all_probs.extend(probs.cpu().numpy().flatten())

    return dates, label_arr, np.array(all_probs, dtype=np.float32)


def split_label(year: int, meta: dict) -> str:
    ty = meta.get("train_years", [2005, 2018])
    vy = meta.get("val_years",   [2019, 2021])
    if ty[0] <= year <= ty[1]:
        return "Train"
    if vy[0] <= year <= vy[1]:
        return "Val"
    return "Test"


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def print_score_distribution(probs: np.ndarray, label_arr: np.ndarray,
                              split_name: str, threshold: float):
    pos = probs[label_arr == 1]
    neg = probs[label_arr == 0]

    sep()
    print(col(f" SCORE DISTRIBUTION — {split_name}", BOLD))
    sep()
    print(f"  {'Metric':<28} {'Positives':>12} {'Negatives':>12}")
    print(f"  {'':─<28} {'':─>12} {'':─>12}")
    for label, arr in [("Count", None), ("Mean", None), ("Median", None),
                       ("Max", None), ("Min", None)]:
        if label == "Count":
            print(f"  {'Count':<28} {len(pos):>12d} {len(neg):>12d}")
        elif label == "Mean":
            print(f"  {'Mean score':<28} {pos.mean():>12.4f} {neg.mean():>12.4f}")
        elif label == "Median":
            print(f"  {'Median score':<28} {np.median(pos):>12.4f} {np.median(neg):>12.4f}")
        elif label == "Max":
            print(f"  {'Max score':<28} {pos.max() if len(pos) else 0:>12.4f} "
                  f"{neg.max() if len(neg) else 0:>12.4f}")
        elif label == "Min":
            print(f"  {'Min score':<28} {pos.min() if len(pos) else 0:>12.4f} "
                  f"{neg.min() if len(neg) else 0:>12.4f}")

    print(f"\n  Threshold = {threshold:.4f}")
    above_thr_pos = (pos >= threshold).sum() if len(pos) else 0
    above_thr_neg = (neg >= threshold).sum() if len(neg) else 0
    print(f"  Pos above threshold : {above_thr_pos} / {len(pos)}  "
          f"({100*above_thr_pos/max(1,len(pos)):.1f}%)")
    print(f"  Neg above threshold : {above_thr_neg} / {len(neg)}  "
          f"({100*above_thr_neg/max(1,len(neg)):.1f}%)")

    # Simple ASCII histogram
    print(f"\n  Score histogram (all samples in {split_name}):")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    bar_width = 30
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        mask = (probs >= lo) & (probs < hi)
        count = mask.sum()
        pos_c = ((probs >= lo) & (probs < hi) & (label_arr == 1)).sum()
        bar = "█" * int(count / max(len(probs), 1) * bar_width * 10)
        label_str = f"[{lo:.1f}-{hi:.1f})"
        pos_str   = col(f"+{pos_c}", GREEN) if pos_c else ""
        print(f"  {label_str:<10} {bar:<30} {count:>5}  {pos_str}")


def print_event_detection(gt_df: pd.DataFrame, scores_dict: dict,
                          threshold: float, meta: dict,
                          split_filter: str = "all"):
    sep()
    print(col(" EVENT-LEVEL DETECTION", BOLD))
    sep()
    print(f"  Detection window : [peak-14d, peak-7d]  (7–14 day lead)")
    print(f"  Threshold        : {threshold:.4f}")
    if split_filter != "all":
        print(f"  Split filter     : {split_filter}")
    print()
    print(f"  {'Peak Date':<14} {'Split':<8} {'Status':<12} {'Lead':<8} "
          f"{'Alert Date':<14} {'Score'}")
    print(f"  {'':─<14} {'':─<8} {'':─<12} {'':─<8} {'':─<14} {'':─<8}")

    counts = {"Train": [0, 0], "Val": [0, 0], "Test": [0, 0]}
    lead_times = []

    for _, row in gt_df.sort_values("peak_start").iterrows():
        peak  = row["peak_start"]
        yr    = peak.year
        split = split_label(yr, meta)
        counts[split][1] += 1

        if split_filter != "all" and split.lower() != split_filter.lower():
            continue

        window_start = peak - timedelta(days=14)
        window_end   = peak - timedelta(days=7)

        earliest_alert = None
        best_score     = 0.0
        for d in pd.date_range(window_start, window_end):
            score = scores_dict.get(d, 0.0)
            if score >= threshold:
                earliest_alert = d
                best_score     = score
                break

        if earliest_alert is not None:
            lead = (peak - earliest_alert).days
            counts[split][0] += 1
            lead_times.append(lead)
            status_str = col("DETECTED", GREEN)
            lead_str   = col(f"{lead}d", GREEN)
            alert_str  = earliest_alert.date().isoformat()
            score_str  = col(f"{best_score:.3f}", GREEN)
        else:
            # Find best score in window even if missed
            window_scores = [scores_dict.get(d, 0.0)
                             for d in pd.date_range(window_start, window_end)]
            best_miss = max(window_scores) if window_scores else 0.0
            status_str = col("MISSED", RED)
            lead_str   = col("N/A", RED)
            alert_str  = "—"
            score_str  = col(f"{best_miss:.3f} (best)", YELLOW)

        in_sample = col(" ← in-sample", YELLOW) if split == "Train" else ""
        print(f"  {peak.date().isoformat():<14} {split:<8} {status_str:<20} "
              f"{lead_str:<16} {alert_str:<14} {score_str}{in_sample}")

    sep("─")
    print(f"\n  {'Split':<8} {'Detection Rate':<20} {'Events'}")
    print(f"  {'':─<8} {'':─<20} {'':─<10}")
    for split, (det, tot) in counts.items():
        if tot == 0:
            rate_str = col("N/A (0 events)", YELLOW)
        else:
            rate = det / tot * 100
            rate_str = col(f"{rate:.1f}%  ({det}/{tot})", GREEN if rate == 100 else
                           (YELLOW if rate >= 60 else RED))
        note = col(" ← in-sample", YELLOW) if split == "Train" else ""
        print(f"  {split:<8} {rate_str}{note}")

    if lead_times:
        print(f"\n  Average lead time : {np.mean(lead_times):.1f} days")
        print(f"  Min lead time     : {min(lead_times)} days")
        print(f"  Max lead time     : {max(lead_times)} days")


def print_fp_audit(all_probs: np.ndarray, label_arr: np.ndarray,
                   threshold: float, top_n: int,
                   dates: pd.DatetimeIndex, meta: dict):
    sep()
    print(col(" FALSE POSITIVE AUDIT", BOLD))
    sep()

    preds    = (all_probs >= threshold).astype(int)
    fp_mask  = (preds == 1) & (label_arr == 0)
    tn_mask  = (preds == 0) & (label_arr == 0)
    tp_mask  = (preds == 1) & (label_arr == 1)
    fn_mask  = (preds == 0) & (label_arr == 1)

    fp = fp_mask.sum()
    tn = tn_mask.sum()
    tp = tp_mask.sum()
    fn = fn_mask.sum()

    fpr  = fp / max(1, fp + tn)
    prec = tp / max(1, tp + fp)
    rec  = tp / max(1, tp + fn)

    fpr_col  = GREEN if fpr <= 0.05 else RED
    prec_col = GREEN if prec >= 0.5 else YELLOW
    rec_col  = GREEN if rec  >= 0.8 else (YELLOW if rec >= 0.5 else RED)

    print(f"  TP = {tp:<6} FP = {fp:<6} TN = {tn:<6} FN = {fn}")
    print(f"  FPR       : {col(f'{fpr*100:.2f}%', fpr_col)}  (target < 5%)")
    print(f"  Precision : {col(f'{prec:.4f}', prec_col)}")
    print(f"  Recall    : {col(f'{rec:.4f}', rec_col)}")

    fpr_status = (col("VALIDATED — FPR within target", GREEN)
                  if fpr <= 0.05 else col("WARNING — FPR exceeds 5%", RED))
    print(f"\n  >> {fpr_status}")

    if top_n > 0 and fp > 0:
        print(f"\n  Top-{top_n} false positive days (highest score, label=0):")
        fp_scores  = all_probs[fp_mask]
        fp_dates   = dates[fp_mask]
        fp_years   = np.array([d.year for d in fp_dates])

        order      = np.argsort(fp_scores)[::-1][:top_n]
        print(f"  {'Date':<14} {'Split':<8} {'Score'}")
        print(f"  {'':─<14} {'':─<8} {'':─<8}")
        for idx in order:
            d = fp_dates[idx]
            s = split_label(fp_years[idx], meta)
            score = fp_scores[idx]
            print(f"  {str(d.date()):<14} {s:<8} {col(f'{score:.4f}', YELLOW)}")


def print_single_date(target_date_str: str, scores_dict: dict,
                      label_arr: np.ndarray, dates: pd.DatetimeIndex,
                      gt_df: pd.DataFrame, threshold: float, meta: dict):
    sep()
    print(col(f" SINGLE DATE PROBE — {target_date_str}", BOLD))
    sep()

    try:
        target = pd.to_datetime(target_date_str)
    except Exception:
        sys.exit(col(f"ERROR: Could not parse date '{target_date_str}'. "
                     "Use YYYY-MM-DD format.", RED))

    score = scores_dict.get(target, None)
    if score is None:
        print(col(f"  Date {target_date_str} not found in inference output.", RED))
        print(f"  (Date may be outside the dataset range or in warmup window)")
        return

    idx       = list(dates).index(target) if target in dates else -1
    true_lbl  = int(label_arr[idx]) if idx >= 0 else "?"
    split     = split_label(target.year, meta)
    risk_cls  = ("HIGH" if score >= 0.7 else
                 "MEDIUM" if score >= 0.3 else "LOW")
    risk_col  = (GREEN if risk_cls == "HIGH" else
                 YELLOW if risk_cls == "MEDIUM" else RESET)

    print(f"  Date        : {target_date_str}")
    print(f"  Split       : {split}")
    print(f"  Risk score  : {col(f'{score:.4f}', risk_col)}")
    print(f"  Risk class  : {col(risk_cls, risk_col)}")
    print(f"  True label  : {'POSITIVE' if true_lbl == 1 else 'NEGATIVE'}")
    print(f"  Threshold   : {threshold:.4f}")
    alert = score >= threshold
    print(f"  Alert fires : {col('YES', GREEN) if alert else col('NO', RED)}")

    # Check if this date falls in any GT event window
    print(f"\n  GT event context:")
    found_event = False
    for _, row in gt_df.iterrows():
        peak = row["peak_start"]
        w_start = peak - timedelta(days=14)
        w_end   = peak - timedelta(days=7)
        if w_start <= target <= w_end:
            lead = (peak - target).days
            print(f"    {col('IN WARNING WINDOW', GREEN)} for outbreak peak "
                  f"{peak.date().isoformat()} "
                  f"(lead = {lead} days)")
            found_event = True
    if not found_event:
        print(f"    Not in any GT event warning window")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="V11 KG-CTCN Manual Performance Check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test", "all"], default="all",
        help="Restrict event detection report to a specific split (default: all)"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Probe a specific date (YYYY-MM-DD). Shows score, label, and GT context."
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override the F2-optimal threshold (default: compute from val set)"
    )
    parser.add_argument(
        "--no-temp", action="store_true",
        help="Skip temperature scaling (use raw sigmoid outputs)"
    )
    parser.add_argument(
        "--top-fp", type=int, default=10, metavar="N",
        help="Show top-N false positive days by score (default: 10, 0 to disable)"
    )
    parser.add_argument(
        "--no-histogram", action="store_true",
        help="Skip the score distribution histogram"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    sep("═")
    print(col(" V11 KG-CTCN MANUAL PERFORMANCE CHECK", BOLD))
    sep("═")

    # ── Load pipeline ────────────────────────────────────────────────────────
    pipeline = load_pipeline(use_temperature=not args.no_temp)
    meta     = pipeline["meta"]

    # ── Load data and run inference ──────────────────────────────────────────
    print(col("\nLoading dataset and running inference...", CYAN))
    df                       = load_dataset(pipeline)
    dates, label_arr, all_probs = run_inference(pipeline, df)
    scores_dict              = dict(zip(dates, all_probs))
    gt_df                    = pd.read_csv(GT_PATH)
    gt_df["peak_start"]      = pd.to_datetime(gt_df["peak_start"])

    # ── Threshold ────────────────────────────────────────────────────────────
    if args.threshold is not None:
        threshold = args.threshold
        print(f"\n  Using manual threshold: {threshold:.4f}")
    else:
        # Compute F2-optimal threshold on val set
        from sklearn.metrics import precision_recall_curve
        val_mask  = np.array([2019 <= d.year <= 2021 for d in dates])
        v_probs   = all_probs[val_mask]
        v_labels  = label_arr[val_mask]
        if v_labels.sum() > 0:
            prec, rec, thr = precision_recall_curve(v_labels, v_probs)
            f2 = (5 * prec * rec) / (4 * prec + rec + 1e-8)
            threshold = float(thr[np.argmax(f2)])
            print(f"\n  F2-optimal threshold (val 2019-2021): {threshold:.4f}")
        else:
            threshold = 0.5
            print(col("\n  WARNING: No val positives found — using threshold=0.5", YELLOW))

    # ── Single date probe ─────────────────────────────────────────────────────
    if args.date:
        print_single_date(args.date, scores_dict, label_arr, dates,
                          gt_df, threshold, meta)
        sep("═")
        return

    # ── Score distribution ───────────────────────────────────────────────────
    if not args.no_histogram:
        # Determine which subset to show distribution for
        if args.split == "all":
            dist_probs  = all_probs
            dist_labels = label_arr
            dist_name   = "ALL SPLITS"
        else:
            year_ranges = {
                "train": meta.get("train_years", [2005, 2018]),
                "val":   meta.get("val_years",   [2019, 2021]),
                "test":  meta.get("test_years",  [2022, 2024]),
            }
            yr = year_ranges[args.split]
            mask = np.array([yr[0] <= d.year <= yr[1] for d in dates])
            dist_probs  = all_probs[mask]
            dist_labels = label_arr[mask]
            dist_name   = args.split.upper()

        print_score_distribution(dist_probs, dist_labels, dist_name, threshold)

    # ── Event detection ───────────────────────────────────────────────────────
    print_event_detection(gt_df, scores_dict, threshold, meta, args.split)

    # ── FP audit ─────────────────────────────────────────────────────────────
    print_fp_audit(all_probs, label_arr, threshold, args.top_fp, dates, meta)

    sep("═")
    print(col(" CHECK COMPLETE", BOLD))
    sep("═")


if __name__ == "__main__":
    main()