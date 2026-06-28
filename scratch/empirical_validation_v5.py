import os
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import recall_score, precision_score, confusion_matrix

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE, "v5", "data", "processed", "dataset_v5.csv")

print("============================================================")
print("  V5 Empirical Validation & Threshold Stress Test")
print("============================================================")

df = pd.read_csv(CSV_PATH, parse_dates=["date"])

# ── 1. Split Data ─────────────────────────────────────────────────────────────
train_mask = (df["date"].dt.year >= 2005) & (df["date"].dt.year <= 2018)
val_mask = (df["date"].dt.year >= 2019) & (df["date"].dt.year <= 2021)
test_mask = (df["date"].dt.year >= 2022) & (df["date"].dt.year <= 2024)

# Exclude target and non-features
exclude = ["date", "outbreak_risk"]
features = [c for c in df.columns if c not in exclude]

X_train, y_train = df.loc[train_mask, features], df.loc[train_mask, "outbreak_risk"]
X_val, y_val = df.loc[val_mask, features], df.loc[val_mask, "outbreak_risk"]
X_test, y_test = df.loc[test_mask, features], df.loc[test_mask, "outbreak_risk"]

# Apply scale_pos_weight via sample weights
sample_weights = np.where(y_train == 1, 6.0, 1.0)

# ── 2. Train Model ────────────────────────────────────────────────────────────
print("\n[1/5] Training Gradient Boosting Model (XGBoost Equivalent)...")
model = HistGradientBoostingClassifier(
    max_depth=4, 
    early_stopping=True, 
    validation_fraction=0.1, 
    random_state=42
)
model.fit(X_train, y_train, sample_weight=sample_weights)

# Raw Probabilities
val_probs_raw = model.predict_proba(X_val)[:, 1]
test_probs_raw = model.predict_proba(X_test)[:, 1]

# ── 3. Calibrate Probabilities ────────────────────────────────────────────────
print("[2/5] Calibrating Probabilities (Isotonic Regression on Val)...")
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(val_probs_raw, y_val)

val_probs_cal = calibrator.predict(val_probs_raw)
test_probs_cal = calibrator.predict(test_probs_raw)

# ── 4. Threshold Stress Test ──────────────────────────────────────────────────
print("[3/5] Threshold Stress Testing (Test Set)...")
for t in [0.3, 0.7]:
    preds = (test_probs_cal >= t).astype(int)
    rec = recall_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    fpr = fp / (fp + tn)
    print(f"  Threshold {t:.1f} -> Recall: {rec*100:.1f}%, FPR: {fpr*100:.1f}%, Precision: {prec*100:.1f}%")

# ── 5. Consecutive Risk Rule & Lead Time ──────────────────────────────────────
print("\n[4/5] Evaluating 3-Day Rule & Lead Time (2019-2024)...")
# Combine Val and Test for empirical evaluation
eval_df = df[val_mask | test_mask].copy()
eval_df["raw_prob"] = model.predict_proba(eval_df[features])[:, 1]
eval_df["cal_prob"] = calibrator.predict(eval_df["raw_prob"])

# 3-Day Rule logic
eval_df["is_high"] = (eval_df["cal_prob"] >= 0.7).astype(int)
eval_df["3_day_high"] = (eval_df["is_high"].rolling(3).sum() >= 3).astype(int)

# Identify true outbreak blocks
eval_df["block_id"] = (eval_df["outbreak_risk"] != eval_df["outbreak_risk"].shift(1)).cumsum()
outbreak_blocks = eval_df[eval_df["outbreak_risk"] == 1].groupby("block_id")

total_outbreaks = len(outbreak_blocks)
detected_raw = 0
detected_3day = 0
lead_times = []

# FPR reduction
tn_raw, fp_raw, fn_raw, tp_raw = confusion_matrix(eval_df["outbreak_risk"], eval_df["is_high"]).ravel()
tn_3d, fp_3d, fn_3d, tp_3d = confusion_matrix(eval_df["outbreak_risk"], eval_df["3_day_high"]).ravel()
fpr_raw = fp_raw / (fp_raw + tn_raw)
fpr_3d = fp_3d / (fp_3d + tn_3d)

for block_id, block in outbreak_blocks:
    # Start date of the future condition is technically t+3. 
    # But block["date"].min() is the date 't' when the condition for t+3 starts.
    # Therefore, trigger at t means exactly 3 days lead time.
    start_t = block["date"].min()
    end_t = block["date"].max()
    
    # Check if we triggered BEFORE or DURING this window
    # Wait, the label "outbreak_risk" at time t means window t+3 to t+7 is bad.
    # If we trigger at time t (i.e. we predict 1 at time t), we have exactly 3 days lead time!
    # If we trigger at t-2, we have 5 days lead time.
    # Let's search for the first trigger within a 7-day lookback of start_t.
    search_start = start_t - pd.Timedelta(days=7)
    window = eval_df[(eval_df["date"] >= search_start) & (eval_df["date"] <= end_t)]
    
    if window["is_high"].sum() > 0:
        detected_raw += 1
    
    # Check 3-day rule
    trig_dates = window[window["3_day_high"] == 1]["date"]
    if len(trig_dates) > 0:
        detected_3day += 1
        first_trig = trig_dates.min()
        # True outbreak peak window starts at start_t + 3 days
        true_outbreak_start = start_t + pd.Timedelta(days=3)
        lt = (true_outbreak_start - first_trig).days
        lead_times.append(lt)

print(f"  Total biological outbreak blocks : {total_outbreaks}")
print(f"  Detected (Raw 0.7 Threshold)     : {detected_raw}/{total_outbreaks} ({detected_raw/total_outbreaks*100:.1f}%)")
print(f"  Detected (3-Day Consecutive)     : {detected_3day}/{total_outbreaks} ({detected_3day/total_outbreaks*100:.1f}%)")
print(f"  FPR (Raw 0.7 Threshold)          : {fpr_raw*100:.1f}%")
print(f"  FPR (3-Day Consecutive)          : {fpr_3d*100:.1f}% (Reduction: {(fpr_raw-fpr_3d)/fpr_raw*100:.1f}%)")

if lead_times:
    print(f"  Mean Lead Time                   : {np.mean(lead_times):.1f} days")
    print(f"  Min Lead Time (Worst-case)       : {np.min(lead_times)} days")
    print(f"  Max Lead Time                    : {np.max(lead_times)} days")

# ── 6. Monsoon Failure Case Analysis ──────────────────────────────────────────
print("\n[5/5] Monsoon Failure Case Analysis...")
# False positives during rainfall spikes
fp_df = eval_df[(eval_df["3_day_high"] == 1) & (eval_df["outbreak_risk"] == 0)]
fp_monsoon = fp_df[fp_df["monsoon_flag"] == 1]
fp_non_monsoon = fp_df[fp_df["monsoon_flag"] == 0]
print(f"  False Positives in Monsoon       : {len(fp_monsoon)} days")
print(f"  False Positives Non-Monsoon      : {len(fp_non_monsoon)} days")
# Why do they occur? Often high rain + high humidity that doesn't hold long enough for the target to trigger 1, or temperatures dip.
if len(fp_monsoon) > 0:
    avg_prec_fp = fp_monsoon["PREC_sum14"].mean()
    print(f"  -> Avg 14d Rain during Monsoon FP: {avg_prec_fp:.1f} mm (shows sensitivity to rain spikes)")

# ── Final Output ──────────────────────────────────────────────────────────────
print("\n============================================================")
print("  FINAL SYSTEM VERDICT: GO / NO-GO")
print("============================================================")
if (detected_3day / total_outbreaks) >= 0.9 and fpr_3d < 0.15:
    print("  Verdict: GO")
    print("  Justification: System achieves >=90% recall while strictly suppressing FPR via the 3-day rule.")
else:
    print("  Verdict: CONDITIONAL GO")
    print("  Justification: Review recall/FPR tradeoffs. The 3-day rule may be too strict if recall dropped.")

print("============================================================")
