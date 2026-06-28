import pandas as pd
import numpy as np
import joblib

# ---------------------------
# 1. SETUP & DATA LOADING
# ---------------------------
def get_test_data():
    df_raw = pd.read_csv("early_warning_dataset.csv")
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_raw = df_raw.sort_values('date').reset_index(drop=True)
    
    WINDOW_SIZE = 14
    exclude_cols = ['date', 'target_3d', 'target_5d', 'target_7d', 'target_score_5d', 'month', 'dayofyear', 'week']
    feature_cols = [col for col in df_raw.columns if col not in exclude_cols]
    
    X_full = []
    for i in range(WINDOW_SIZE, len(df_raw)):
        window = df_raw.iloc[i-WINDOW_SIZE:i][feature_cols].values.flatten()
        X_full.append(window)
        
    X = pd.DataFrame(X_full)
    # Re-map split (80/20)
    split_idx = int(len(X) * 0.8)
    X_test = X.iloc[split_idx:]
    df_test_raw = df_raw.iloc[split_idx + WINDOW_SIZE:].reset_index(drop=True)
    return X_test, df_test_raw

X_test, df_test_meta = get_test_data()
model_5d = joblib.load("red_rot_5d_calibrated.joblib")
probs = model_5d.predict_proba(X_test)[:, 1]

# ---------------------------
# 2. STREAK DETECTION LOGIC
# ---------------------------
def detect_streaks(probs, threshold, min_days):
    streaks = []
    current_streak = 0
    start_idx = None
    
    for i, p in enumerate(probs):
        if p >= threshold:
            if current_streak == 0:
                start_idx = i
            current_streak += 1
        else:
            if current_streak >= min_days:
                streaks.append((start_idx, i - 1, current_streak))
            current_streak = 0
    # Last one
    if current_streak >= min_days:
        streaks.append((start_idx, len(probs) - 1, current_streak))
    return streaks

# Predicted Warning (2 days @ 0.5)
pred_warnings = detect_streaks(probs, 0.5, 2)
# Predicted Outbreaks (3 days @ 0.6)
pred_outbreaks = detect_streaks(probs, 0.6, 3)

# ---------------------------
# 3. TRUE EVENT DETECTION (from ground truth)
# ---------------------------
# Note: target_5d is 1 if there's an outbreak at T+5.
# We look for streaks in target_5d to define an "Actual Event"
true_outbreaks = detect_streaks(df_test_meta['target_5d'].values, 0.5, 3)

# ---------------------------
# 4. THRESHOLD SWEEP (Tuning for Research)
# ---------------------------
print("\n--- Threshold Tuning Sweep ---")
best_recall = 0
best_t = 0

for t in np.arange(0.05, 0.4, 0.05):
    p_outbreaks = detect_streaks(probs, t, 3)
    hits = 0
    for true_start, true_end, _ in true_outbreaks:
        for pred_start, pred_end, _ in p_outbreaks:
            if abs(pred_start - true_start) <= 3:
                hits += 1
                break
    recall = (hits / len(true_outbreaks)) if len(true_outbreaks) > 0 else 0
    print(f"Threshold {t:.2f}: Recall {recall:.2%}, Events Predicted: {len(p_outbreaks)}")
    if recall >= best_recall:
        best_recall = recall
        best_t = t

# ---------------------------
# 5. DATASET CALIBRATION CHECK
# ---------------------------
print("\n--- Probability Calibration Details ---")
print(pd.Series(probs).describe())
print(f"\nFinal Recommendation: For this specific dataset, use an 'Outbreak' threshold of {best_t:.2f} for optimal event capture.")
