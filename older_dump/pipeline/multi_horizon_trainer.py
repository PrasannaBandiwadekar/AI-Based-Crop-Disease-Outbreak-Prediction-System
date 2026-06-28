import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
import joblib
import os

# ---------------------------
# 1. CONFIGURATION
# ---------------------------
INPUT_FILE = "sequence_based_dataset.csv"
TARGETS = ["target_3d", "target_5d", "target_7d"]
WINDOW_SIZE = 14

# We need to load the original targets from the unreshaped data to map multi-horizon correctly
# But wait, reshape_sequences.py used'target_5d' from the unreshaped data as 'target'.
# For multi-horizon, we need to create a new sequenced dataset that keeps all targets or re-run reshaping.
# Let's re-load the unreshaped data and create a multi-horizon sequence mapping.

def get_multi_horizon_data():
    df_raw = pd.read_csv("early_warning_dataset.csv")
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_raw = df_raw.sort_values('date').reset_index(drop=True)
    
    exclude_cols = ['date', 'target_3d', 'target_5d', 'target_7d', 'target_score_5d', 'month', 'dayofyear', 'week']
    feature_cols = [col for col in df_raw.columns if col not in exclude_cols]
    
    sequences = []
    y_3d, y_5d, y_7d = [], [], []
    dates = []
    
    for i in range(WINDOW_SIZE, len(df_raw)):
        window = df_raw.iloc[i-WINDOW_SIZE:i][feature_cols].values.flatten()
        sequences.append(window)
        y_3d.append(df_raw.iloc[i-1]['target_3d'])
        y_5d.append(df_raw.iloc[i-1]['target_5d'])
        y_7d.append(df_raw.iloc[i-1]['target_7d'])
        dates.append(df_raw.iloc[i-1]['date'])
        
    X = pd.DataFrame(sequences)
    # Generate columns names
    col_names = []
    for day in range(1, WINDOW_SIZE + 1):
        for feat in feature_cols:
            col_names.append(f"{feat}_d{day}")
    X.columns = col_names
    
    return X, pd.Series(y_3d), pd.Series(y_5d), pd.Series(y_7d), pd.Series(dates)

# ---------------------------
# 2. TRAINING PIPELINE
# ---------------------------
X, y3, y5, y7, dates = get_multi_horizon_data()
split_idx = int(len(X) * 0.8)

horizons = {"3d": y3, "5d": y5, "7d": y7}
models = {}

print(f"Starting Multi-Horizon Training (Calibrated RF)...")

for h_name, y in horizons.items():
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\n--- Training {h_name} Horizon ---")
    
    # Base RF
    base_rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    
    # Calibrated Wrapper (Isotonic for better probability scaling)
    calibrated_model = CalibratedClassifierCV(base_rf, method='isotonic', cv=3)
    calibrated_model.fit(X_train, y_train)
    
    # Eval
    probs = calibrated_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Brier Score (Calibration): {brier:.4f}")
    
    # Save
    model_path = f"red_rot_{h_name}_calibrated.joblib"
    joblib.dump(calibrated_model, model_path)
    models[h_name] = calibrated_model

# ---------------------------
# 3. FEATURE GROUP ANALYSIS (on 5d model)
# ---------------------------
# Note: CalibratedClassifierCV doesn't have direct feature_importances_. 
# We look at the importances of the underlying base models in the CV folds.
print("\n--- Feature Group Importance (5d Horizon) ---")
final_5d = models["5d"]
# Averaging importances across the CV folds
all_importances = []
for fold_model in final_5d.calibrated_classifiers_:
    # In newer scikit-learn, it's .estimator
    if hasattr(fold_model, 'estimator'):
        all_importances.append(fold_model.estimator.feature_importances_)
    else:
        all_importances.append(fold_model.base_estimator.feature_importances_)

avg_importances = np.mean(all_importances, axis=0)
feat_series = pd.Series(avg_importances, index=X.columns)

groups = {
    "Short-term (3d)": [c for c in X.columns if "3d" in c],
    "Mid-term (7d)": [c for c in X.columns if "7d" in c],
    "Long-term (14d)": [c for c in X.columns if "14d" in c],
    "Biological Streaks": [c for c in X.columns if "streak" in c],
    "Activation Triggers": [c for c in X.columns if "trigger" in c or "composite" in c]
}

for g, cols in groups.items():
    if cols:
        print(f"{g:20} Mean Importance: {feat_series[cols].mean():.6f}")

print("\nAll models trained and calibrated successfully.")
