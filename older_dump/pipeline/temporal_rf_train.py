import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

# ---------------------------
# 1. LOAD DATA
# ---------------------------
df = pd.read_csv("sequence_based_dataset.csv")

# ---------------------------
# 2. TRAIN/TEST SPLIT (Time-based)
# ---------------------------
# We must split by time, not randomly!
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

# Define X and y
exclude_cols = ['target', 'prediction_date']
X_cols = [col for col in df.columns if col not in exclude_cols]

X_train, y_train = train_df[X_cols], train_df['target']
X_test, y_test = test_df[X_cols], test_df['target']

print(f"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features.")
print(f"Testing on {X_test.shape[0]} samples.")

# ---------------------------
# 3. TRAIN TEMPORAL RANDOM FOREST
# ---------------------------
# Using balanced class weights since outbreaks (target=1) might be rarer
rf = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    class_weight='balanced',
    n_jobs=-1
)
rf.fit(X_train, y_train)

# ---------------------------
# 4. EVALUATION
# ---------------------------
probs = rf.predict_proba(X_test)[:, 1]
preds = rf.predict(X_test)

auc = roc_auc_score(y_test, probs)
report = classification_report(y_test, preds)

print("\n--- Model Performance (Temporal RF) ---")
print(f"ROC-AUC: {auc:.4f}")
print("\nClassification Report:")
print(report)

# ---------------------------
# 5. FEATURE IMPORTANCE (Red Rot Insight)
# ---------------------------
importances = pd.Series(rf.feature_importances_, index=X_cols)
top_20 = importances.sort_values(ascending=False).head(20)

print("\n--- Top 20 Predictors (Bio-Temporal Insight) ---")
print(top_20)

# Check specifically for our streak features
streak_feats = [c for c in X_cols if 'streak' in c or 'trigger' in c or 'composite' in c]
streak_imps = importances[streak_feats].sort_values(ascending=False)
print("\n--- Biological Feature Contribution ---")
print(streak_imps.head(10))

# ---------------------------
# 6. SAVE MODEL
# ---------------------------
model_filename = "red_rot_temporal_rf.joblib"
joblib.dump(rf, model_filename)
print(f"\nModel saved as {model_filename}")
