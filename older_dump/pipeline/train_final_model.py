import pandas as pd
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# ---------------------------
# 1. LOAD DATA
# ---------------------------
print("📦 Loading optimized dataset...")
df = pd.read_csv("optimized_early_warning_dataset.csv")

TARGET = 'target_5d'
drop_cols = ['date', 'target_3d', 'target_5d', 'target_7d', 'target_score_5d']
features = [col for col in df.columns if col not in drop_cols]

X = df[features]
y = df[TARGET]

# Split for final internal validation check
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# ---------------------------
# 2. TRAIN PRODUCTION MODEL
# ---------------------------
print(f"🚀 Training production model on {len(features)} optimized features...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Quick validation
y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print(f"✅ Final Validation ROC-AUC: {auc:.4f}")

# ---------------------------
# 3. EXPORT MODEL & METADATA
# ---------------------------
print("💾 Saving model to 'red_rot_forecast_v1.joblib'...")
joblib.dump(model, "red_rot_forecast_v1.joblib")

metadata = {
    "model_name": "Sugarcane Red Rot Forecast - Optimized",
    "version": "1.0",
    "lead_time_days": 5,
    "features": features,
    "validation_auc": auc
}

print("📄 Saving metadata to 'model_metadata.json'...")
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("\n✨ Final model is ready for production!")
