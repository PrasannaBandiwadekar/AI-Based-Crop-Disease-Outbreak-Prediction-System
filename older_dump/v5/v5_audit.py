import pandas as pd
import numpy as np
import joblib
import sys
import os

sys.path.append("e:/crop-disease-outbreak-prediction-system-feature-zip-changes/crop-disease-outbreak-prediction-system-feature-zip-changes/model_train/v5")
from src import utils, features, config

# Load Data
raw_df = utils.load_raw_data()
df = features.build_features(raw_df)
df = features.add_label(df)
df = df.iloc[28:-7].dropna(subset=config.ENGINEERED_FEATURES + [config.TARGET])

# Validate set
val_mask = (df["date"].dt.year >= config.VAL_YEARS[0]) & (df["date"].dt.year <= config.VAL_YEARS[1])
X_val = df.loc[val_mask, config.ENGINEERED_FEATURES]
y_val = df.loc[val_mask, config.TARGET]

# Load model (HistGradientBoosting artifact)
model = joblib.load(config.MODEL_PATH.replace(".json", ".pkl"))

# Predict
probs = model.predict_proba(X_val)[:, 1]

print("=== Distribution ===")
print(f"Min: {probs.min():.4f}")
print(f"Max: {probs.max():.4f}")
print(f"Mean: {probs.mean():.4f}")
print(f"> 0.1: {(probs > 0.1).mean()*100:.2f}%")
print(f"> 0.5: {(probs > 0.5).mean()*100:.2f}%")
print(f"> 0.9: {(probs > 0.9).mean()*100:.2f}%")

print("\n=== Threshold Severity ===")
print(f"85th Percentile: {np.percentile(probs, 85):.4f}")
print(f"90th Percentile: {np.percentile(probs, 90):.4f}")
print(f"95th Percentile: {np.percentile(probs, 95):.4f}")

# Simulate Recall
def get_metrics(threshold):
    preds = (probs >= threshold).astype(int)
    recall = (preds & y_val).sum() / y_val.sum() if y_val.sum() > 0 else 0
    fpr = (preds & (y_val == 0)).sum() / (y_val == 0).sum() if (y_val == 0).sum() > 0 else 0
    return recall, fpr

for t in [0.9847, 0.9, 0.5, 0.3, 0.2, 0.1]:
    r, f = get_metrics(t)
    print(f"Threshold {t:.4f} -> Recall: {r:.4f}, FPR: {f:.4f}")

