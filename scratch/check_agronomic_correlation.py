
import pandas as pd
import numpy as np
import os

BASE_DIR = r"e:\crop-disease-outbreak-prediction-system-feature-zip-changes\crop-disease-outbreak-prediction-system-feature-zip-changes\model_train\v11"
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
GT_PATH = os.path.join(BASE_DIR, "research_comp", "evidence_base", "outbreak_events", "sangli_gt_v2.csv")

# We need the labeling logic
import sys
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
from assign_causal_labels_v2 import assign_labels

df = pd.read_csv(os.path.join(DATA_DIR, "v11_features.csv"))
df["date"] = pd.to_datetime(df["date"])
df = df[df["warmup_mask"] == 0].reset_index(drop=True)

df = assign_labels(df, GT_PATH)

print("\n--- Correlation Audit ---")
print(df.groupby("variety_susceptibility")["risk_label"].mean())
print("\n--- Ratoon Audit ---")
print(df.groupby("is_ratoon")["risk_label"].mean())

# Check mapping:
# 0: Resistant
# 1: Moderate
# 2: Susceptible

# Expected: mean(risk_label) for 2 > 1 > 0
# Actually, 0 is suppressed to 0 (except GT overrides).
