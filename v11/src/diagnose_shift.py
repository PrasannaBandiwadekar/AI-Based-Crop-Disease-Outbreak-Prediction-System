import pandas as pd
import numpy as np
import os
import sys

# Add src to path to import assign_labels
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

from assign_causal_labels_v2 import assign_labels

df_path = os.path.join(BASE_DIR, "data", "processed", "v11_features.csv")
df = pd.read_csv(df_path)
df["date"] = pd.to_datetime(df["date"])
df = df[df["warmup_mask"] == 0].reset_index(drop=True)

# Compare KG feature distributions across splits
kg_features = ["RH_persist_7d", "Rain_sum_14d", "RH_high_flag", "T2M_MIN_lag_15d"]

train = df[df["date"].dt.year.between(2005, 2014)]
val   = df[df["date"].dt.year.between(2015, 2018)]
test  = df[df["date"].dt.year.between(2019, 2021)]

print(f"{'Feature':<22} {'Train mean':>12} {'Val mean':>10} {'Test mean':>10}")
print("-" * 56)
for f in kg_features:
    print(f"{f:<22} {train[f].mean():>12.4f} {val[f].mean():>10.4f} {test[f].mean():>10.4f}")

# More importantly — what do positive-day features look like vs negative-day features in test?
GT_PATH = os.path.join(BASE_DIR, "research_comp", "evidence_base", "outbreak_events", "sangli_gt_v2.csv")
df = assign_labels(df, gt_path=GT_PATH)

test_df = df[df["date"].dt.year.between(2019, 2021)]
pos = test_df[test_df["risk_label"] == 1]
neg = test_df[test_df["risk_label"] == 0]

print(f"\nTest set positive vs negative day KG features:")
print(f"{'Feature':<22} {'Pos mean':>10} {'Neg mean':>10} {'Separation':>12}")
print("-" * 56)
for f in kg_features:
    p_mean = pos[f].mean()
    n_mean = neg[f].mean()
    # Separation = difference in means / standard deviation
    sep = (p_mean - n_mean) / (test_df[f].std() + 1e-8)
    print(f"{f:<22} {p_mean:>10.4f} {n_mean:>10.4f} {sep:>12.4f}")
