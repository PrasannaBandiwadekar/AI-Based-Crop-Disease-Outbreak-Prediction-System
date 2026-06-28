import pandas as pd

# ---------------------------
# 1. LOAD DATA
# ---------------------------
df = pd.read_csv("red_rot_labeled_dataset.csv")

# Ensure sorted by time
df = df.sort_values('date').reset_index(drop=True)

# ---------------------------
# 2. CREATE FUTURE TARGETS
# ---------------------------

# Predict future outbreak risk
df['target_3d'] = df['red_rot_risk'].shift(-3)
df['target_5d'] = df['red_rot_risk'].shift(-5)
df['target_7d'] = df['red_rot_risk'].shift(-7)

# Optional: future continuous score
df['target_score_5d'] = df['red_rot_score'].shift(-5)

# ---------------------------
# 3. REMOVE DATA LEAKAGE
# ---------------------------

# Drop columns that represent "current outcome"
leakage_cols = [
    'red_rot_risk',
    'red_rot_score',
    'red_rot_category'
]

df = df.drop(columns=[col for col in leakage_cols if col in df.columns])

# ---------------------------
# 4. DROP NaNs (from shifting)
# ---------------------------
df = df.dropna()

# ---------------------------
# 5. FINAL CLEAN DATASET
# ---------------------------

# Optional: remove raw date (or keep if needed)
# df = df.drop(columns=['date'])

# Save dataset
df.to_csv("early_warning_dataset.csv", index=False)

print("Done! Saved as early_warning_dataset.csv")