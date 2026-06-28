import pandas as pd
import numpy as np

# ---------------------------
# 1. CONFIGURATION
# ---------------------------
INPUT_FILE = "early_warning_dataset.csv"
OUTPUT_FILE = "sequence_based_dataset.csv"
WINDOW_SIZE = 14  # 14 days of history
TARGET_COL = "target_5d"

# ---------------------------
# 2. LOAD DATA
# ---------------------------
df = pd.read_csv(INPUT_FILE)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Define features to include in the sequence
# We include weather, streaks, and engineered features.
# We exclude date and other targets.
exclude_cols = [
    'date', 'target_3d', 'target_5d', 'target_7d', 'target_score_5d', 
    'month', 'dayofyear', 'week' # These are captured by sin/cos
]
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"Reshaping with {len(feature_cols)} features per day.")
print(f"Features: {feature_cols}")

# ---------------------------
# 3. SLIDING WINDOW GENERATION
# ---------------------------
sequences = []
targets = []
dates = []

# Iterating to create windows
# We need WINDOW_SIZE consecutive days to make a sequence
for i in range(WINDOW_SIZE, len(df)):
    # Slice the last WINDOW_SIZE days
    window = df.iloc[i-WINDOW_SIZE:i][feature_cols].values
    
    # Target is from the CURRENT row (which already shifted 5 days forward in forecast_dataset.py)
    # Wait, forecast_dataset.py already did the shift:
    # df['target_5d'] = df['red_rot_risk'].shift(-5)
    # This means at index 'i', the target_5d column contains the risk for (i+5)
    
    target = df.iloc[i-1][TARGET_COL] # Target for the last day of the window
    
    # Flatten the window (14 days * N features)
    flattened_window = window.flatten()
    
    sequences.append(flattened_window)
    targets.append(target)
    dates.append(df.iloc[i-1]['date'])

# ---------------------------
# 4. CREATE SEQUENCED DATAFRAME
# ---------------------------
# Column names: feature1_day1, feature2_day1, ..., featureN_day14
new_cols = []
for day in range(1, WINDOW_SIZE + 1):
    for feat in feature_cols:
        new_cols.append(f"{feat}_d{day}")

seq_df = pd.DataFrame(sequences, columns=new_cols)
seq_df['target'] = targets
seq_df['prediction_date'] = dates

# ---------------------------
# 5. SAVE
# ---------------------------
seq_df.to_csv(OUTPUT_FILE, index=False)
print(f"Done! Created sequence dataset with {seq_df.shape[0]} samples.")
print(f"Input features per sample: {seq_df.shape[1] - 2}")
print(f"Saved as {OUTPUT_FILE}")
