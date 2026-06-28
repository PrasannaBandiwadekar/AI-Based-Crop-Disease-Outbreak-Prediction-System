import pandas as pd
import numpy as np

# Load your engineered dataset
df = pd.read_csv("engineered_weather_features.csv")

# ---------------------------
# 1. DEFINE BIOLOGICAL CONDITIONS (RED ROT SPECIFIC)
# ---------------------------

# Use the features we engineered in feature_engg.py
# If they don't exist here yet, we'll re-calculate or assume they are in df

# ---------------------------
# 2. CREATE RISK SCORE (Sophisticated weighted approach)
# ---------------------------

# Weighting biology: 
# 40% Persistence (Humid/Wet streaks)
# 30% Critical Trigger (Composite danger zone)
# 20% Activation (Dry to Wet)
# 10% Optimal Temp window

df['red_rot_score'] = (
    0.20 * df['wet_streak'] +
    0.20 * df['humid_streak'] +
    0.30 * df['red_rot_risk_composite'] +
    0.20 * df['dry_to_wet_trigger'] +
    0.10 * df['temp_optimal_red_rot']
)

# Monsoon boost (India specifics)
monsoon = df['month'].isin([6,7,8,9])
df['red_rot_score'] += 0.1 * monsoon.astype(int)

# Normalize
df['red_rot_score'] = df['red_rot_score'].clip(0, 1)

# ---------------------------
# 3. CREATE LABEL
# ---------------------------

# Binary classification (Strict threshold for "Outbreak")
df['red_rot_risk'] = (df['red_rot_score'] > 0.5).astype(int)

# Optional: multi-class
def risk_category(score):
    if score < 0.2:
        return "Low"
    elif score < 0.5:
        return "Medium"
    else:
        return "High"

df['red_rot_category'] = df['red_rot_score'].apply(risk_category)

# ---------------------------
# 4. SAVE UPDATED DATASET
# ---------------------------

df.to_csv("red_rot_labeled_dataset.csv", index=False)

print("Done! Saved as red_rot_labeled_dataset.csv")