import pandas as pd
import os

# Adjust path based on execution location
file_path = "v11/data/processed/v11_features.csv"
if not os.path.exists(file_path):
    file_path = "data/processed/v11_features.csv" # If run from within v11

df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])
df = df[df['warmup_mask'] == 0]

# Show all positive-labelled rows with their dates
pos = df[df['risk_label'] == 1][['date', 'risk_label', 'RH_persist_7d', 'Rain_sum_14d']]
print(pos.to_string())
print(f"\nTotal positives: {len(pos)}")
if not pos.empty:
    print(f"Date range: {pos['date'].min()} → {pos['date'].max()}")
else:
    print("No positives found.")
