import os
import sys
import pandas as pd

# Add src to path to import dataset_pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_pipeline import (
    load_and_clean_data,
    apply_rolling_zscore,
    engineer_kg_features,
    engineer_agronomic_features,
    WARMUP_DAYS
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTHETIC_RAW = os.path.join(BASE_DIR, "data", "synthetic", "v11_synthetic_raw.csv")
SYNTHETIC_FEATURES_OUT = os.path.join(BASE_DIR, "data", "synthetic", "v11_synthetic_features.csv")

def process_synthetic():
    print(f"Processing synthetic raw data: {SYNTHETIC_RAW}")
    
    # 1. Load and clean
    df = load_and_clean_data(SYNTHETIC_RAW)
    
    # 2. Apply rolling Z-score
    df = apply_rolling_zscore(df)
    
    # 3. Engineer KG features
    df = engineer_kg_features(df)
    
    # 4. Engineer agronomic features
    df = engineer_agronomic_features(df)
    
    # Save
    df.to_csv(SYNTHETIC_FEATURES_OUT, index=False)
    print(f"Saved synthetic features: {SYNTHETIC_FEATURES_OUT}")
    
    # Comparison with original features.csv (for feature parity check)
    ORIGINAL_FEATURES = os.path.join(BASE_DIR, "data", "processed", "v11_features.csv")
    if os.path.exists(ORIGINAL_FEATURES):
        orig_df = pd.read_csv(ORIGINAL_FEATURES, nrows=1)
        synth_df = pd.read_csv(SYNTHETIC_FEATURES_OUT, nrows=1)
        
        orig_cols = set(orig_df.columns)
        synth_cols = set(synth_df.columns)
        
        missing = orig_cols - synth_cols
        extra = synth_cols - orig_cols
        
        print("\n--- Feature Parity Check ---")
        if not missing and not extra:
            print("[PASS] Columns match exactly.")
        else:
            if missing: print(f"[FAIL] Missing columns: {missing}")
            if extra: print(f"[FAIL] Extra columns: {extra}")
    else:
        print(f"\n[WARN] Original features.csv not found at {ORIGINAL_FEATURES}. Skipping parity check.")

if __name__ == "__main__":
    process_synthetic()
