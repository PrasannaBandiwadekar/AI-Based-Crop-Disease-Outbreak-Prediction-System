"""
V11 KG-CTCN Pipeline Diagnostic: Kolhapur & Pune Audit
--------------------------------------------------
Performs a deep signal audit on non-Sangli regions to detect latent 
distribution shifts or coordinate mismatches.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference_engine import V11InferenceEngine

def audit_region(location, date_str):
    print(f"\nAUDITING: {location} on {date_str}")
    engine = V11InferenceEngine()
    
    target_date = pd.to_datetime(date_str)
    agro_inputs = {"variety_susceptibility": 3, "is_ratoon": 0, "crop_age_days": 150}
    
    # Run inference and capture the internal state
    # We will manually fetch and transform to see the layers
    
    # 1. Fetch
    past_weather = engine._fetch_past_weather_slice(location, target_date)
    print(f"Layer 1 (Input): {location} RH2M mean: {past_weather['RH2M'].mean():.2f}%")
    
    # 2. Transform
    transformed = engine._apply_kg_transforms(past_weather)
    w_slice = transformed.tail(28)
    rh_persist_var = w_slice['RH_persist_7d'].var()
    print(f"Layer 2 (Feature): RH_persist_7d Variance: {rh_persist_var:.4f}")
    
    # 3. Scaling & Model
    out = engine.run_inference(location, target_date, agro_inputs)
    print(f"Layer 3 (Model): Logits: {out.get('logits', 'N/A')} | Risk: {out['risk_score']:.4f}")
    
    if abs(out.get('logits', 0)) > 10:
        print("!!! ALERT: Signal Saturation Detected !!!")

if __name__ == "__main__":
    # Peak monsoon dates
    audit_region("Kolhapur", "2021-08-15")
    audit_region("Pune", "2021-08-15")
