"""
V11 KG-CTCN Pipeline Diagnostic: Deep Signal Audit
--------------------------------------------------
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference_engine import V11InferenceEngine

def deep_audit(location, date_str):
    print(f"\nDEEP AUDIT: {location} on {date_str}")
    engine = V11InferenceEngine()
    target_date = pd.to_datetime(date_str)
    
    # Fetch and Transform
    pw = engine._fetch_past_weather_slice(location, target_date)
    tr = engine._apply_kg_transforms(pw)
    w_slice = tr.tail(28)
    
    # Check Weather Features before scaling
    feat_values = w_slice[engine.weather_features].values
    print(f"Raw Features Tail (Last Step):\n{feat_values[-1]}")
    
    # Scaling
    w_sc_data = engine.w_sc.transform(feat_values)
    print(f"Scaled Features Tail (Last Step):\n{w_sc_data[-1]}")
    
    # Model
    X_w = torch.FloatTensor(w_sc_data).unsqueeze(0).to(engine.device)
    X_a = torch.FloatTensor(np.zeros((1, 3))).to(engine.device) # dummy agro
    
    with torch.no_grad():
        logits, prob, _ = engine.model(X_w, X_a)
        
    print(f"Logits: {logits.item()} | Prob: {prob.item()}")

if __name__ == "__main__":
    deep_audit("Kolhapur", "2021-08-15")
