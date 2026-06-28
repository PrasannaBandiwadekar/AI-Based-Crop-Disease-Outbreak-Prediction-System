
import os
import sys
import pandas as pd
from datetime import datetime

BASE_DIR = r"e:\crop-disease-outbreak-prediction-system-feature-zip-changes\crop-disease-outbreak-prediction-system-feature-zip-changes\model_train\v11"
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from deployment_layer import DeploymentAPI

api = DeploymentAPI()

location = "Sangli"
target_date = "2020-09-21" # A date with known high risk in training

print(f"--- Inference Test on {target_date} ---")

# Test Variety Susceptibility
for v in [0, 1, 2]:
    inputs = {"variety_susceptibility": v, "is_ratoon": 0, "crop_age_days": 180}
    res = api.predict(location, target_date, inputs)
    label = {0: "Resistant", 1: "Moderate", 2: "Susceptible"}[v]
    print(f"Variety: {label:<12} | Risk Score: {res['risk_score']:.4f} | Conf: {res['confidence_score']:.4f}")

print("\n--- Test Ratoon ---")
for r in [0, 1]:
    inputs = {"variety_susceptibility": 1, "is_ratoon": r, "crop_age_days": 180}
    res = api.predict(location, target_date, inputs)
    label = {0: "Plant", 1: "Ratoon"}[r]
    print(f"Crop Type: {label:<10} | Risk Score: {res['risk_score']:.4f} | Conf: {res['confidence_score']:.4f}")
