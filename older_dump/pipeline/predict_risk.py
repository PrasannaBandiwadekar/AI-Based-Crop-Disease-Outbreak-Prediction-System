import pandas as pd
import numpy as np
import joblib
import json

def classify_risk(prob):
    if prob > 0.7:
        return "HIGH - Immediate action required!", "🚨"
    elif prob > 0.4:
        return "MEDIUM - Monitor closely.", "👀"
    else:
        return "LOW - No action needed.", "✅"

# 1. LOAD MODEL & METADATA
print("🔍 Initializing Early Warning System...")
try:
    model = joblib.load("red_rot_forecast_v1.joblib")
    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)
    
    features = metadata['features']
    print(f"✅ Model Loaded (AUC: {metadata['validation_auc']:.4f})")
    print(f"📊 Using {len(features)} optimized features.")
except FileNotFoundError:
    print("❌ Error: Model files not found. Please run 'train_final_model.py' first.")
    exit()

# 2. LOAD LATEST DATA (Simulated real-time buffer)
# In a real app, this would come from a weather API call formatted to these features
df = pd.read_csv('optimized_early_warning_dataset.csv')
latest_data = df[features].tail(5)
dates = df['date'].tail(5).tolist()

# 3. RUN INFERENCE
probabilities = model.predict_proba(latest_data)[:, 1]

print("\n=== Sugarcane Red Rot Early Warning System ===")
print(f"{'Date Trace':<12} | {'Probability':<12} | {'Risk Level'}")
print("-" * 65)

for i, prob in enumerate(probabilities):
    status, emoji = classify_risk(prob)
    print(f"{dates[i]:<12} | {prob:<12.2f} | {emoji} {status}")

print("\nAdvice: If Medium/High risk is detected, check leaf humidity and recent rainfall history.")
