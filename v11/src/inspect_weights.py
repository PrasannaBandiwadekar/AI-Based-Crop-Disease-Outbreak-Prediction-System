import torch, joblib, json, os, numpy as np
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
SRC_DIR = os.path.join(BASE_DIR, "src")

sys.path.insert(0, SRC_DIR)
from model import KGCTCN

print("============================================================")
print(" V11 MODEL DIAGNOSTIC: WEIGHTS & TEMPERATURE")
print("============================================================")

# 1. Temperature Check
try:
    T = joblib.load(os.path.join(MODEL_DIR, "temperature.pkl"))
    print(f"Temperature T: {T:.4f}")
except Exception as e:
    print(f"Error loading temperature: {e}")

# 2. Metadata Check
try:
    with open(os.path.join(MODEL_DIR, "v11_metadata.json")) as f:
        meta = json.load(f)
    print(f"Pipeline Version: {meta.get('pipeline_version', 'Unknown')}")
except Exception as e:
    print(f"Error loading metadata: {e}")
    meta = {'weather_features': range(14), 'agro_features': range(3)}

# 3. Model Weight Norms
try:
    model = KGCTCN(len(meta['weather_features']), len(meta['agro_features']))
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "v11_kg_ctcn.pth"), map_location="cpu", weights_only=True))
    model.eval()

    print(f"\n{'Layer Name':50s}  {'Norm':>10}  {'Mean':>10}")
    print("-" * 75)
    for name, param in model.named_parameters():
        if param.requires_grad:
            norm = param.data.norm().item()
            mean = param.data.mean().item()
            print(f"{name:50s}  {norm:10.4f}  {mean:10.4f}")
except Exception as e:
    print(f"Error analyzing model: {e}")
