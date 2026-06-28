"""
Correct agro weight direction analysis.
W has shape (32, 3): W[neuron, feature].
The net effect of feature k on logit is proportional to W[:, k].sum()
(assuming all downstream weights are roughly uniform — an approximation,
but useful for directional signal audit).
"""
import os, sys, joblib, torch, json
import numpy as np

sys.path.insert(0, r"e:\crop-disease-outbreak-prediction-system-feature-zip-changes\crop-disease-outbreak-prediction-system-feature-zip-changes\model_train\v11\src")
MODEL_DIR = r"e:\crop-disease-outbreak-prediction-system-feature-zip-changes\crop-disease-outbreak-prediction-system-feature-zip-changes\model_train\v11\models"
from model import KGCTCN

with open(os.path.join(MODEL_DIR, "v11_metadata.json")) as f:
    meta = json.load(f)
AGRO_FEATURES  = meta["agro_features"]
WEATHER_FEATURES = meta["weather_features"]

model = KGCTCN(len(WEATHER_FEATURES), len(AGRO_FEATURES))
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "v11_kg_ctcn.pth"), map_location="cpu", weights_only=True))
model.eval()

# Get the first agro linear layer weight (32, 3)
W = None
for name, param in model.named_parameters():
    if "agro" in name and param.dim() == 2:
        W = param.detach().numpy()
        print(f"Layer: {name}  shape: {W.shape}")
        break

# Column = feature axis
print("\nPer-feature net weight direction (sum across all neurons):")
for i, feat in enumerate(AGRO_FEATURES):
    col_sum = W[:, i].sum()
    print(f"  [{i}] {feat:<30} col_sum={col_sum:+.4f}  ({'POSITIVE OK' if col_sum > 0 else 'NEGATIVE BAD'})")

# ── Manual logit probe ──────────────────────────────────────────────────────
# Push two agronomic inputs through the entire model with a fixed zero
# weather sequence to isolate the agro contribution to logit.
a_sc = joblib.load(os.path.join(MODEL_DIR, "agro_scaler.pkl"))
T    = float(joblib.load(os.path.join(MODEL_DIR, "temperature.pkl")))

zero_weather = torch.zeros(1, meta["seq_len"], len(WEATHER_FEATURES))

def probe(raw_agro):
    scaled = a_sc.transform([raw_agro])
    xa = torch.FloatTensor(scaled)
    with torch.no_grad():
        logit, _, _ = model(zero_weather, xa)
    prob = torch.sigmoid(logit / T).item()
    return logit.item(), prob

print("\n--- Variety probe (zero weather, ratoon=0, age=180) ---")
for v, label in [(0, "Resistant"), (1, "Moderate"), (2, "Susceptible")]:
    logit, prob = probe([v, 0, 180])
    print(f"  {label:<12} v={v}  logit={logit:+.4f}  P={prob:.4f}")

print("\n--- Ratoon probe (zero weather, variety=1, age=180) ---")
for r, label in [(0, "Plant"), (1, "Ratoon")]:
    logit, prob = probe([1, r, 180])
    print(f"  {label:<12} r={r}  logit={logit:+.4f}  P={prob:.4f}")
