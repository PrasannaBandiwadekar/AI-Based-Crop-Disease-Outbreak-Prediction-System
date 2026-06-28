import pickle

with open(r"E:\crop-disease-outbreak-prediction-system-feature-zip-changes\crop-disease-outbreak-prediction-system-feature-zip-changes\model_train\v11\models\temperature.pkl", "rb") as f:
    T = pickle.load(f)

# Handle tensor case
if hasattr(T, "item"):
    T = T.item()

print("Temperature:", T)

if 0.8 <= T <= 1.8:
    print("OK: In expected range")
else:
    print("WARNING: Suspicious value")