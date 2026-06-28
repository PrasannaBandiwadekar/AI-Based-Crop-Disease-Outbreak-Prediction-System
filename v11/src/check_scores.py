import torch, joblib, numpy as np, pandas as pd, os, sys
from model import KGCTCN

# Add src to path to import assign_labels
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

from assign_causal_labels_v2 import assign_labels

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

meta_path = os.path.join(MODELS_DIR, "v11_metadata.json")
meta = __import__('json').load(open(meta_path))
wf, af, sl = meta["weather_features"], meta["agro_features"], meta["seq_len"]

df_path = os.path.join(DATA_DIR, "v11_features.csv")
df = pd.read_csv(df_path)
df["date"] = pd.to_datetime(df["date"])
df = df[df["warmup_mask"] == 0].reset_index(drop=True)

# agro_scaler
a_sc = joblib.load(os.path.join(MODELS_DIR, "agro_scaler.pkl"))
w = df[wf].values.astype("float32")
a = a_sc.transform(df[af].values.astype("float32"))

model = KGCTCN(len(wf), len(af))
model_path = os.path.join(MODELS_DIR, "v11_kg_ctcn.pth")
model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
model.eval()

# labels
df = assign_labels(df, gt_path=os.path.join(BASE_DIR, "research_comp", "evidence_base", "outbreak_events", "sangli_gt_v2.csv"))
y_true = df["risk_label"].values[sl:]

# build all sequences
X_w_all = []
X_a_all = []
for i in range(sl, len(df)):
    X_w_all.append(w[i - sl + 1 : i + 1])
    X_a_all.append(a[i])

X_w_all = torch.FloatTensor(np.array(X_w_all))
X_a_all = torch.FloatTensor(np.array(X_a_all))
y_true = torch.FloatTensor(y_true).view(-1, 1)

with torch.no_grad():
    logits, probs, _ = model(X_w_all, X_a_all)

p = probs.numpy().flatten()
y = y_true.numpy().flatten()

# build dates array
dates = pd.to_datetime(df["date"].values[sl:])

# Splits
train_mask = (dates.year >= 2005) & (dates.year <= 2014)
val_mask   = (dates.year >= 2015) & (dates.year <= 2018)
test_mask  = (dates.year >= 2019) & (dates.year <= 2021)

def print_split_stats(name, mask):
    if not mask.any():
        print(f"\nNo samples in {name} split.")
        return
    p_split = p[mask]
    y_split = y[mask]
    print(f"\n--- {name} Split ({mask.sum()} samples) ---")
    print(f"  Positives: {int(y_split.sum())}")
    print(f"  Scores: min={p_split.min():.4f}  max={p_split.max():.4f}  mean={p_split.mean():.4f}")
    print(f"  >0.1: {(p_split>0.1).sum()}  >0.3: {(p_split>0.3).sum()}  >0.5: {(p_split>0.5).sum()}")
    
    pos_idx = np.where(y_split == 1)[0]
    if len(pos_idx) > 0:
        p_pos = p_split[pos_idx]
        print(f"  Scores on Positives: min={p_pos.min():.4f}  max={p_pos.max():.4f}  mean={p_pos.mean():.4f}")
    else:
        print("  No positive samples in this split.")

print_split_stats("Train (2005-2014)", train_mask)
print_split_stats("Val (2015-2018)", val_mask)
print_split_stats("Test (2019-2021)", test_mask)
