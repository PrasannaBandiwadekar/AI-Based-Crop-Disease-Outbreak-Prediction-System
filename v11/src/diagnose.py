import os, numpy as np, torch, joblib
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from model import KGCTCN
from assign_causal_labels_v2 import assign_labels

# Set paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

WEATHER_FEATURES = [
    "WS10M","T2M","RH2M","T2M_MIN","T2M_MAX","PRECTOTCORR",
    "T2M_MIN_lag_15d","RH_high_flag","RH_persist_7d",
    "Rain_sum_7d","Rain_sum_14d","Monsoon_ind",
    "RH2M_latent_window","T2M_latent_window",
]
AGRO_FEATURES = ["variety_susceptibility","is_ratoon","crop_age_days"]
SEQ_LEN = 28

# Load features
FEATURES_PATH = os.path.join(DATA_DIR, "v11_features.csv")
df = pd.read_csv(FEATURES_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df[df["warmup_mask"] == 0].reset_index(drop=True)

# Load GT and assign labels
GT_PATH = os.path.join(BASE_DIR,"research_comp","evidence_base",
                       "outbreak_events","sangli_gt_v2.csv")
df = assign_labels(df, gt_path=GT_PATH)
df = df.dropna(subset=WEATHER_FEATURES+AGRO_FEATURES+["risk_label"]).reset_index(drop=True)

fv = df[WEATHER_FEATURES].values.astype("float32")
av = df[AGRO_FEATURES].values.astype("float32")
lv = df["risk_label"].values.astype("float32")
dv = pd.to_datetime(df["date"].values)

X_w, X_a, y, dates = [], [], [], []
for i in range(SEQ_LEN, len(df)):
    X_w.append(fv[i-SEQ_LEN+1:i+1])
    X_a.append(av[i])
    y.append(lv[i])
    dates.append(dv[i])

X_w = np.array(X_w, dtype="float32")
X_a = np.array(X_a, dtype="float32")
y   = np.array(y,   dtype="float32")
dates = pd.DatetimeIndex(dates)

train_mask = (dates.year>=2005)&(dates.year<=2014)
val_mask   = (dates.year>=2015)&(dates.year<=2018)
test_mask  = (dates.year>=2019)&(dates.year<=2021)

# Load artifacts
a_sc = joblib.load(os.path.join(MODEL_DIR, "agro_scaler.pkl"))
X_a_sc = X_a.copy()
# Note: In train.py, scaling was applied per split using the same scaler fitted on train.
# Since we are loading a pre-fitted scaler, we can just transform everything.
# Wait, let's be safe and apply it exactly as in train.py if possible.
# Actually, a_sc is already fitted.
X_a_sc = a_sc.transform(X_a)

T = joblib.load(os.path.join(MODEL_DIR, "temperature.pkl"))
device = torch.device("cpu")
model = KGCTCN(len(WEATHER_FEATURES), len(AGRO_FEATURES)).to(device)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR,"v11_kg_ctcn.pth")))
model.eval()

def get_scores(mask):
    ds = TensorDataset(torch.FloatTensor(X_w[mask]),
                       torch.FloatTensor(X_a_sc[mask]),
                       torch.FloatTensor(y[mask]).view(-1,1))
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    preds, labels = [], []
    with torch.no_grad():
        for bw, ba, by in loader:
            logits, _, _ = model(bw, ba)
            preds.extend(torch.sigmoid(logits/T).numpy().flatten())
            labels.extend(by.numpy().flatten())
    return np.array(preds), np.array(labels)

print(f"Temperature T = {T:.4f}")

for name, mask in [("Val", val_mask), ("Test", test_mask)]:
    preds, labels = get_scores(mask)
    pos = preds[labels==1]
    neg = preds[labels==0]
    auc = roc_auc_score(labels, preds)
    ap = average_precision_score(labels, preds)
    print(f"\n{name} (AUC={auc:.3f}, AP={ap:.3f}):")
    print(f"  Positives ({len(pos)}): min={pos.min():.4f} median={np.median(pos):.4f} max={pos.max():.4f}")
    print(f"  Negatives ({len(neg)}): min={neg.min():.4f} median={np.median(neg):.4f} max={neg.max():.4f}")
    print(f"  Pos >0.1: {(pos>0.1).sum()}  Pos >0.2: {(pos>0.2).sum()}  Pos >0.3: {(pos>0.3).sum()}")
    print(f"  Neg >0.1: {(neg>0.1).sum()}  Neg >0.2: {(neg>0.2).sum()}  Neg >0.3: {(neg>0.3).sum()}")

# Relative threshold: alert when score exceeds 30-day rolling 95th percentile
test_dates  = dates[test_mask]
test_preds_rel, test_labels_rel = get_scores(test_mask)

score_series = pd.Series(test_preds_rel, index=test_dates)
roll_95      = score_series.rolling(30, min_periods=5).quantile(0.95).shift(1)
relative_alert = (score_series > roll_95).astype(int)

valid = roll_95.notna().values
from sklearn.metrics import precision_score, recall_score
p = precision_score(test_labels_rel[valid], relative_alert.values[valid], zero_division=0)
r = recall_score(test_labels_rel[valid], relative_alert.values[valid], zero_division=0)
alerts = relative_alert[valid].sum()
print(f"\nRelative threshold (top 5% of rolling 30d window):")
print(f"  Precision: {p:.3f}")
print(f"  Recall:    {r:.3f}")
print(f"  Alerts fired: {alerts} / {valid.sum()} days")
print(f"  Alert rate: {alerts/valid.sum():.1%}")
