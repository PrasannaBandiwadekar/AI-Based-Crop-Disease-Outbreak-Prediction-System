import os, sys, json, numpy as np, pandas as pd, torch, joblib

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "v11", "src"))
from model import KGCTCN
from assign_causal_labels_v2 import assign_labels

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "v11", "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "v11", "models")
GT_PATH = os.path.join(BASE_DIR, "v11", "research_comp", "evidence_base", "outbreak_events", "sangli_gt_v2.csv")

def calculate_f1():
    # Load metadata
    meta = json.load(open(os.path.join(MODEL_DIR, "v11_metadata.json")))
    weather_features = meta["weather_features"]
    agro_features = meta["agro_features"]
    seq_len = int(meta["seq_len"])

    # Load scalers and model
    a_sc = joblib.load(os.path.join(MODEL_DIR, "agro_scaler.pkl"))
    T = joblib.load(os.path.join(MODEL_DIR, "temperature.pkl"))
    
    model = KGCTCN(len(weather_features), len(agro_features)).to("cpu")
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "v11_kg_ctcn.pth"), 
                                   map_location="cpu", weights_only=True))
    model.eval()

    # Load and process data
    df = pd.read_csv(os.path.join(DATA_DIR, "v11_features.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["warmup_mask"] == 0].reset_index(drop=True)
    df = assign_labels(df, gt_path=GT_PATH)

    # Prepare features
    w_vals = df[weather_features].values.astype(np.float32)
    # We don't scale agro here yet because we need to generate variants
    agro_vals_raw = df[agro_features].values.astype(np.float32)
    labels = df["risk_label"].values.astype(np.float32)

    X_w_list, X_a_list, label_list = [], [], []
    for i in range(seq_len, len(df)):
        X_w_list.append(w_vals[i-seq_len+1:i+1])
        X_a_list.append(agro_vals_raw[i])
        label_list.append(labels[i])

    X_w_np = np.array(X_w_list, dtype=np.float32)
    X_a_raw_np = np.array(X_a_list, dtype=np.float32)
    label_arr = np.array(label_list, dtype=np.float32)

    # Inference with Patch
    corrected_probs = []
    print(f"Running inference on {len(X_w_np)} samples with monotonicity patch...")
    
    with torch.no_grad():
        # Smaller batch size because we're running 6x more samples
        for s in range(0, len(X_w_np), 128):
            batch_size = min(128, len(X_w_np) - s)
            bw = torch.FloatTensor(X_w_np[s:s+batch_size])
            ba_raw_batch = X_a_raw_np[s:s+batch_size]
            
            # For each sample in batch, we need 6 agro variants
            # variety [0,1,2] x ratoon [0,1]
            # agro_features: ["variety_susceptibility", "is_ratoon", "crop_age_days"]
            
            # Construct expanded batch
            expanded_agro = []
            for i in range(batch_size):
                age = ba_raw_batch[i, 2]
                for v in [0, 1, 2]:
                    for r in [0, 1]:
                        expanded_agro.append([v, r, age])
            
            xa_scaled = torch.FloatTensor(a_sc.transform(expanded_agro))
            xw_expanded = bw.repeat_interleave(6, dim=0)
            
            logits, _, _ = model(xw_expanded, xa_scaled)
            probs = torch.sigmoid(logits / T).cpu().numpy().flatten()
            
            # Correct each group of 6
            for i in range(batch_size):
                p_group = probs[i*6 : (i+1)*6].reshape(3, 2)
                p_corrected = p_group.copy()
                for v in range(3):
                    for r in range(2):
                        p_corrected[v, r] = np.max(p_group[:v+1, :r+1])
                
                # Get the score for the original inputs
                orig_v = int(ba_raw_batch[i, 0])
                orig_r = int(ba_raw_batch[i, 1])
                corrected_probs.append(p_corrected[orig_v, orig_r])
    
    corrected_probs = np.array(corrected_probs, dtype=np.float32)

    # KG Gate
    rh_persist = df["RH_persist_7d"].values[seq_len:]
    rain_sum   = df["Rain_sum_7d"].values[seq_len:]
    gated_probs = corrected_probs.copy()
    for i in range(len(corrected_probs)):
        if rh_persist[i] < 2.0 and rain_sum[i] < 5.0:
            gated_probs[i] = min(gated_probs[i], 0.15)

    # Metrics at 0.20 threshold
    preds = (gated_probs >= 0.20).astype(int)
    tp = int(((preds == 1) & (label_arr == 1)).sum())
    fp = int(((preds == 1) & (label_arr == 0)).sum())
    fn = int(((preds == 0) & (label_arr == 1)).sum())
    tn = int(((preds == 0) & (label_arr == 0)).sum())

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    print("\n--- RESULTS WITH MONOTONICITY PATCH ---")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    calculate_f1()
