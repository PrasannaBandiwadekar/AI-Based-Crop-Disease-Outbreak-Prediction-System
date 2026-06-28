import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

# Constants
BASE_DIR = "v3_multiscale_validation"
DATA_FILE = "early_warning_dataset.csv"
CLEAN_FEATURES = [
    'WS10M', 'T2M', 'RH2M', 'PRECTOTCORR', 'T2M_MIN', 'T2M_MAX', 
    'temp_mean_7d', 'rain_7d', 'rh_7d', 'sin_day', 'cos_day', 
    'crop_age_days', 'moisture_stress', 'fungal_risk', 'red_rot_risk_composite'
]
WINDOWS = [3, 5, 7, 14, 28]

def log_event(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(f"{BASE_DIR}/logs/full_run_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message)

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1, dropout=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, h_n = self.gru(x)
        out = self.dropout(h_n[-1])
        out = self.fc(out)
        return self.sigmoid(out)

def train_gru(X_train, y_train, X_test, y_test, window, num_features):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    
    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=False)
    model = GRUModel(num_features)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    model_dir = f"{BASE_DIR}/models/gru/{window}"
    os.makedirs(model_dir, exist_ok=True)
    log_path = f"{model_dir}/train_log.txt"
    
    start_time = time.time()
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Training GRU for window {window}\n")
        model.train()
        for epoch in range(25):
            epoch_loss = 0
            for bx, by in loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            f.write(f"Epoch {epoch+1}/25, Loss: {avg_loss:.4f}\n")
            
    train_duration = time.time() - start_time
    torch.save(model.state_dict(), f"{model_dir}/model.pth")
    
    model.eval()
    with torch.no_grad():
        probs = model(X_test_t).numpy().flatten()
    return probs, train_duration

def detect_events(y_pred, y_true):
    events = []
    start = None
    count = 0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            if start is None: start = i
            count += 1
        else:
            if count >= 3: events.append((start, i-1))
            start, count = None, 0
    if count >= 3: events.append((start, len(y_true)-1))
    
    if not events: return 0, 0, 0, 0
    
    detected = 0
    early_warning = 0
    for s, e in events:
        if any(y_pred[s:e+1] == 1): detected += 1
        warning_window = range(max(0, s - 7), max(0, s - 2))
        if any(y_pred[j] == 1 for j in warning_window if j < len(y_pred)): early_warning += 1
            
    return detected / len(events), early_warning / len(events), len(events) - detected, len(events)

def evaluate_extended(y_true, probs, thresholds):
    best_f1, best_metrics = -1, {}
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f = f1_score(y_true, preds, zero_division=0)
        
        tn = np.sum((preds == 0) & (y_true == 0))
        fp = np.sum((preds == 1) & (y_true == 0))
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        if f > best_f1:
            best_f1 = f
            best_metrics = {
                "precision": p, "recall": r, "f1_score": f,
                "accuracy": accuracy_score(y_true, preds),
                "threshold": thresh, "far": far,
                "brier_score": brier_score_loss(y_true, probs)
            }
    best_metrics["roc_auc"] = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.0
    return best_metrics

def main():
    if not os.path.exists(f"{BASE_DIR}/logs/full_run_log.txt"):
        os.makedirs(f"{BASE_DIR}/logs", exist_ok=True)
        with open(f"{BASE_DIR}/logs/full_run_log.txt", "w", encoding="utf-8") as f:
            f.write("=== Unified RF vs GRU Multi-Scale Comparison Log ===\n")

    log_event("Loading data...")
    df = pd.read_csv(DATA_FILE)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    cleaned_df = df[CLEAN_FEATURES + ['target_5d', 'date']].dropna().reset_index(drop=True)
    num_features = len(CLEAN_FEATURES)
    log_event(f"Dataset shape: {cleaned_df.shape}")

    master_results = []

    for w in WINDOWS:
        log_event(f"--- Processing Scale: {w} days ---")
        
        # Step 4: Create and Store Datasets
        data_vals = cleaned_df[CLEAN_FEATURES].values
        target_vals = cleaned_df['target_5d'].values
        dates_vals = cleaned_df['date'].values
        
        X_seq, y_seq, dates_seq = [], [], []
        for i in range(w, len(cleaned_df)):
            X_seq.append(data_vals[i-w:i])
            y_seq.append(target_vals[i])
            dates_seq.append(dates_vals[i])
            
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)
        
        data_dir = f"{BASE_DIR}/data/seq_{w}"
        os.makedirs(data_dir, exist_ok=True)
        np.save(f"{data_dir}/sequence.npy", X_seq)
        
        meta = {
            "window": w,
            "shape": X_seq.shape,
            "features": CLEAN_FEATURES,
            "time_range": [str(dates_seq[0]), str(dates_seq[-1])]
        }
        with open(f"{data_dir}/meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)
        
        log_event(f"Sequence {w} stored. Shape: {X_seq.shape}")
        
        # Split
        split = int(len(X_seq) * 0.7)
        X_train, X_test = X_seq[:split], X_seq[split:]
        y_train, y_test = y_seq[:split], y_seq[split:]
        
        # Normalization
        scaler = StandardScaler()
        X_train_flat = X_train.reshape(-1, w * num_features)
        X_test_flat = X_test.reshape(-1, w * num_features)
        X_train_scaled_flat = scaler.fit_transform(X_train_flat)
        X_test_scaled_flat = scaler.transform(X_test_flat)
        
        # 🟢 GRU
        log_event(f"Training GRU {w}...")
        X_train_3d = X_train_scaled_flat.reshape(-1, w, num_features)
        X_test_3d = X_test_scaled_flat.reshape(-1, w, num_features)
        gru_probs, gru_time = train_gru(X_train_3d, y_train, X_test_3d, y_test, w, num_features)
        
        gru_metrics = evaluate_extended(y_test, gru_probs, [0.1, 0.2, 0.3, 0.4, 0.5])
        gru_event = detect_events((gru_probs >= gru_metrics['threshold']).astype(int), y_test)
        
        gru_res = {
            "window": w, "model": "GRU", "f1": gru_metrics['f1_score'], 
            "recall": gru_metrics['recall'], "auc": gru_metrics['roc_auc'],
            "early_warning": gru_event[1], "missed": gru_event[2]
        }
        master_results.append(gru_res)
        
        with open(f"{BASE_DIR}/metrics/{w}_gru_metrics.json", "w", encoding="utf-8") as f:
            json.dump({**gru_metrics, "event": gru_event}, f, indent=4)
            
        # 🟡 RF
        log_event(f"Training RF {w}...")
        rf_dir = f"{BASE_DIR}/models/rf/{w}"
        os.makedirs(rf_dir, exist_ok=True)
        rf_start = time.time()
        rf = RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled_flat, y_train)
        rf_time = time.time() - rf_start
        joblib.dump(rf, f"{rf_dir}/model.pkl")
        
        rf_probs = rf.predict_proba(X_test_scaled_flat)[:, 1]
        rf_metrics = evaluate_extended(y_test, rf_probs, [0.1, 0.2, 0.3, 0.4, 0.5])
        rf_event = detect_events((rf_probs >= rf_metrics['threshold']).astype(int), y_test)
        
        rf_res = {
            "window": w, "model": "RF", "f1": rf_metrics['f1_score'], 
            "recall": rf_metrics['recall'], "auc": rf_metrics['roc_auc'],
            "early_warning": rf_event[1], "missed": rf_event[2]
        }
        master_results.append(rf_res)
        
        with open(f"{BASE_DIR}/metrics/{w}_rf_metrics.json", "w", encoding="utf-8") as f:
            json.dump({**rf_metrics, "event": rf_event}, f, indent=4)
            
        log_event(f"Window {w} complete. GRU: {gru_metrics['f1_score']:.4f}, RF: {rf_metrics['f1_score']:.4f}")

    # Step 7: Master Comparison Table
    results_df = pd.DataFrame(master_results)
    with open(f"{BASE_DIR}/reports/rf_vs_gru_multiscale.txt", "w", encoding="utf-8") as f:
        f.write("=== Master Multi-Scale Comparison (RF vs GRU) ===\n\n")
        f.write(results_df.to_string(index=False))
        
        f.write("\n\n=== Interpretation ===\n")
        gru_avg_f1 = results_df[results_df['model']=='GRU']['f1'].mean()
        rf_avg_f1 = results_df[results_df['model']=='RF']['f1'].mean()
        f.write(f"Average F1 - GRU: {gru_avg_f1:.4f} | RF: {rf_avg_f1:.4f}\n")
        
        if abs(gru_avg_f1 - rf_avg_f1) < 0.02:
            f.write("Verdict: Predictability is primarily driven by TEMPORAL WINDOW STRUCTURE.\n")
        else:
            f.write(f"Verdict: Predictability is driven by MODEL CHOICE ({'GRU' if gru_avg_f1 > rf_avg_f1 else 'RF'} Advantage).\n")

    # Step 8: Visualization
    os.makedirs(f"{BASE_DIR}/plots", exist_ok=True)
    
    # F1 vs Window
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x='window', y='f1', hue='model', marker='o')
    plt.title('F1-Score vs Window Size (RF vs GRU)')
    plt.savefig(f"{BASE_DIR}/plots/f1_vs_window.png")
    plt.close()
    
    # Early Warning Success
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x='window', y='early_warning', hue='model', marker='s')
    plt.title('Early Warning Success vs Window Size')
    plt.savefig(f"{BASE_DIR}/plots/ew_vs_window.png")
    plt.close()
    
    # Model Gap Plot
    pivot_f1 = results_df.pivot(index='window', columns='model', values='f1')
    pivot_f1['gap'] = pivot_f1['GRU'] - pivot_f1['RF']
    plt.figure(figsize=(10, 6))
    plt.bar(pivot_f1.index.astype(str), pivot_f1['gap'], color='skyblue')
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Performance Gap (GRU - RF)')
    plt.ylabel('F1 Difference')
    plt.savefig(f"{BASE_DIR}/plots/model_gap.png")
    plt.close()

    log_event("Full experiment complete.")

if __name__ == "__main__":
    main()
