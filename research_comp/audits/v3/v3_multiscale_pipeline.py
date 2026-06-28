import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = "v3_multiscale_validation"
DATA_FILE = "early_warning_dataset.csv"

def setup_folders():
    paths = [
        f"{BASE_DIR}/data", f"{BASE_DIR}/models",
        f"{BASE_DIR}/metrics", f"{BASE_DIR}/plots",
        f"{BASE_DIR}/logs", f"{BASE_DIR}/reports"
    ]
    for p in paths:
        os.makedirs(p, exist_ok=True)
    print("Folder structure initialized.")

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

def train_gru(X_train, y_train, X_test, y_test, num_features):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    
    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=False)
    model = GRUModel(num_features)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    model.train()
    for epoch in range(25):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        probs = model(X_test_t).numpy().flatten()
    return probs, model

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
        # Early warning success (warning 3-7 days before event)
        warning_window = range(max(0, s - 7), max(0, s - 2))
        if any(y_pred[j] == 1 for j in warning_window if j < len(y_pred)): early_warning += 1
            
    return detected / len(events), early_warning / len(events), len(events) - detected, len(events)

def main():
    setup_folders()
    df = pd.read_csv(DATA_FILE)
    
    # Step 2: Feature Cleaning
    clean_feats = [
        'WS10M', 'T2M', 'RH2M', 'PRECTOTCORR', 'T2M_MIN', 'T2M_MAX', 
        'temp_mean_7d', 'rain_7d', 'rh_7d', 'sin_day', 'cos_day', 
        'crop_age_days', 'moisture_stress', 'fungal_risk', 'red_rot_risk_composite'
    ]
    df = df[clean_feats + ['target_5d']].dropna().reset_index(drop=True)
    num_features = len(clean_feats)
    print(f"Using {num_features} cleaned features.")
    
    windows = [3, 5, 7, 14, 28]
    all_results = []
    
    for win in windows:
        print(f"\n--- Processing Window: {win} days ---")
        X_seq, y_seq = [], []
        data_vals = df[clean_feats].values
        target_vals = df['target_5d'].values
        
        for i in range(win, len(df)):
            X_seq.append(data_vals[i-win:i])
            y_seq.append(target_vals[i])
            
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)
        split = int(len(X_seq) * 0.7)
        X_train, X_test = X_seq[:split], X_seq[split:]
        y_train, y_test = y_seq[:split], y_seq[split:]
        
        # Scaling
        scaler = StandardScaler()
        X_train_flat = X_train.reshape(-1, win * num_features)
        X_test_flat = X_test.reshape(-1, win * num_features)
        X_train_scaled_flat = scaler.fit_transform(X_train_flat)
        X_test_scaled_flat = scaler.transform(X_test_flat)
        
        X_train_3d = X_train_scaled_flat.reshape(-1, win, num_features)
        X_test_3d = X_test_scaled_flat.reshape(-1, win, num_features)
        
        # 🟢 GRU
        print("Training GRU...")
        gru_probs, _ = train_gru(X_train_3d, y_train, X_test_3d, y_test, num_features)
        gru_metrics = evaluate_extended(y_test, gru_probs, [0.1, 0.2, 0.3, 0.4, 0.5])
        gru_event = detect_events((gru_probs >= gru_metrics['threshold']).astype(int), y_test)
        
        # 🟡 RF
        print("Training RF...")
        rf = RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled_flat, y_train)
        rf_probs = rf.predict_proba(X_test_scaled_flat)[:, 1]
        rf_metrics = evaluate_extended(y_test, rf_probs, [0.1, 0.2, 0.3, 0.4, 0.5])
        rf_event = detect_events((rf_probs >= rf_metrics['threshold']).astype(int), y_test)
        
        res = {
            "window": win,
            "gru_f1": gru_metrics['f1_score'], "gru_recall": gru_metrics['recall'], "gru_auc": gru_metrics['roc_auc'],
            "gru_ew": gru_event[1], "gru_miss": gru_event[2],
            "rf_f1": rf_metrics['f1_score'], "rf_recall": rf_metrics['recall'], "rf_auc": rf_metrics['roc_auc'],
            "rf_ew": rf_event[1], "rf_miss": rf_event[2]
        }
        all_results.append(res)
        
        # Save metrics for this window
        with open(f"{BASE_DIR}/metrics/window_{win}_metrics.txt", "w") as f:
            f.write(f"Window Size: {win}\nGRU Metrics: {gru_metrics}\nGRU Event: {gru_event}\n")
            f.write(f"RF Metrics: {rf_metrics}\nRF Event: {rf_event}\n")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{BASE_DIR}/reports/window_analysis_data.csv", index=False)
    
    # Step 7: Analysis Report
    with open(f"{BASE_DIR}/reports/window_analysis.txt", "w") as f:
        f.write("=== Multi-Scale Temporal Validation Analysis ===\n\n")
        f.write(results_df.to_string())
        best_win = results_df.loc[results_df['gru_f1'].idxmax(), 'window']
        f.write(f"\n\nOptimal Window (GRU F1): {best_win} days\n")
        
    # Step 8: Visualization
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(results_df['window'], results_df['gru_f1'], marker='o', label='GRU F1')
    plt.plot(results_df['window'], results_df['rf_f1'], marker='s', label='RF F1')
    plt.title('F1-Score vs Window Size'); plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(results_df['window'], results_df['gru_recall'], marker='o', label='GRU Recall')
    plt.plot(results_df['window'], results_df['rf_recall'], marker='s', label='RF Recall')
    plt.title('Recall vs Window Size'); plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(results_df['window'], results_df['gru_ew'], marker='o', label='GRU Early Warning %')
    plt.plot(results_df['window'], results_df['rf_ew'], marker='s', label='RF Early Warning %')
    plt.title('Early Warning Success % vs Window Size'); plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(results_df['window'], results_df['gru_miss'], marker='o', label='GRU Missed Outbreaks')
    plt.plot(results_df['window'], results_df['rf_miss'], marker='s', label='RF Missed Outbreaks')
    plt.title('Missed Outbreaks vs Window Size'); plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{BASE_DIR}/plots/window_scale_comparison.png")
    plt.close()
    
    print(f"\nExperiment Complete. Results in {BASE_DIR}/")

if __name__ == "__main__":
    main()
