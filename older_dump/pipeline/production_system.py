import os
import time
import json
import uuid
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

# Internal Modules
import feature_engine as fe
import models as mo
import fusion_logic as fl

class RunManager:
    def __init__(self, base_path="../runs"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_path = os.path.abspath(os.path.join(script_dir, base_path))
        self.run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        self.run_path = os.path.join(self.base_path, self.run_id)
        self._init_folders()
        
    def _init_folders(self):
        subfolders = ["data_snapshot", "features", "models", "metrics", "logs", "drift", "plots"]
        for f in subfolders:
            os.makedirs(os.path.join(self.run_path, f), exist_ok=True)
            
    def log(self, msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(self.run_path, "logs/system_log.txt"), "a") as f:
            f.write(f"[{ts}] {msg}\n")
        print(f"[{self.run_id}] {msg}")

class DriftMonitor:
    def __init__(self, run_manager):
        self.rm = run_manager
        
    def check_drift(self, current_metrics, historical_metrics=None):
        drift_report = {
            "drift_detected": False,
            "reasons": []
        }
        
        if historical_metrics:
            # Check F1 degradation (> 10%)
            if current_metrics['f1'] < historical_metrics['f1'] * 0.9:
                drift_report["drift_detected"] = True
                drift_report["reasons"].append("F1 degradation > 10%")
                
        with open(os.path.join(self.rm.run_path, "drift/drift_report.json"), "w") as f:
            json.dump(drift_report, f, indent=4)
        return drift_report

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
    
    detected = sum(1 for s, e in events if any(y_pred[s:e+1] == 1))
    early_warning = 0
    for s, e in events:
        warning_window = range(max(0, s - 7), max(0, s - 2))
        if any(y_pred[j] == 1 for j in warning_window if j < len(y_pred)): early_warning += 1
            
    return detected / len(events), early_warning / len(events), len(events) - detected, len(events)

def run_production_cycle():
    rm = RunManager()
    rm.log("Initializing Production-Grade System Upgrade...")
    
    # 1. DATA INGESTION & SNAPSHOT
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_source = os.path.abspath(os.path.join(script_dir, "../data/raw/POWER_Point_Daily_20050101_20241231_016d54N_069d78E_LST.csv"))
    shutil.copy(raw_source, os.path.join(rm.run_path, "data_snapshot/raw_weather.csv"))
    raw_df = pd.read_csv(raw_source, skiprows=14)
    weather_cols = ['WS10M', 'T2M', 'RH2M', 'T2M_MIN', 'T2M_MAX', 'PRECTOTCORR']
    for c in weather_cols: raw_df = raw_df[raw_df[c] != -999]
    raw_df['risk_point'] = ((raw_df['RH2M'] > 80) & (raw_df['T2M'] > 25) & (raw_df['PRECTOTCORR'] > 1)).astype(int)
    raw_df['target_5d'] = raw_df['risk_point'].rolling(window=5, min_periods=1).max().shift(-5).fillna(0).astype(int)
    
    # 2. FEATURE VERSIONING
    rm.log("Phase 2: Feature Engineering & Versioning...")
    df_unified = fe.build_unified_feature_space(raw_df)
    df_unified.to_csv(os.path.join(rm.run_path, "features/feature_matrix.csv"), index=False)
    
    features = [c for c in df_unified.columns if c not in ['YEAR', 'DOY', 'date', 'risk_point', 'target_5d']]
    signature = {"feature_count": len(features), "features": features, "version": "v3.bio"}
    with open(os.path.join(rm.run_path, "features/feature_signature.json"), "w") as f:
        json.dump(signature, f, indent=4)
        
    # 3. TEMPORAL SPLIT (70/15/15)
    rm.log("Phase 3: Temporal Split (70/15/15)...")
    n = len(df_unified)
    train_df = df_unified.iloc[:int(n*0.7)]
    val_df = df_unified.iloc[int(n*0.7):int(n*0.85)]
    test_df = df_unified.iloc[int(n*0.85):]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(train_df[features])
    X_val_s = scaler.transform(val_df[features])
    X_test_s = scaler.transform(test_df[features])
    joblib.dump(scaler, os.path.join(rm.run_path, "models/scaler.pkl"))
    
    # 4. TRAINING & EVALUATION
    rm.log("Phase 4: Multi-Scale Training & Evaluation...")
    
    # RF Stability Engine (14d)
    X_rf_tr_seq, y_rf_tr = mo.create_sequences(X_train_s, train_df['target_5d'].values, 14)
    X_rf_te_seq, y_rf_te = mo.create_sequences(X_test_s, test_df['target_5d'].values, 14)
    X_rf_tr = X_rf_tr_seq.reshape(len(X_rf_tr_seq), -1)
    X_rf_te = X_rf_te_seq.reshape(len(X_rf_te_seq), -1)
    
    rf = RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_rf_tr, y_rf_tr)
    joblib.dump(rf, os.path.join(rm.run_path, "models/rf_stability.pkl"))
    rf_probs = rf.predict_proba(X_rf_te)[:, 1]
    
    # GRU Early Warning Engine (5d)
    X_gru_tr, y_gru_tr = mo.create_sequences(X_train_s, train_df['target_5d'].values, 5)
    X_gru_te, y_gru_te = mo.create_sequences(X_test_s, test_df['target_5d'].values, 5)
    
    gru = mo.GRUModel(len(features))
    loader = DataLoader(TensorDataset(torch.tensor(X_gru_tr, dtype=torch.float32), 
                                      torch.tensor(y_gru_tr, dtype=torch.float32).view(-1, 1)), batch_size=32, shuffle=False)
    opt = optim.Adam(gru.parameters(), lr=0.0005)
    crit = nn.BCELoss()
    for e in range(15):
        for bx, by in loader:
            opt.zero_grad(); out = gru(bx); loss = crit(out, by); loss.backward(); opt.step()
    torch.save(gru.state_dict(), os.path.join(rm.run_path, "models/gru_spike.pth"))
    
    gru.eval()
    with torch.no_grad():
        gru_probs = gru(torch.tensor(X_gru_te, dtype=torch.float32)).numpy().flatten()
        
    # Metrics
    rm.log("Computing Metrics...")
    rf_preds = (rf_probs >= 0.5).astype(int)
    gru_preds = (gru_probs >= 0.5).astype(int)
    
    # Alignment: RF metrics (on its test set)
    rf_met = {
        "f1": f1_score(y_rf_te, rf_preds),
        "recall": recall_score(y_rf_te, rf_preds),
        "auc": roc_auc_score(y_rf_te, rf_probs),
        "event_metrics": detect_events(rf_preds, y_rf_te)
    }
    gru_met = {
        "f1": f1_score(y_gru_te, gru_preds),
        "recall": recall_score(y_gru_te, gru_preds),
        "auc": roc_auc_score(y_gru_te, gru_probs),
        "event_metrics": detect_events(gru_preds, y_gru_te)
    }
    
    with open(os.path.join(rm.run_path, "metrics/model_metrics.json"), "w") as f:
        json.dump({"rf": rf_met, "gru": gru_met}, f, indent=4)
        
    # 5. DRIFT MONITORING
    dm = DriftMonitor(rm)
    dm.check_drift(rf_met) # Placeholder for now as it's the first run
    
    # 6. FUSION REPORT
    status, reason = fl.compute_fusion_risk(rf_probs[-1], gru_probs[-1])
    report_path = os.path.join(rm.run_path, "fusion_report.txt")
    with open(report_path, "w") as f:
        f.write(f"RUN_ID: {rm.run_id}\nStatus: {status}\nReason: {reason}\n")
        f.write(f"RF F1: {rf_met['f1']:.4f} | GRU F1: {gru_met['f1']:.4f}\n")
        
    rm.log(f"Cycle Complete. Run versioned at {rm.run_path}")

if __name__ == "__main__":
    run_production_cycle()
