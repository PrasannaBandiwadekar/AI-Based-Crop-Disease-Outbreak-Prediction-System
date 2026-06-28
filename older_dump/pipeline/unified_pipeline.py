import os
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

# Internal Modules
import feature_engine as fe
import models as mo
import fusion_logic as fl

# Paths
ROOT = ".." # Script is in production/pipelines/
RAW_DATA = os.path.join(ROOT, "data", "raw", "POWER_Point_Daily_20050101_20241231_016d54N_069d78E_LST.csv")
MODEL_DIR = os.path.join(ROOT, "models")
REPORT_DIR = os.path.join(ROOT, "reports")
LOG_DIR = os.path.join(ROOT, "logs")

def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(LOG_DIR, "system_log.txt"), "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(msg)

def run_pipeline():
    log("Starting Unified Production Pipeline...")
    
    # 1. RAW DATA INGESTION
    log("Phase 1: Ingesting raw weather data...")
    raw_df = pd.read_csv(RAW_DATA, skiprows=14)
    # Filter -999s
    weather_cols = ['WS10M', 'T2M', 'RH2M', 'T2M_MIN', 'T2M_MAX', 'PRECTOTCORR']
    for c in weather_cols: raw_df = raw_df[raw_df[c] != -999]
    
    # Target definition (for training/eval)
    raw_df['risk_point'] = ((raw_df['RH2M'] > 80) & (raw_df['T2M'] > 25) & (raw_df['PRECTOTCORR'] > 1)).astype(int)
    raw_df['target_5d'] = raw_df['risk_point'].rolling(window=5, min_periods=1).max().shift(-5).fillna(0).astype(int)
    
    # 2. FEATURE ENGINEERING
    log("Phase 2: Running Bio-Feature Engine + Temporal Engine...")
    df_unified = fe.build_unified_feature_space(raw_df)
    df_unified.to_csv(os.path.join(ROOT, "features", "feature_matrix.csv"), index=False)
    
    # Features to use
    features = [c for c in df_unified.columns if c not in ['YEAR', 'DOY', 'date', 'risk_point', 'target_5d']]
    num_features = len(features)
    log(f"Unified Feature Space created: {num_features} features.")
    
    # 3. SEQUENCE BUILDING
    log("Phase 3: Building Multi-Scale Sequences (5d for GRU, 14d for RF)...")
    split_idx = int(len(df_unified) * 0.7)
    train_df = df_unified.iloc[:split_idx]
    test_df = df_unified.iloc[split_idx:]
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[features])
    X_test_scaled = scaler.transform(test_df[features])
    
    # GRU sequences (5-day)
    X_gru_tr, y_gru_tr = mo.create_sequences(X_train_scaled, train_df['target_5d'].values, 5)
    X_gru_te, y_gru_te = mo.create_sequences(X_test_scaled, test_df['target_5d'].values, 5)
    
    # RF inputs (14-day flattened)
    X_rf_tr_seq, y_rf_tr = mo.create_sequences(X_train_scaled, train_df['target_5d'].values, 14)
    X_rf_te_seq, y_rf_te = mo.create_sequences(X_test_scaled, test_df['target_5d'].values, 14)
    X_rf_tr = X_rf_tr_seq.reshape(len(X_rf_tr_seq), -1)
    X_rf_te = X_rf_te_seq.reshape(len(X_rf_te_seq), -1)
    
    # 4. MODEL TRAINING
    log("Phase 4: Training Models...")
    
    # Train RF
    log("Training RF Stability Engine (14-day window)...")
    rf = RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_rf_tr, y_rf_tr)
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_stability_engine.pkl"))
    rf_probs = rf.predict_proba(X_rf_te)[:, 1]
    
    # Train GRU
    log("Training GRU Early Warning Engine (5-day sequence)...")
    gru_loader = DataLoader(TensorDataset(torch.tensor(X_gru_tr, dtype=torch.float32), 
                                          torch.tensor(y_gru_tr, dtype=torch.float32).view(-1, 1)), 
                            batch_size=32, shuffle=False)
    gru = mo.GRUModel(num_features)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(gru.parameters(), lr=0.0005)
    
    gru.train()
    for epoch in range(15): # Short training for demo
        for bx, by in gru_loader:
            optimizer.zero_grad(); out = gru(bx); loss = criterion(out, by); loss.backward(); optimizer.step()
    
    mo.save_gru_model(gru, os.path.join(MODEL_DIR, "gru_early_warning_engine.pth"))
    gru.eval()
    with torch.no_grad():
        gru_probs = gru(torch.tensor(X_gru_te, dtype=torch.float32)).numpy().flatten()
    
    # 5. FUSION DECISION ENGINE
    log("Phase 5: Executing Fusion Decision Logic...")
    # Alignment: RF needs 14 days, GRU needs 5. We align on the latest available data.
    # The last element in both test sets corresponds to the same final date.
    latest_rf_prob = rf_probs[-1]
    latest_gru_prob = gru_probs[-1]
    
    status, reason = fl.compute_fusion_risk(latest_rf_prob, latest_gru_prob)
    color = fl.get_alert_color(status)
    
    # 6. REPORTING
    log("Phase 6: Generating Production Reports...")
    report_path = os.path.join(REPORT_DIR, "fusion_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Crop Disease Early Warning Fusion Report ===\n")
        f.write(f"Generated At: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Risk Status: {status} [{color}]\n")
        f.write(f"Logic Verdict: {reason}\n\n")
        f.write("--- Signal Breakdown ---\n")
        f.write(f"RF Stability Score (14d): {latest_rf_prob:.4f}\n")
        f.write(f"GRU Spike Score (5d):     {latest_gru_prob:.4f}\n\n")
        f.write("--- Key Biological Triggers ---\n")
        # Get latest features for context
        latest_row = test_df.iloc[-1]
        f.write(f"- Fungal Risk:         {latest_row['fungal_risk']:.2f}\n")
        f.write(f"- Moisture Stress:     {latest_row['moisture_stress']:.2f}\n")
        f.write(f"- Red Rot Composite:   {latest_row['red_rot_risk_composite']:.2f}\n")
        f.write(f"- Dry-to-Wet Trigger:  {latest_row['dry_to_wet_trigger']}\n")
    
    log(f"Report saved to {report_path}")
    log("Pipeline Execution Complete.")

if __name__ == "__main__":
    run_pipeline()
