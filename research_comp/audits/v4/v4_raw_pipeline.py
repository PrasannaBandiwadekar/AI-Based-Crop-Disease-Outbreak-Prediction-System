import os
import time
import shutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = "v4"
RAW_FILE = "POWER_Point_Daily_20050101_20241231_016d54N_069d78E_LST.csv"

def setup_folders():
    for path in [
        f"{BASE_DIR}/data/raw", f"{BASE_DIR}/data/processed",
        f"{BASE_DIR}/models/rf", f"{BASE_DIR}/models/gru",
        f"{BASE_DIR}/metrics", f"{BASE_DIR}/plots",
        f"{BASE_DIR}/logs", f"{BASE_DIR}/reports"
    ]:
        os.makedirs(path, exist_ok=True)
    print("v4 folder structure created.")

def evaluate_thresholds(y_true, probs, thresholds):
    best_f1, best_thresh, best_metrics = -1, 0.5, {}
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f = f1_score(y_true, preds, zero_division=0)
        a = accuracy_score(y_true, preds)
        if f > best_f1:
            best_f1 = f
            best_thresh = thresh
            best_metrics = {"precision": p, "recall": r, "f1_score": f, "accuracy": a}
    auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.0
    best_metrics["roc_auc"] = auc
    best_metrics["threshold"] = best_thresh
    return best_metrics

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

def main():
    setup_folders()
    log_lines = []

    # Step 1: Copy raw data
    dest = os.path.join(BASE_DIR, "data", "raw", RAW_FILE)
    if not os.path.exists(dest):
        shutil.copy2(RAW_FILE, dest)
    log_lines.append(f"Raw data copied to {dest}")

    # Step 2: Load data (skip 14-line header)
    print("Loading raw POWER data...")
    df = pd.read_csv(dest, skiprows=14)
    log_lines.append(f"Raw dataset shape: {df.shape}")
    log_lines.append(f"Columns: {list(df.columns)}")

    # Create date from YEAR + DOY
    df['date'] = pd.to_datetime(df['YEAR'].astype(str) + df['DOY'].astype(str).str.zfill(3), format='%Y%j')
    df = df.sort_values('date').reset_index(drop=True)

    # Remove missing data rows (-999)
    raw_features = ['WS10M', 'T2M', 'RH2M', 'T2M_MIN', 'T2M_MAX', 'PRECTOTCORR']
    for col in raw_features:
        df = df[df[col] != -999]
    df = df.reset_index(drop=True)
    log_lines.append(f"After removing -999 rows: {df.shape}")

    # Step 4: Create target_5d (same logic as v3)
    # target_5d = 1 if any of the next 5 days has conditions favorable for disease
    # Using a simplified proxy: high humidity (>80%) AND warm temp (>25C) AND precipitation > 1mm
    df['disease_risk'] = ((df['RH2M'] > 80) & (df['T2M'] > 25) & (df['PRECTOTCORR'] > 1)).astype(int)
    df['target_5d'] = df['disease_risk'].rolling(window=5, min_periods=1).max().shift(-5).fillna(0).astype(int)
    df = df.dropna().reset_index(drop=True)
    log_lines.append(f"Target distribution: {dict(df['target_5d'].value_counts())}")

    # Step 3: Create 14-day sequences using ONLY raw features
    print("Creating 14-day sequences...")
    window = 14
    num_features = len(raw_features)
    X_list, y_list, dates_list = [], [], []

    feature_data = df[raw_features].values
    target_data = df['target_5d'].values
    date_data = df['date'].values

    for i in range(window, len(df)):
        X_list.append(feature_data[i - window:i].flatten())
        y_list.append(target_data[i])
        dates_list.append(date_data[i])

    # Build sequence CSV
    col_names = []
    for day in range(window):
        for f in raw_features:
            col_names.append(f"{f}_lag_{window - day}")

    seq_df = pd.DataFrame(X_list, columns=col_names)
    seq_df['target_5d'] = y_list
    seq_df['date'] = dates_list
    seq_path = os.path.join(BASE_DIR, "data", "processed", "seq_window_14_raw.csv")
    seq_df.to_csv(seq_path, index=False)
    log_lines.append(f"Sequence dataset shape: {seq_df.shape}")
    log_lines.append(f"Sequence count: {len(seq_df)}")
    print(f"Sequences saved: {seq_df.shape}")

    # Step 5: Train-Test Split
    split = int(len(seq_df) * 0.7)
    X_all = seq_df[col_names].values
    y_all = seq_df['target_5d'].values.astype(float)

    X_train_raw, X_test_raw = X_all[:split], X_all[split:]
    y_train, y_test = y_all[:split], y_all[split:]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    # ============ RANDOM FOREST ============
    print("\n--- Training Random Forest ---")
    rf_start = time.time()
    rf = RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train.astype(int))
    rf_time = time.time() - rf_start

    rf_probs = rf.predict_proba(X_test_scaled)[:, 1]
    rf_metrics = evaluate_thresholds(y_test.astype(int), rf_probs, thresholds)
    rf_preds = (rf_probs >= rf_metrics['threshold']).astype(int)

    joblib.dump(rf, os.path.join(BASE_DIR, "models", "rf", "rf_model.pkl"))
    with open(os.path.join(BASE_DIR, "metrics", "rf_metrics.txt"), "w", encoding="utf-8") as f:
        for k, v in rf_metrics.items():
            f.write(f"{k}: {v}\n")

    log_lines.append(f"RF training time: {rf_time:.2f}s")
    log_lines.append(f"RF F1: {rf_metrics['f1_score']:.4f}")
    print(f"RF F1={rf_metrics['f1_score']:.4f}, AUC={rf_metrics['roc_auc']:.4f}")

    # RF plot
    plt.figure(figsize=(15, 5))
    test_dates = seq_df.iloc[split:]['date'].values
    plt.plot(pd.to_datetime(test_dates), y_test, label='Actual', alpha=0.7)
    plt.plot(pd.to_datetime(test_dates), rf_preds, label='RF Predicted', alpha=0.7)
    plt.title('RF Prediction vs Actual (v4 Raw)')
    plt.xlabel('Date'); plt.ylabel('Target 5d'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "plots", "rf_prediction_vs_actual.png"))
    plt.close()

    # ============ GRU ============
    print("\n--- Training GRU ---")
    X_train_3d = X_train_scaled.reshape(-1, window, num_features)
    X_test_3d = X_test_scaled.reshape(-1, window, num_features)

    X_train_t = torch.tensor(X_train_3d, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test_3d, dtype=torch.float32)

    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=False)

    torch.manual_seed(42)
    model = GRUModel(num_features, hidden_size=128, num_layers=1, dropout=0.2)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    gru_start = time.time()
    model.train()
    loss_history = []
    for epoch in range(25):
        epoch_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(loader)
        loss_history.append(avg)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/25, Loss: {avg:.4f}")
    gru_time = time.time() - gru_start

    model.eval()
    with torch.no_grad():
        gru_probs = model(X_test_t).numpy().flatten()
    gru_metrics = evaluate_thresholds(y_test.astype(int), gru_probs, thresholds)
    gru_preds = (gru_probs >= gru_metrics['threshold']).astype(int)

    torch.save(model.state_dict(), os.path.join(BASE_DIR, "models", "gru", "gru_model.pth"))
    with open(os.path.join(BASE_DIR, "metrics", "gru_metrics.txt"), "w", encoding="utf-8") as f:
        for k, v in gru_metrics.items():
            f.write(f"{k}: {v}\n")

    log_lines.append(f"GRU training time: {gru_time:.2f}s")
    log_lines.append(f"GRU F1: {gru_metrics['f1_score']:.4f}")
    print(f"GRU F1={gru_metrics['f1_score']:.4f}, AUC={gru_metrics['roc_auc']:.4f}")

    # GRU plot
    plt.figure(figsize=(15, 5))
    plt.plot(pd.to_datetime(test_dates), y_test, label='Actual', alpha=0.7)
    plt.plot(pd.to_datetime(test_dates), gru_preds, label='GRU Predicted', alpha=0.7)
    plt.title('GRU Prediction vs Actual (v4 Raw)')
    plt.xlabel('Date'); plt.ylabel('Target 5d'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "plots", "gru_prediction_vs_actual.png"))
    plt.close()

    # ============ COMPARISON REPORT ============
    # v3 best results (from previous experiments)
    v3_rf = {"f1_score": 0.3161, "precision": 0.2480, "recall": 0.4357, "roc_auc": 0.8289}
    v3_gru = {"f1_score": 0.3224, "precision": 0.2988, "recall": 0.3500, "roc_auc": 0.8216}

    with open(os.path.join(BASE_DIR, "reports", "v4_vs_v3_analysis.txt"), "w", encoding="utf-8") as f:
        f.write("=== v4 (Raw Temporal) vs v3 (Engineered Features) Analysis ===\n\n")

        f.write("### 1. RF (v4) vs RF (v3)\n")
        f.write(f"Metric     | v3 RF    | v4 RF (raw)\n")
        f.write(f"-----------|----------|----------\n")
        f.write(f"F1-Score   | {v3_rf['f1_score']:.4f}   | {rf_metrics['f1_score']:.4f}\n")
        f.write(f"Precision  | {v3_rf['precision']:.4f}   | {rf_metrics['precision']:.4f}\n")
        f.write(f"Recall     | {v3_rf['recall']:.4f}   | {rf_metrics['recall']:.4f}\n")
        f.write(f"ROC-AUC    | {v3_rf['roc_auc']:.4f}   | {rf_metrics['roc_auc']:.4f}\n")
        rf_diff = rf_metrics['f1_score'] - v3_rf['f1_score']
        f.write(f"Impact: {'Raw features IMPROVED RF' if rf_diff > 0 else 'Feature engineering HELPED RF'} (diff: {rf_diff:+.4f})\n\n")

        f.write("### 2. GRU (v4) vs GRU (v3)\n")
        f.write(f"Metric     | v3 GRU   | v4 GRU (raw)\n")
        f.write(f"-----------|----------|----------\n")
        f.write(f"F1-Score   | {v3_gru['f1_score']:.4f}   | {gru_metrics['f1_score']:.4f}\n")
        f.write(f"Precision  | {v3_gru['precision']:.4f}   | {gru_metrics['precision']:.4f}\n")
        f.write(f"Recall     | {v3_gru['recall']:.4f}   | {gru_metrics['recall']:.4f}\n")
        f.write(f"ROC-AUC    | {v3_gru['roc_auc']:.4f}   | {gru_metrics['roc_auc']:.4f}\n")
        gru_diff = gru_metrics['f1_score'] - v3_gru['f1_score']
        f.write(f"Impact: {'Raw sequences IMPROVED GRU' if gru_diff > 0 else 'Engineered features HELPED GRU'} (diff: {gru_diff:+.4f})\n\n")

        f.write("### 3. RF vs GRU (v4 raw data)\n")
        f.write(f"Metric     | RF (v4)  | GRU (v4)\n")
        f.write(f"-----------|----------|----------\n")
        f.write(f"F1-Score   | {rf_metrics['f1_score']:.4f}   | {gru_metrics['f1_score']:.4f}\n")
        f.write(f"Precision  | {rf_metrics['precision']:.4f}   | {gru_metrics['precision']:.4f}\n")
        f.write(f"Recall     | {rf_metrics['recall']:.4f}   | {gru_metrics['recall']:.4f}\n")
        f.write(f"ROC-AUC    | {rf_metrics['roc_auc']:.4f}   | {gru_metrics['roc_auc']:.4f}\n")
        v4_winner = "RF" if rf_metrics['f1_score'] >= gru_metrics['f1_score'] else "GRU"
        f.write(f"Winner on raw data: {v4_winner}\n\n")

        f.write("### 4. Feature Engineering Impact\n")
        avg_v3 = (v3_rf['f1_score'] + v3_gru['f1_score']) / 2
        avg_v4 = (rf_metrics['f1_score'] + gru_metrics['f1_score']) / 2
        fe_impact = avg_v3 - avg_v4
        if fe_impact > 0.02:
            f.write(f"YES, feature engineering significantly improves results (avg F1 diff: {fe_impact:+.4f})\n\n")
        elif fe_impact > 0:
            f.write(f"Feature engineering provides marginal improvement (avg F1 diff: {fe_impact:+.4f})\n\n")
        else:
            f.write(f"NO, raw temporal modeling matches or beats engineered features (avg F1 diff: {fe_impact:+.4f})\n\n")

        f.write("### 5. Final Decision\n")
        best_overall_f1 = max(v3_rf['f1_score'], v3_gru['f1_score'], rf_metrics['f1_score'], gru_metrics['f1_score'])
        if best_overall_f1 == rf_metrics['f1_score'] or best_overall_f1 == gru_metrics['f1_score']:
            f.write("Recommendation: v4 (raw temporal) - raw features are sufficient\n")
        else:
            f.write("Recommendation: v3 (engineered features) - feature engineering adds value\n")

        f.write(f"\nAnswer: Can raw temporal modeling replace engineered features?\n")
        if avg_v4 >= avg_v3 - 0.02:
            f.write("YES - raw temporal modeling can effectively replace engineered features for this problem.\n")
        else:
            f.write("NO - engineered features provide meaningful improvement that raw temporal modeling cannot match.\n")

    # ============ LOG ============
    with open(os.path.join(BASE_DIR, "logs", "run_log.txt"), "w", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")

    print(f"\nv4 Pipeline completed. All outputs in {BASE_DIR}/")

if __name__ == "__main__":
    main()
