import os
import time
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import class_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Track generated files
generated_files = []

def track_file(filepath):
    generated_files.append(filepath)
    return filepath

def init_directories(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    dirs = ['models', 'metrics', 'plots', 'logs']
    for d in dirs:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)
    print(f"Initialized subdirectories in {base_dir}")

def evaluate_thresholds(y_true, y_prob, thresholds):
    best_f1 = -1
    best_thresh = 0.5
    best_metrics = {}
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy_score(y_true, y_pred)
            }
            
    return best_thresh, best_metrics

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # We only need the last time step output
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

def main():
    BASE_DIR = os.path.join("v3", "lstm_run")
    init_directories(BASE_DIR)
    
    log_file = track_file(os.path.join(BASE_DIR, 'logs', 'training_log.txt'))
    with open(log_file, 'w') as f:
        f.write("Training Log - LSTM Run (PyTorch)\n=====================\n")
        f.write("Thresholds tested: [0.1, 0.2, 0.3, 0.4, 0.5]\n\n")
        
    core_features = ['T2M', 'RH2M', 'PRECTOTCORR', 'WS10M', 'temp_range']
    window_size = 14
    num_features = len(core_features)
    
    seq_path = os.path.join("v3", "data", f"seq_window_{window_size}.csv")
    if not os.path.exists(seq_path):
        raise FileNotFoundError(f"Sequence data file not found: {seq_path}")
        
    print(f"Loading sequences from {seq_path}...")
    df = pd.read_csv(seq_path)
    
    feature_cols = [c for c in df.columns if '_lag_' in c]
    
    X_raw = df[feature_cols].values
    y_raw = df['target_5d'].values.astype(float)
    
    test_size = 0.3
    split_idx = int(len(df) * (1 - test_size))
    
    X_train_flat = X_raw[:split_idx]
    y_train_raw = y_raw[:split_idx]
    X_test_flat = X_raw[split_idx:]
    y_test_raw = y_raw[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    # Reshape to (samples, time_steps, features)
    X_train_reshaped = X_train_scaled.reshape(-1, window_size, num_features)
    X_test_reshaped = X_test_scaled.reshape(-1, window_size, num_features)
    
    # Convert to Tensors
    X_train_tensor = torch.tensor(X_train_reshaped, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_raw, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_reshaped, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_raw, dtype=torch.float32).view(-1, 1)
    
    # Calculate pos_weight for class imbalance
    num_pos = np.sum(y_train_raw)
    num_neg = len(y_train_raw) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32) if num_pos > 0 else torch.tensor([1.0])
    
    print(f"Class imbalance: Pos={num_pos}, Neg={num_neg}, Weight={pos_weight.item():.4f}")
    
    # DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False) # Keep chronological split intact within batches if needed, though shuffle=False is safer for time series
    
    # Build Model
    model = LSTMModel(num_features)
    criterion = nn.BCELoss(reduction='mean') # We'll handle weight manually or use pos_weight in BCEWithLogitsLoss
    # For BCELoss, we'll manually apply weight if needed, or just use BCEWithLogitsLoss
    # Let's use BCEWithLogitsLoss for better stability and it supports pos_weight
    
    class LSTMModelLogits(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=1):
            super(LSTMModelLogits, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.dropout = nn.Dropout(0.2)
            self.fc1 = nn.Linear(hidden_size, 32)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(32, 1)
            
        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            out = self.dropout(out)
            out = self.fc1(out)
            out = self.relu(out)
            out = self.fc2(out)
            return out

    model = LSTMModelLogits(num_features)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    print("Training LSTM Model...")
    start_time = time.time()
    epochs = 20
    history_loss = []
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        history_loss.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
            
    train_time = time.time() - start_time
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        
    y_test_np = y_test_raw.astype(int)
    roc_auc = roc_auc_score(y_test_np, probs) if len(np.unique(y_test_np)) > 1 else 0.0
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    best_thresh, best_metrics = evaluate_thresholds(y_test_np, probs, thresholds)
    best_metrics['roc_auc'] = roc_auc
    
    y_pred_best = (probs >= best_thresh).astype(int)
    
    # Save Metrics
    metrics_path = track_file(os.path.join(BASE_DIR, 'metrics', 'lstm_metrics.txt'))
    with open(metrics_path, 'w') as f:
        f.write(f"Best Threshold: {best_thresh}\n")
        for k, v in best_metrics.items():
            f.write(f"{k}: {v}\n")
            
    with open(log_file, 'a') as f:
        f.write(f"Training time: {train_time:.2f}s\n")
        f.write(f"Best Threshold selected: {best_thresh}\n")
        f.write(f"Final F1 Score: {best_metrics['f1_score']:.4f}\n")
        f.write(f"Pos Weight used: {pos_weight.item():.4f}\n")
        
    # Compare with RF
    rf_metrics_path = os.path.join("v3", "improved_run", "metrics", f"window_{window_size}_metrics.txt")
    rf_metrics = {}
    if os.path.exists(rf_metrics_path):
        with open(rf_metrics_path, 'r') as f:
            for line in f:
                if ":" in line:
                    parts = line.strip().split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        try:
                            val = float(parts[1].strip())
                            rf_metrics[key] = val
                        except ValueError:
                            pass
                            
    comp_path = track_file(os.path.join(BASE_DIR, 'metrics', 'comparison_with_rf.txt'))
    with open(comp_path, 'w') as f:
        f.write("=== Comparison With RF (14-day) ===\n\n")
        if rf_metrics:
            rf_f1 = rf_metrics.get('f1_score', 0)
            rf_prec = rf_metrics.get('precision', 0)
            rf_rec = rf_metrics.get('recall', 0)
            lstm_f1 = best_metrics['f1_score']
            lstm_prec = best_metrics['precision']
            lstm_rec = best_metrics['recall']
            
            f.write(f"RF F1-Score:   {rf_f1:.4f} | LSTM F1-Score:   {lstm_f1:.4f}\n")
            f.write(f"RF Precision:  {rf_prec:.4f} | LSTM Precision:  {lstm_prec:.4f}\n")
            f.write(f"RF Recall:     {rf_rec:.4f} | LSTM Recall:     {lstm_rec:.4f}\n\n")
            
            if lstm_f1 > rf_f1:
                f.write("Conclusion: LSTM outperformed the Random Forest model, confirming that deep temporal modeling effectively captures complex sequence patterns.\n")
            else:
                f.write("Conclusion: Random Forest outperformed the LSTM. While LSTM models are powerful for sequences, RF might be more robust to noise or the sequence length might not be sufficient to exploit LSTM's strengths without further tuning.\n")
        else:
            f.write("RF metrics not found for comparison.\n")
            
    # Visualizations
    print("Generating plots...")
    plt.figure(figsize=(15, 5))
    plot_df = df.iloc[split_idx:].copy()
    
    if 'date' in plot_df.columns:
        plot_df['y_pred'] = y_pred_best
        plot_df['date'] = pd.to_datetime(plot_df['date'])
        plot_df = plot_df.sort_values('date')
        
        plt.plot(plot_df['date'], plot_df['target_5d'], label='Actual', alpha=0.7)
        plt.plot(plot_df['date'], plot_df['y_pred'], label='Predicted (LSTM)', alpha=0.7)
        plt.xlabel('Date')
    else:
        plt.plot(y_test_np, label='Actual', alpha=0.7)
        plt.plot(y_pred_best, label='Predicted (LSTM)', alpha=0.7)
        plt.xlabel('Sample')
        
    plt.ylabel('Target 5d')
    plt.title('LSTM Prediction vs Actual (PyTorch)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(track_file(os.path.join(BASE_DIR, 'plots', 'prediction_vs_actual.png')))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), history_loss, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('LSTM Training Loss Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(track_file(os.path.join(BASE_DIR, 'plots', 'training_loss_curve.png')))
    plt.close()

    # Save Model
    model_path = track_file(os.path.join(BASE_DIR, 'models', 'lstm_model.pt'))
    torch.save(model.state_dict(), model_path)
    
    # Final Validation
    print("\n--- Final Validation ---")
    print(f"Verifying all outputs exist inside '{BASE_DIR}/'...")
    validation_failed = False
    
    print("List of generated files:")
    for f_path in generated_files:
        print(f" - {f_path}")
        abs_base = os.path.abspath(BASE_DIR)
        abs_file = os.path.abspath(f_path)
        if not abs_file.startswith(abs_base):
            print(f"ERROR: File {f_path} is located outside {BASE_DIR}!")
            validation_failed = True
            
    if validation_failed:
        raise RuntimeError(f"Validation Failed! Some files were saved outside {BASE_DIR}/")
    else:
        print(f"\nLSTM Pipeline completed successfully. All outputs are in {BASE_DIR}/")

if __name__ == "__main__":
    main()
