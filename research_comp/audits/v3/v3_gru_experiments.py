import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = "v3/gru_experiments"

def setup_folders():
    for folder in ["models", "metrics", "logs", "reports"]:
        os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)
    print("Folders created successfully.")

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.0):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, h_n = self.gru(x)
        out = self.dropout(h_n[-1])
        out = self.fc(out)
        return self.sigmoid(out)

def evaluate_thresholds(y_true, probs, thresholds):
    best_f1 = -1
    best_thresh = 0.5
    best_metrics = {}
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
    roc_auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.0
    best_metrics["roc_auc"] = roc_auc
    best_metrics["threshold"] = best_thresh
    return best_metrics

def train_and_evaluate(X_train, y_train, X_test, y_test, hidden_size, num_layers, dropout, lr, epochs, seed, num_features):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=False)

    model = GRUModel(num_features, hidden_size, num_layers, dropout)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    losses = []
    start = time.time()
    for epoch in range(epochs):
        epoch_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))
    train_time = time.time() - start

    model.eval()
    with torch.no_grad():
        probs = model(X_test_t).numpy().flatten()

    metrics = evaluate_thresholds(y_test, probs, [0.1, 0.2, 0.3, 0.4, 0.5])
    return model, metrics, train_time, losses

def main():
    setup_folders()

    # Load data
    df = pd.read_csv("v3/data/seq_window_14.csv")
    feature_cols = [c for c in df.columns if "_lag_" in c]
    X = df[feature_cols].values
    y = df["target_5d"].values.astype(float)

    split = int(len(X) * 0.7)
    X_train_raw, X_test_raw = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    num_features = len(feature_cols) // 14
    X_train = X_train_scaled.reshape(-1, 14, num_features)
    X_test = X_test_scaled.reshape(-1, 14, num_features)

    # Define 10 experiment configurations
    experiments = [
        {"id": 1,  "hidden": 32,  "layers": 1, "dropout": 0.0, "lr": 0.001,  "epochs": 15},
        {"id": 2,  "hidden": 64,  "layers": 1, "dropout": 0.0, "lr": 0.001,  "epochs": 15},
        {"id": 3,  "hidden": 128, "layers": 1, "dropout": 0.0, "lr": 0.001,  "epochs": 15},
        {"id": 4,  "hidden": 64,  "layers": 2, "dropout": 0.2, "lr": 0.001,  "epochs": 20},
        {"id": 5,  "hidden": 128, "layers": 2, "dropout": 0.2, "lr": 0.001,  "epochs": 20},
        {"id": 6,  "hidden": 64,  "layers": 1, "dropout": 0.2, "lr": 0.0005, "epochs": 25},
        {"id": 7,  "hidden": 128, "layers": 1, "dropout": 0.2, "lr": 0.0005, "epochs": 25},
        {"id": 8,  "hidden": 32,  "layers": 2, "dropout": 0.0, "lr": 0.0005, "epochs": 25},
        {"id": 9,  "hidden": 64,  "layers": 1, "dropout": 0.0, "lr": 0.0005, "epochs": 15},
        {"id": 10, "hidden": 128, "layers": 2, "dropout": 0.0, "lr": 0.001,  "epochs": 25},
    ]

    all_results = []
    log_lines = []

    for exp in experiments:
        eid = exp["id"]
        print(f"\n--- Experiment {eid} ---")
        print(f"  Hidden={exp['hidden']}, Layers={exp['layers']}, Dropout={exp['dropout']}, LR={exp['lr']}, Epochs={exp['epochs']}")

        model, metrics, t_time, losses = train_and_evaluate(
            X_train, y_train, X_test, y_test,
            exp["hidden"], exp["layers"], exp["dropout"], exp["lr"], exp["epochs"],
            seed=42, num_features=num_features
        )

        result = {**exp, **metrics, "train_time": t_time}
        all_results.append(result)

        # Save per-experiment metrics
        with open(os.path.join(BASE_DIR, f"metrics/experiment_{eid}.txt"), "w", encoding="utf-8") as f:
            f.write(f"Hidden: {exp['hidden']}\n")
            f.write(f"Layers: {exp['layers']}\n")
            f.write(f"Dropout: {exp['dropout']}\n")
            f.write(f"LR: {exp['lr']}\n")
            f.write(f"Epochs: {exp['epochs']}\n")
            f.write(f"Best Threshold: {metrics['threshold']}\n")
            for k in ["precision", "recall", "f1_score", "accuracy", "roc_auc"]:
                f.write(f"{k}: {metrics[k]:.4f}\n")
            f.write(f"Train Time: {t_time:.2f}s\n")

        line = f"Exp {eid}: H={exp['hidden']} L={exp['layers']} D={exp['dropout']} LR={exp['lr']} E={exp['epochs']} -> F1={metrics['f1_score']:.4f} AUC={metrics['roc_auc']:.4f} T={t_time:.1f}s"
        log_lines.append(line)
        print(f"  F1={metrics['f1_score']:.4f}, AUC={metrics['roc_auc']:.4f}")

    # Find best experiment
    best = max(all_results, key=lambda r: r["f1_score"])
    print(f"\nBest Experiment: #{best['id']} with F1={best['f1_score']:.4f}")

    # Save best model report
    with open(os.path.join(BASE_DIR, "reports/best_gru_model.txt"), "w", encoding="utf-8") as f:
        f.write("=== Best GRU Model ===\n\n")
        f.write(f"Experiment ID: {best['id']}\n")
        f.write(f"Hidden: {best['hidden']}\n")
        f.write(f"Layers: {best['layers']}\n")
        f.write(f"Dropout: {best['dropout']}\n")
        f.write(f"LR: {best['lr']}\n")
        f.write(f"Epochs: {best['epochs']}\n")
        f.write(f"Best Threshold: {best['threshold']}\n")
        f.write(f"F1-Score: {best['f1_score']:.4f}\n")
        f.write(f"Precision: {best['precision']:.4f}\n")
        f.write(f"Recall: {best['recall']:.4f}\n")
        f.write(f"ROC-AUC: {best['roc_auc']:.4f}\n")

    # Stability check: retrain best config with 3 different seeds
    print("\n--- Stability Check (3 seeds) ---")
    stability_f1s = []
    for seed in [42, 123, 999]:
        _, m, _, _ = train_and_evaluate(
            X_train, y_train, X_test, y_test,
            best["hidden"], best["layers"], best["dropout"], best["lr"], best["epochs"],
            seed=seed, num_features=num_features
        )
        stability_f1s.append(m["f1_score"])
        print(f"  Seed {seed}: F1={m['f1_score']:.4f}")

    mean_f1 = np.mean(stability_f1s)
    std_f1 = np.std(stability_f1s)

    # RF baseline
    rf_f1 = 0.3161
    rf_prec = 0.2480
    rf_rec = 0.4357
    rf_auc = 0.8289

    # Final comparison report
    with open(os.path.join(BASE_DIR, "reports/final_comparison.txt"), "w", encoding="utf-8") as f:
        f.write("=== Final Comparison: Best GRU vs Random Forest ===\n\n")

        f.write("### 1. Best GRU vs RF\n")
        f.write(f"Metric     | RF (14d) | Best GRU (Exp #{best['id']})\n")
        f.write(f"-----------|----------|----------\n")
        f.write(f"F1-Score   | {rf_f1:.4f}   | {best['f1_score']:.4f}\n")
        f.write(f"Precision  | {rf_prec:.4f}   | {best['precision']:.4f}\n")
        f.write(f"Recall     | {rf_rec:.4f}   | {best['recall']:.4f}\n")
        f.write(f"ROC-AUC    | {rf_auc:.4f}   | {best['roc_auc']:.4f}\n\n")

        f.write("### 2. Stability\n")
        f.write(f"GRU F1 across 3 seeds: {stability_f1s}\n")
        f.write(f"Mean F1: {mean_f1:.4f}, Std: {std_f1:.4f}\n")
        is_stable = std_f1 < 0.02
        f.write(f"Stability: {'Consistent' if is_stable else 'Unstable (high variance)'}\n\n")

        f.write("### 3. Final Decision\n")
        if best["f1_score"] > rf_f1 + 0.02 and is_stable:
            decision = "GRU is better -> use GRU"
        elif abs(best["f1_score"] - rf_f1) <= 0.02:
            decision = "Both similar -> prefer simpler model (RF)"
        else:
            decision = "RF is better -> use RF"
        f.write(f"Decision: {decision}\n\n")

        f.write("### Answer:\n")
        if best["f1_score"] > rf_f1 and is_stable:
            f.write("Yes, GRU can reliably outperform Random Forest for this problem.\n")
        else:
            f.write("No, GRU cannot reliably outperform Random Forest for this problem.\n")
            f.write("Random Forest with class balancing and threshold tuning remains the recommended approach.\n")

    # Experiment log
    with open(os.path.join(BASE_DIR, "logs/experiment_log.txt"), "w", encoding="utf-8") as f:
        f.write("=== GRU Hyperparameter Exploration Log ===\n\n")
        for line in log_lines:
            f.write(line + "\n")
        f.write(f"\nBest: Experiment #{best['id']} (F1={best['f1_score']:.4f})\n")
        f.write(f"\nStability (3 seeds): {stability_f1s}\n")
        f.write(f"Mean={mean_f1:.4f}, Std={std_f1:.4f}\n")

    # Save best model weights
    best_model, _, _, _ = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        best["hidden"], best["layers"], best["dropout"], best["lr"], best["epochs"],
        seed=42, num_features=num_features
    )
    torch.save(best_model.state_dict(), os.path.join(BASE_DIR, "models/best_gru_model.pth"))

    print(f"\nPipeline completed. All outputs in {BASE_DIR}/")

if __name__ == "__main__":
    main()
