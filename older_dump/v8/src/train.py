import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import average_precision_score

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

from model import V8GRUModel

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        return F_loss.mean()

def train_v8():
    print("="*50)
    print("V8 Red Rot Training Pipeline (UPGRADED ARCH)")
    print("="*50)

    # Load data
    data_path = os.path.join(PROCESSED_DIR, "sequences.npz")
    data = np.load(data_path, allow_pickle=True)
    X, y, dates = data["X"], data["y"], data["dates"]
    dates = pd.to_datetime(dates)

    # Indices
    STREAK_IDX = 20
    SPIKE_IDX = 21

    print(f"Loaded {len(X)} samples.")

    # Chronological Split
    train_mask = dates.year < 2019
    val_mask = dates.year >= 2019

    X_train, y_train = X[train_mask], y[train_mask]
    y_train = y_train.astype(np.float32)
    X_val, y_val = X[val_mask], y[val_mask]
    y_val = y_val.astype(np.float32)

    # Weighted Sampler
    class_counts = np.bincount(y_train.astype(int))
    weights = 1.0 / class_counts
    sample_weights = np.array([weights[int(t)] for t in y_train])
    sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), len(sample_weights))

    # Model
    model = V8GRUModel(n_features=X.shape[2])
    device = torch.device("cpu")
    model.to(device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, sampler=sampler)
    
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    best_fpr = 101.0
    patience = 25
    trigger_times = 0
    model_path = os.path.join(OUTPUTS_DIR, "v8_model.pth")
    
    # Track hard negatives
    hard_neg_indices = []

    for epoch in range(150):
        model.train()
        total_loss = 0
        
        # 5. HARD NEGATIVE MINING (Injecting previously identified FPs)
        if len(hard_neg_indices) > 0:
            hn_X = X_train_t[hard_neg_indices]
            hn_y = y_train_t[hard_neg_indices]
            optimizer.zero_grad()
            hn_logits, _ = model(hn_X)
            hn_loss = criterion(hn_logits, hn_y) * 5.0 # Increased weight
            hn_loss.backward()
            optimizer.step()

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward: model returns final_logits and attn_weights
            # BUT we need the sequence logits for the penalty!
            # Let's modify the model to return sequence logits too.
            # (I'll do that in model.py next)
            logits, _ = model(batch_X)
            loss = criterion(logits, batch_y)
            
            # TRANSITION AWARE LOSS (Sequence-level Plateau Penalty)
            # Apply to last day of sequence for now as proxy
            streak = batch_X[:, -1, STREAK_IDX]
            spike = batch_X[:, -1, SPIKE_IDX]
            plateau_mask = (streak >= 8) & (spike == 0)
            
            if plateau_mask.any():
                probs = torch.sigmoid(logits[plateau_mask])
                penalty = torch.where(probs > 0.4, 2.0 * probs, torch.zeros_like(probs)).mean()
                loss += penalty

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Eval & Hard Negative Identification
        model.eval()
        with torch.no_grad():
            train_logits, _ = model(X_train_t)
            train_probs = torch.sigmoid(train_logits).numpy().flatten()
            # Find top 20% FPs in training set
            fps = (train_probs > 0.5) & (y_train == 0)
            if fps.any():
                fp_probs = train_probs[fps]
                threshold = np.percentile(fp_probs, 80)
                hard_neg_indices = np.where((train_probs >= threshold) & (y_train == 0))[0]
            
            # Validation Metric (Event-level FPR approximation)
            val_logits, _ = model(X_val_t)
            val_probs = torch.sigmoid(val_logits).numpy().flatten()
            val_alert = val_probs > 0.5
            
            # Since full evaluator is slow, use daily FPR as proxy for early stopping
            val_fpr = (val_alert & (y_val == 0)).sum() / (y_val == 0).sum() if (y_val == 0).sum() > 0 else 0
            val_recall = (val_alert & (y_val == 1)).sum() / (y_val == 1).sum() if (y_val == 1).sum() > 0 else 0
            
        print(f"Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.4f} | Val Rec: {val_recall:.2f} | Val FPR: {val_fpr:.4f}")

        # Early stopping on FPR
        if val_fpr < best_fpr and val_recall >= 0.33: # Ensure some recall
            best_fpr = val_fpr
            torch.save(model.state_dict(), model_path)
            print(f"  --> Model Saved (Best FPR: {val_fpr:.4f})")
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Final Backtest
    print("\nRunning Final Upgraded Backtest...")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        all_X_t = torch.tensor(X, dtype=torch.float32)
        all_logits, _ = model(all_X_t)
        all_probs = torch.sigmoid(all_logits).numpy().flatten()

    results = pd.DataFrame({
        "date": dates,
        "risk_score": all_probs,
        "alert": (all_probs >= 0.5)
    })
    results.to_csv(os.path.join(OUTPUTS_DIR, "v8_backtest_results.csv"), index=False)
    print(f"Backtest results saved to v8/outputs/v8_backtest_results.csv")

if __name__ == "__main__":
    train_v8()
