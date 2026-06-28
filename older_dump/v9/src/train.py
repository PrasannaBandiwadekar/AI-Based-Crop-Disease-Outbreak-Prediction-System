import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from model_tcn import V9TCNModel

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

class V9FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super(V9FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        return F_loss.mean()

def train_v9():
    print("="*50)
    print("V9 RED ROT PRODUCTION TRAINING (FUSION ARCH)")
    print("="*50)

    # 1. Load Data
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "features.csv"))
    df['date'] = pd.to_datetime(df['date'])
    
    # Define Feature Groups
    WEATHER_FEATURES = [
        'RH2M', 'T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR',
        'RH2M_mean_14', 'RH2M_mean_28', 'T2M_mean_14', 'T2M_mean_28',
        'humidity_streak', 'temp_streak', 'rainfall_streak', 'rainfall_sum_3',
        'T2M_MIN_lag_15', 'RH2M_lag_15', 'RH2M_diff_1', 'RH2M_accel'
    ]
    AGRO_FEATURES = [
        'NDVI', 'NDVI_trend_7', 'variety_susceptibility', 
        'ratoon_flag', 'sanitation_score'
    ]
    
    # Scaling
    scaler_w = StandardScaler()
    df[WEATHER_FEATURES] = scaler_w.fit_transform(df[WEATHER_FEATURES])
    
    scaler_a = StandardScaler()
    df[AGRO_FEATURES] = scaler_a.fit_transform(df[AGRO_FEATURES])
    
    # 2. Sequence Creation (14 days)
    seq_len = 14
    X_weather, X_agro, y, dates = [], [], [], []
    
    for i in range(seq_len, len(df)):
        # Weather Sequence
        X_weather.append(df[WEATHER_FEATURES].iloc[i-seq_len:i].values)
        # Agro state (Current day)
        X_agro.append(df[AGRO_FEATURES].iloc[i-1].values)
        y.append(df['risk_label'].iloc[i-1])
        dates.append(df['date'].iloc[i-1])
        
    X_weather = np.array(X_weather)
    X_agro = np.array(X_agro)
    y = np.array(y)
    
    # 3. Train/Val Split (Chronological)
    split_idx = int(len(X_weather) * 0.8)
    X_w_train, X_w_val = X_weather[:split_idx], X_weather[split_idx:]
    X_a_train, X_a_val = X_agro[:split_idx], X_agro[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Dataloaders
    train_ds = TensorDataset(torch.FloatTensor(X_w_train), torch.FloatTensor(X_a_train), torch.FloatTensor(y_train).view(-1, 1))
    
    # Sampler for imbalance
    counts = np.bincount(y_train.astype(int), minlength=2)
    neg, pos = counts[0], counts[1]
    
    if pos == 0:
        print("WARNING: No positive labels in training set! Weights will be uniform.")
        weights = np.ones_like(y_train) / float(neg)
    else:
        weights = np.where(y_train == 1, 1.0/pos, 1.0/neg)
        
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(weights))
    
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
    val_ds = TensorDataset(torch.FloatTensor(X_w_val), torch.FloatTensor(X_a_val), torch.FloatTensor(y_val).view(-1, 1))
    val_loader = DataLoader(val_ds, batch_size=32)

    # 4. Model & Optimizer
    model = V9TCNModel(len(WEATHER_FEATURES), len(AGRO_FEATURES))
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = V9FocalLoss()
    
    # Indices for penalty logic (from AGRO_FEATURES)
    NDVI_TREND_IDX = 1 # NDVI_trend_7
    VARIETY_IDX = 2    # variety_susceptibility
    
    best_val_loss = float('inf')
    patience = 20
    counter = 0

    print("Starting V9 Training Loop...")
    for epoch in range(100):
        model.train()
        total_loss = 0
        for b_w, b_a, b_y in train_loader:
            optimizer.zero_grad()
            logits, attn = model(b_w, b_a)
            loss = criterion(logits, b_y)
            
            # --- V9 PENALTY SYSTEM (FALSE POSITIVE CONTROL) ---
            probs = torch.sigmoid(logits)
            
            # 1. NDVI Stability Penalty (Suppress risk if NDVI is rising/stable)
            # (Note: Scaled NDVI_trend_7)
            ndvi_trend = b_a[:, NDVI_TREND_IDX]
            # If trend > 0.5 (scaled) and risk is high, penalize
            ndvi_penalty = torch.where((ndvi_trend > 0.5) & (probs > 0.4), 2.0 * probs, torch.zeros_like(probs)).mean()
            loss += ndvi_penalty
            
            # 2. Variety Sensitivity Penalty (Suppress risk if variety is resistant)
            variety = b_a[:, VARIETY_IDX]
            # variety is scaled, but we assume lower is more resistant in simulation
            variety_penalty = torch.where((variety < -0.5) & (probs > 0.5), 1.0 * probs, torch.zeros_like(probs)).mean()
            loss += variety_penalty

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for b_w, b_a, b_y in val_loader:
                logits, _ = model(b_w, b_a)
                val_loss += criterion(logits, b_y).item()
        
        avg_val_loss = val_loss / len(val_loader)
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "v9_tcn_corrected.pth"))
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping.")
                break

    print("V9 Training Complete. Model saved.")

if __name__ == "__main__":
    train_v9()
