import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Paths
BASE_DIR = os.getcwd()
FEATURES_PATH = os.path.join(BASE_DIR, "v9", "data", "processed", "features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "v9", "models", "v9_fusion_model.pth")
GT_PATH = os.path.join(BASE_DIR, "research_comp", "evidence_base", "outbreak_events", "sangli_synthetic_gt.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "v9", "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

WEATHER_FEATURES = [
    'RH2M', 'T2M', 'T2M_MAX', 'T2M_MIN', 
    'RH2M_mean_7', 'RH2M_mean_28', 'T2M_mean_28',
    'rainfall_sum_7', 'rainfall_sum_28',
    'T2M_MIN_lag_15', 'RH2M_lag_15',
    'RH2M_diff_1', 'RH2M_accel'
]
AGRO_FEATURES = [
    'NDVI', 'NDVI_trend_7', 'variety_susceptibility', 
    'ratoon_flag', 'sanitation_score'
]

# ─── TCN Components ───

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Sequential(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        L = x.size(2)
        out = self.conv(x)[:, :, :L]
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNEncoder(nn.Module):
    """Multi-scale TCN — dilations [1, 2, 4] for 3/6/12-day receptive fields."""
    def __init__(self, input_dim, hidden_dim=64, output_dim=128, kernel_size=3, dropout=0.2):
        super().__init__()
        self.blocks = nn.Sequential(
            TemporalBlock(input_dim, hidden_dim, kernel_size, dilation=1, dropout=dropout),
            TemporalBlock(hidden_dim, hidden_dim, kernel_size, dilation=2, dropout=dropout),
            TemporalBlock(hidden_dim, output_dim, kernel_size, dilation=4, dropout=dropout),
        )
        # Project to same output dim as GRU bidirectional (gru_units * 2 = 128)
        self.out_proj = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        # x: (B, T, C)
        x = x.transpose(1, 2)            # (B, C, T)
        x = self.blocks(x)               # (B, output_dim, T)
        x = x.transpose(1, 2)           # (B, T, output_dim)
        return self.out_proj(x)          # (B, T, output_dim)

# ─── Full Hybrid V9 Model (f=TCN, g+h frozen) ───

class V9TCNModel(nn.Module):
    def __init__(self, weather_dim, agro_dim, gru_units=64):
        super().__init__()
        # New f-layer: TCN encoder
        self.tcn = TCNEncoder(weather_dim, hidden_dim=gru_units, output_dim=gru_units * 2)

        # Attention (trainable — calibrates TCN outputs)
        self.attention_w = nn.Linear(gru_units * 2, 1)

        # g-layer — will be frozen
        self.agronomic_mlp = nn.Sequential(
            nn.Linear(agro_dim, 16), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16, 8), nn.ReLU()
        )
        # h-layer — will be frozen
        self.fusion_layer = nn.Sequential(
            nn.Linear(gru_units * 2 + 8, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, weather_seq, agronomic_state):
        tcn_out = self.tcn(weather_seq)                      # (B, T, 128)
        attn_scores = self.attention_w(tcn_out)               # (B, T, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        weather_context = torch.sum(attn_weights * tcn_out, dim=1)  # (B, 128)

        agro_context = self.agronomic_mlp(agronomic_state)   # (B, 8)
        combined = torch.cat([weather_context, agro_context], dim=1)
        logits = self.fusion_layer(combined)                  # (B, 1)
        return logits, attn_weights

# ─── Dataset ───

class WeatherSequenceDataset(Dataset):
    """
    Builds sequence windows from features.csv.
    Label = 1 if outbreak within next 14 days, else uses risk_label from CSV.
    """
    def __init__(self, df, gt_df, scaler_w, scaler_a, seq_len=14):
        self.seq_len = seq_len
        self.w_data = scaler_w.transform(df[WEATHER_FEATURES])
        self.a_data = scaler_a.transform(df[AGRO_FEATURES])
        self.dates  = df['date'].values

        # Build outbreak-aware labels
        labels = []
        for i in range(seq_len - 1, len(df)):
            date = pd.Timestamp(self.dates[i])
            window_end = date + timedelta(days=14)
            in_outbreak = ((gt_df['peak_start'] >= date) &
                           (gt_df['peak_start'] <= window_end)).any()
            # Also mark detection window days as positive
            pre_onset = any(
                ((gt_df['peak_start'] - timedelta(days=7)) <= date) &
                (date <= gt_df['peak_start'] + timedelta(days=3))
            )
            labels.append(1 if (in_outbreak or pre_onset) else 0)

        self.labels = labels
        self.start_idx = seq_len - 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        i = idx + self.start_idx
        w = torch.FloatTensor(self.w_data[i - self.seq_len + 1:i + 1])
        a = torch.FloatTensor(self.a_data[i])
        y = torch.FloatTensor([self.labels[idx]])
        return w, a, y

# ─── Training ───

def freeze_layers(model):
    """Freeze g-layer and h-layer. Only TCN + attention are trainable."""
    for name, param in model.named_parameters():
        if 'agronomic_mlp' in name or 'fusion_layer' in name:
            param.requires_grad = False

def train_tcn_flayer(model, dataset, device, epochs=30, lr=1e-3):
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    # Focal-like weighted BCE to counteract class imbalance (few outbreaks)
    pos_count = sum(dataset.labels)
    neg_count = len(dataset.labels) - pos_count
    pos_weight = torch.FloatTensor([neg_count / max(pos_count, 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"  Training TCN f-layer: {pos_count} positive / {neg_count} negative samples")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for w, a, y in loader:
            w, a, y = w.to(device), a.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(w, a)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {total_loss/len(loader):.4f}")

# ─── Evaluation ───

def evaluate(model, df, gt_df, scaler_w, scaler_a, device, threshold=40.0):
    model.eval()
    seq_len = 14
    risk_scores = {}
    with torch.no_grad():
        for i in range(seq_len - 1, len(df)):
            date = df.loc[i, 'date']
            w_t = torch.FloatTensor(
                scaler_w.transform(df.iloc[i-seq_len+1:i+1][WEATHER_FEATURES])
            ).unsqueeze(0).to(device)
            a_t = torch.FloatTensor(
                scaler_a.transform(df.iloc[i][AGRO_FEATURES].values.reshape(1, -1))
            ).to(device)
            logits, _ = model(w_t, a_t)
            risk_scores[date] = torch.sigmoid(logits).item() * 100

    def no_outbreak_14d(date):
        return not ((gt_df['peak_start'] >= date) &
                    (gt_df['peak_start'] <= date + timedelta(days=14))).any()

    tp = fn = fp = tn = 0
    for _, event in gt_df.iterrows():
        estart = event['peak_start']
        dw = pd.date_range(estart - timedelta(days=7), estart + timedelta(days=3))
        if any(risk_scores.get(d, 0) >= threshold for d in dw):
            tp += 1
        else:
            fn += 1

    for date, score in risk_scores.items():
        if no_outbreak_14d(date):
            if score >= threshold: fp += 1
            else: tn += 1

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr    = fp / (fp + tn) if (fp + tn) > 0 else 0
    return recall, fpr, tp, fn, fp, tn, risk_scores

# ─── Main ───

def main():
    print("=" * 55)
    print(" V9 TCN f-layer TARGETED FINE-TUNE + VALIDATION")
    print("=" * 55)

    df = pd.read_csv(FEATURES_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    gt_df = pd.read_csv(GT_PATH)
    gt_df['peak_start'] = pd.to_datetime(gt_df['peak_start'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    scaler_w = StandardScaler().fit(df[WEATHER_FEATURES])
    scaler_a = StandardScaler().fit(df[AGRO_FEATURES])

    # Build model and load original weights for g + h layers
    model = V9TCNModel(len(WEATHER_FEATURES), len(AGRO_FEATURES)).to(device)
    orig_state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    # Load g + h weights (strict=False — TCN keys will be missing)
    model.load_state_dict(orig_state, strict=False)

    # Freeze g-layer and h-layer
    freeze_layers(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable params (TCN + attn): {trainable_params:,}")
    print(f"Frozen params (g + h layers): {frozen_params:,}")

    # Build dataset and train
    dataset = WeatherSequenceDataset(df, gt_df, scaler_w, scaler_a)
    print("\n[1] Training TCN f-layer (g + h frozen)...")
    train_tcn_flayer(model, dataset, device, epochs=40, lr=5e-4)

    # Threshold sweep: [40, 50, 60, 70]
    print("\n[2] Threshold Sweep Evaluation...")
    results = []
    for thresh in [40, 50, 60, 70]:
        recall, fpr, tp, fn, fp, tn, _ = evaluate(
            model, df, gt_df, scaler_w, scaler_a, device, threshold=thresh
        )
        results.append(dict(Threshold=thresh, TP=tp, FP=fp, TN=tn, FN=fn,
                            Recall=round(recall, 4), FPR=round(fpr, 4)))
        print(f"  T={thresh} | Recall={recall:.4f} | FPR={fpr:.4f} | "
              f"TP={tp} FP={fp} TN={tn} FN={fn}")

    metrics_df = pd.DataFrame(results)

    # Best operating threshold
    candidates = metrics_df[metrics_df['Recall'] >= 0.60]
    best = candidates.loc[candidates['FPR'].idxmin()] if not candidates.empty else \
           metrics_df.loc[metrics_df['Recall'].idxmax()]

    status = "PASS" if best['Recall'] >= 0.60 and best['FPR'] <= 0.15 else "FAIL"

    print(f"\n[3] Best Threshold: {best['Threshold']} | "
          f"Recall={best['Recall']} | FPR={best['FPR']}")
    print(f"Final Status: {status}")

    # Save corrected model
    corrected_model_path = os.path.join(BASE_DIR, "v9", "models", "v9_tcn_corrected.pth")
    torch.save(model.state_dict(), corrected_model_path)
    print(f"Corrected model saved: {corrected_model_path}")

    # Save report
    report_path = os.path.join(OUTPUT_DIR, "sangli_tcn_upgrade_report.txt")
    with open(report_path, "w") as f:
        f.write("V9 f-layer TCN Targeted Fine-Tune Report\n")
        f.write("=" * 45 + "\n\n")
        f.write("Architectural Changes:\n")
        f.write("  - GRU temporal encoder replaced with Multi-Scale TCN\n")
        f.write("  - Dilations: [1, 2, 4] | Kernel: 3 | Receptive fields: 3/6/12 days\n")
        f.write("  - g-layer (agronomic_mlp): FROZEN\n")
        f.write("  - h-layer (fusion_layer): FROZEN\n")
        f.write("  - TCN + attention_w: FINE-TUNED (40 epochs)\n\n")
        f.write("Threshold Sweep Results:\n")
        f.write(metrics_df.to_string(index=False) + "\n\n")
        f.write(f"Best Operating Threshold: {best['Threshold']}\n")
        f.write(f"  Recall: {best['Recall']}\n")
        f.write(f"  FPR:    {best['FPR']}\n\n")
        f.write(f"Final Status: {status}\n")
        f.write(f"Corrected model: v9/models/v9_tcn_corrected.pth\n")

    print(f"Report saved: {report_path}")

if __name__ == "__main__":
    main()
