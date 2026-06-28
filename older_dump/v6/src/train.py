"""
V6 Training Pipeline (PyTorch Version)

Fixes all V5 training failures:
1. Uses TRUE outbreak-onset labels (not proxy env_interaction_score)
2. Uses 14-day sequence windows (not single-day scalar features)
3. Uses GRU architecture (not tabular HistGradientBoosting)
4. Selects threshold on validation EVENT-LEVEL recall (not daily F1)
5. Strict chronological split -- no shuffling
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    MODEL_PATH, THRESHOLD_PATH, OUTPUTS_DIR, LOGS_DIR,
    BATCH_SIZE, MAX_EPOCHS, PATIENCE, SEQUENCE_FEATURES, SEQUENCE_LENGTH, TARGET,
    LEARNING_RATE
)
from utils import load_raw_data
from label_engine import (
    load_outbreak_events, build_onset_labels, build_biological_labels, validate_label_alignment
)
from sequence_builder import build_daily_features, build_sequences, chronological_split
from model import build_gru_model, compute_class_weights


def select_threshold_on_event_recall(model, X_val_tensor, y_val, val_dates, peak_dates,
                                     thresholds=None):
    """
    Select the decision threshold that maximizes event-level recall on the
    validation set -- NOT daily F1. This is the correct biological metric.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.025)

    model.eval()
    with torch.no_grad():
        val_probs = model(X_val_tensor).squeeze().cpu().numpy()

    date_prob = dict(zip(val_dates, val_probs))
    best_thresh = 0.5
    best_recall = 0.0
    best_fpr    = 1.0
    results = []

    for t in thresholds:
        alert_dates = set(d for d, p in date_prob.items() if p >= t)
        detected = 0
        for peak in peak_dates:
            win_start = peak - pd.Timedelta(days=7)
            win_end   = peak
            if any(win_start <= d <= win_end for d in alert_dates):
                detected += 1

        total_outbreaks = len(peak_dates)
        recall = detected / total_outbreaks if total_outbreaks > 0 else 0

        preds = (val_probs >= t).astype(int)
        fp = ((preds == 1) & (y_val == 0)).sum()
        tn = ((preds == 0) & (y_val == 0)).sum()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        results.append({"threshold": t, "recall": recall, "fpr": fpr})

        if recall > best_recall or (recall == best_recall and fpr < best_fpr):
            best_recall = recall
            best_thresh = t
            best_fpr    = fpr

    print(f"\n[Train] Best threshold: {best_thresh:.3f}")
    print(f"[Train] Val event-recall: {best_recall:.2%}  |  Val FPR: {best_fpr:.2%}")
    pd.DataFrame(results).to_csv(
        os.path.join(OUTPUTS_DIR, "v6_threshold_sweep.csv"), index=False
    )
    return float(best_thresh), float(best_recall)


def run_backtest(model, threshold, feature_df, labeled_df):
    """
    Run inference over the full date range to produce daily risk scores
    """
    print("\n[Train] Running full backtest...")
    X_all, y_all, dates_all = build_sequences(feature_df, labeled_df)
    
    X_tensor = torch.tensor(X_all, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        probs = model(X_tensor).squeeze().cpu().numpy()
        
    alerts = probs >= threshold

    results = pd.DataFrame({
        "date": dates_all,
        "risk_score": probs,
        "risk_band": pd.cut(
            probs,
            bins=[0, 0.30, 0.60, 0.80, 1.01],
            labels=["LOW", "MODERATE", "HIGH", "EXTREME"],
            right=False
        ),
        "alert": alerts
    })

    out_path = os.path.join(OUTPUTS_DIR, "v6_backtest_results.csv")
    results.to_csv(out_path, index=False)
    print(f"[Train] Backtest results saved: {out_path}")
    print(f"[Train] Total alerts: {alerts.sum()} / {len(alerts)} days "
          f"({alerts.mean()*100:.1f}%)")
    return results


def train_model(model, X_train, y_train, X_val, y_val, class_weights, epochs, lr, patience, model_save_path):
    device = torch.device("cpu")
    model.to(device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # We use BCELoss. Apply sample weights.
    criterion = nn.BCELoss(reduction='none')

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_auc = 0.0
    patience_counter = 0

    print(f"Starting training loop for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            
            loss = criterion(preds, batch_y)
            # Apply weights: weight_pos for y=1, weight_neg for y=0
            weights = torch.where(batch_y == 1, torch.tensor(class_weights[1]), torch.tensor(class_weights[0]))
            loss = (loss * weights).mean()
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_X)
            
        train_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t)
            val_weights = torch.where(y_val_t == 1, torch.tensor(class_weights[1]), torch.tensor(class_weights[0]))
            val_loss = (val_loss * val_weights).mean().item()
            
            y_val_np = y_val_t.cpu().numpy().flatten()
            val_preds_np = val_preds.cpu().numpy().flatten()
            
            if sum(y_val_np) > 0:
                val_auc = average_precision_score(y_val_np, val_preds_np)
            else:
                val_auc = 0.0

        print(f"Epoch {epoch+1:02d}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_pr_auc: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            # Save the state dict
            # Make sure the directory exists
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print("  --> Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best weights
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
    return model, best_val_auc


def run_v6_training():
    print("=" * 60, flush=True)
    print("V6 Red Rot Outbreak Forecasting System -- Training Pipeline (PyTorch)", flush=True)
    print("=" * 60, flush=True)

    print("[Train] Loading raw weather data...", flush=True)
    raw_df = load_raw_data()
    raw_df["date"] = pd.to_datetime(raw_df["date"])

    print("\n[Train] Engineering sequence features...", flush=True)
    feature_df = build_daily_features(raw_df)
    print("[Train] Feature engineering complete.", flush=True)

    # ── Step 2: Phase 1 - Pretraining on Biological Seasonality ──────────────
    print("\n" + "-" * 40, flush=True)
    print("PHASE 1: Pretraining on Biological Labels (2005-2018)", flush=True)
    print("-" * 40, flush=True)
    
    bio_labeled_df = build_biological_labels(raw_df)
    print("[Train] Building sequences...", flush=True)
    X_bio, y_bio, dates_bio = build_sequences(feature_df, bio_labeled_df)
    print("[Train] Splitting data...", flush=True)
    
    splits_bio = chronological_split(X_bio, y_bio, dates_bio)
    X_train_bio, y_train_bio, _ = splits_bio["train"]
    X_val_bio,   y_val_bio,   _ = splits_bio["val"]

    print(f"[Train] Building GRU model for pretraining...", flush=True)
    model = build_gru_model(n_features=len(SEQUENCE_FEATURES))
    
    class_weights_bio = compute_class_weights(y_train_bio)
    pretrained_path = MODEL_PATH.replace(".keras", "_pretrained.pth")

    print(f"[Train] Phase 1 Training: {len(X_train_bio)} samples...")
    model, _ = train_model(
        model, X_train_bio, y_train_bio, X_val_bio, y_val_bio,
        class_weights=class_weights_bio,
        epochs=MAX_EPOCHS // 2,
        lr=LEARNING_RATE,
        patience=PATIENCE,
        model_save_path=pretrained_path
    )

    # ── Step 3: Phase 2 - Fine-tuning on Confirmed Onset Labels ──────────────
    print("\n" + "-" * 40)
    print("PHASE 2: Fine-tuning on Confirmed Onset Labels (2019-2021)")
    print("-" * 40)
    
    peaks = load_outbreak_events()
    onset_labeled_df = build_onset_labels(raw_df, peaks)
    X_onset, y_onset, dates_onset = build_sequences(feature_df, onset_labeled_df)
    
    mask_ft_train = [d.year < 2021 for d in dates_onset if d.year >= 2019]
    mask_ft_val   = [d.year == 2021 for d in dates_onset if d.year >= 2019]
    
    X_ft_pool = X_onset[[d.year >= 2019 for d in dates_onset]]
    y_ft_pool = y_onset[[d.year >= 2019 for d in dates_onset]]
    dates_ft_pool = dates_onset[[d.year >= 2019 for d in dates_onset]]
    
    X_train_ft = X_ft_pool[mask_ft_train]
    y_train_ft = y_ft_pool[mask_ft_train]
    
    X_val_ft   = X_ft_pool[mask_ft_val]
    y_val_ft   = y_ft_pool[mask_ft_val]
    dates_val_ft = dates_ft_pool[mask_ft_val]

    print(f"[Train] Phase 2 Fine-tuning: {len(X_train_ft)} samples...")
    class_weights_ft = compute_class_weights(y_train_ft)
    model_ft_path = MODEL_PATH.replace(".keras", ".pth")

    model, best_val_auc = train_model(
        model, X_train_ft, y_train_ft, X_val_ft, y_val_ft,
        class_weights=class_weights_ft,
        epochs=MAX_EPOCHS,
        lr=LEARNING_RATE / 5,
        patience=PATIENCE,
        model_save_path=model_ft_path
    )

    # ── Step 4: Select Threshold on Event-Level Recall ─────────────────────
    print("\n[Train] Selecting threshold on validation (2021) EVENT-LEVEL recall...")
    peaks_2021 = peaks[peaks.dt.year == 2021]
    X_val_ft_t = torch.tensor(X_val_ft, dtype=torch.float32)
    threshold, val_recall = select_threshold_on_event_recall(
        model, X_val_ft_t, y_val_ft, dates_val_ft, peaks_2021
    )

    with open(THRESHOLD_PATH, "w") as f:
        f.write(str(threshold))
    print(f"[Train] Threshold saved: {THRESHOLD_PATH}")

    # ── Step 5: Full Backtest ──────────────────────────────────────────────
    run_backtest(model, threshold, feature_df, onset_labeled_df)

    # ── Step 6: Training Summary ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("V6 TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best Val PR-AUC:      {best_val_auc:.4f}")
    print(f"  Best Val Recall:      {val_recall:.2%}")
    print(f"  Decision Threshold:   {threshold:.4f}")
    print(f"  Model saved:          {model_ft_path}")
    print(f"\n  Next step: Run frozen evaluator:")
    print(f"  python v5/evaluation/final_v5_evaluator.py")
    print(f"  (Point it to v6/outputs/v6_backtest_results.csv)")
    print("=" * 60)


if __name__ == "__main__":
    run_v6_training()
