"""
V6 Model Architecture — GRU with Temporal Feature Encoding (PyTorch Version)

Architecture Choice: GRU (not pure XGBoost)
Reason: Red Rot onset is driven by ORDERED multi-day biological phase progression:
    Phase 1: Rising humidity trend
    Phase 2: Sustained saturation
    Phase 3: Rainfall spike
    Phase 4: Temperature stabilization

A tabular model (V5 XGBoost) collapses this to a single row and loses ordering.
The GRU's hidden state carries phase information forward through the sequence.

Architecture:
    Input: (batch, 14, n_features)
    → Bidirectional GRU (captures both forward and backward context in window)
    → Dropout (regularization — critical with ~7k samples)
    → Dense (32) + ReLU
    → Dense (1) + Sigmoid → outbreak onset probability
"""

import os
import numpy as np
import torch
import torch.nn as nn

class V6GRUModel(nn.Module):
    def __init__(self, n_features, gru_units=64, dropout_rate=0.3):
        super(V6GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=gru_units,
            batch_first=True,
            bidirectional=True
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense = nn.Linear(gru_units * 2, 32)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate / 2)
        self.out = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        _, h_n = self.gru(x)
        # h_n shape: (num_directions, batch_size, hidden_size)
        # Concatenate final forward and backward hidden states
        x = torch.cat((h_n[0, :, :], h_n[1, :, :]), dim=1)
        
        x = self.dropout1(x)
        x = self.relu(self.dense(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.out(x))
        return x

def build_gru_model(n_features: int, gru_units: int = 64, dropout_rate: float = 0.3):
    """
    Builds the V6 GRU model for outbreak onset prediction.

    Args:
        n_features:    Number of features per time step.
        gru_units:     Number of GRU hidden units.
        dropout_rate:  Dropout rate for regularization.

    Returns:
        PyTorch model instance.
    """
    model = V6GRUModel(n_features=n_features, gru_units=gru_units, dropout_rate=dropout_rate)
    return model

def compute_class_weights(y_train: np.ndarray) -> dict:
    """
    Compute class weights to handle severe class imbalance.
    With ~19 outbreaks / ~5000 training days, positive class needs heavy upweighting.
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    if n_pos == 0:
        return {0: 1.0, 1: 1.0}
    weight_pos = n_neg / n_pos
    print(f"[Model] Class weights: neg=1.0, pos={weight_pos:.2f}  "
          f"(n_neg={n_neg}, n_pos={n_pos})")
    return {0: 1.0, 1: float(weight_pos)}

