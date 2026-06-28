import torch
import torch.nn as nn
import numpy as np

class V8Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(V8Attention, self).__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, out):
        # out: (batch, seq_len, hidden_dim)
        attn_logits = self.attn_net(out)
        attn_weights = torch.softmax(attn_logits, dim=1)
        # Sequence-to-sequence style weighting
        weighted_out = out * attn_weights
        return weighted_out, attn_weights

class V8GRUModel(nn.Module):
    def __init__(self, n_features, gru_units=64, dropout_rate=0.3):
        super(V8GRUModel, self).__init__()
        
        # We will add 4 temporal diff features on the fly: 
        # RH_diff1, RH_diff3, Rain_diff1, Humidity_accel
        self.gru = nn.GRU(
            input_size=n_features + 4, 
            hidden_size=gru_units,
            batch_first=True,
            bidirectional=True
        )
        
        self.attention = V8Attention(gru_units * 2)
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense_seq = nn.Linear(gru_units * 2, 1) # Predict for each step
        
    def forward(self, x):
        # x: (batch, seq_len, n_features)
        batch_size, seq_len, n_feats = x.shape
        
        # 1. Temporal Difference Features (Input Level)
        # RH2M = idx 0, PRECTOTCORR = idx 1
        rh = x[:, :, 0:1]
        rain = x[:, :, 1:2]
        
        rh_diff1 = rh - torch.roll(rh, shifts=1, dims=1)
        rh_diff3 = rh - torch.roll(rh, shifts=3, dims=1)
        rain_diff1 = rain - torch.roll(rain, shifts=1, dims=1)
        
        # Zero out rolled values for first indices
        rh_diff1[:, 0, :] = 0
        rh_diff3[:, :3, :] = 0
        rain_diff1[:, 0, :] = 0
        
        humidity_accel = rh_diff3 - rh_diff1
        
        # Concatenate diffs to input
        x_aug = torch.cat([x, rh_diff1, rh_diff3, rain_diff1, humidity_accel], dim=2)
        
        # 2. GRU Processing
        gru_out, _ = self.gru(x_aug)
        # gru_out: (batch, seq_len, gru_units * 2)
        
        # 3. Temporal Attention
        weighted_out, attn_weights = self.attention(gru_out)
        
        # 4. Sequence Output (Predict risk for each timestep)
        x = self.dropout1(weighted_out)
        seq_logits = self.dense_seq(x) # (batch, seq_len, 1)
        
        # 5. Final Output: Max of sequence (When risk emerges)
        final_logits, _ = torch.max(seq_logits, dim=1)
        
        return final_logits, attn_weights

def compute_class_weights(y_train):
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    if n_pos == 0: return {0: 1.0, 1: 5.0}
    weight_pos = n_neg / n_pos
    return {0: 1.0, 1: float(weight_pos)}
