"""
V9 TCN Model — f-layer (Environmental Pressure Model) replacement.
Exact architecture used in v9_correction.py targeted fine-tuning.
g-layer (agronomic_mlp) and h-layer (fusion_layer) are structurally
identical to V9FusionModel so weights load seamlessly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()

        self.padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               padding=0, dilation=dilation)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               padding=0, dilation=dilation)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

    def forward(self, x):
        x_padded = F.pad(x, (self.padding, 0))

        out = self.conv1(x_padded)
        out = self.relu(out)
        out = self.dropout(out)

        out = F.pad(out, (self.padding, 0))
        out = self.conv2(out)
        out = self.relu(out)

        # 🔴 CRITICAL: trim to original length
        out = out[:, :, -x.size(2):]

        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)


class TCNEncoder(nn.Module):
    """
    Multi-scale Temporal Convolutional Network.
    Dilations [1, 2, 4] — receptive fields: 3 / 6 / 12 days.
    Output shape: (batch, seq_len, gru_units*2) — compatible with attention_w.
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=128,
                 kernel_size=3, dropout=0.2):
        super().__init__()
        self.blocks = nn.Sequential(
            TemporalBlock(input_dim,  hidden_dim,  kernel_size, dilation=1, dropout=dropout),
            TemporalBlock(hidden_dim, hidden_dim,  kernel_size, dilation=2, dropout=dropout),
            TemporalBlock(hidden_dim, output_dim,  kernel_size, dilation=4, dropout=dropout),
        )
        self.out_proj = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)              # (batch, input_dim, seq_len)
        x = self.blocks(x).transpose(1, 2) # (batch, seq_len, output_dim)
        return self.out_proj(x)


class V9TCNModel(nn.Module):
    """
    V9 with TCN f-layer. g-layer and h-layer are structurally identical
    to V9FusionModel — weights transfer directly via load_state_dict(strict=False).
    """
    def __init__(self, weather_features, agronomic_features, gru_units=64, dropout_rate=0.3):
        super().__init__()

        # 1. Environmental Pressure Model (f) — TCN replaces GRU
        self.tcn = TCNEncoder(
            input_dim=weather_features,
            hidden_dim=gru_units,
            output_dim=gru_units * 2,   # matches GRU bidirectional output dim
        )

        # Temporal Attention — SAME interface as original (input: gru_units*2)
        self.attention_w = nn.Linear(gru_units * 2, 1)

        # 2. Agronomic Modulation Model (g) — UNCHANGED
        self.agronomic_mlp = nn.Sequential(
            nn.Linear(agronomic_features, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        # 3. Fusion / Interaction Layer (h) — UNCHANGED
        self.fusion_layer = nn.Sequential(
            nn.Linear(gru_units * 2 + 8, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, weather_seq, agronomic_state):
        # 1. f-layer: TCN temporal encoding
        tcn_out = self.tcn(weather_seq)              # (batch, seq_len, gru_units*2)

        # Temporal Attention
        attn_scores  = self.attention_w(tcn_out)     # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        weather_ctx  = torch.sum(attn_weights * tcn_out, dim=1)  # (batch, gru_units*2)

        # 2. g-layer: Agronomic Modulation
        agro_ctx = self.agronomic_mlp(agronomic_state)  # (batch, 8)

        # 3. h-layer: Fusion
        combined = torch.cat([weather_ctx, agro_ctx], dim=1)
        logits   = self.fusion_layer(combined)           # (batch, 1)

        return logits, attn_weights


if __name__ == "__main__":
    model = V9TCNModel(weather_features=17, agronomic_features=5)
    x_w = torch.randn(4, 14, 17)
    x_a = torch.randn(4, 5)
    out, attn = model(x_w, x_a)
    print(f"Output : {out.shape}")   # (4, 1)
    print(f"Attn   : {attn.shape}")  # (4, 14, 1)

