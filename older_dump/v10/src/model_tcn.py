import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()
        # Causal padding = (kernel_size - 1) * dilation
        self.padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=0, dilation=dilation)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=0, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

    def forward(self, x):
        # x shape: (B, C, L)
        # Pad left by 'self.padding' to maintain causality
        x_padded = F.pad(x, (self.padding, 0))
        out = self.conv1(x_padded)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = F.pad(out, (self.padding, 0))
        out = self.conv2(out)
        out = self.relu(out)
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=128, kernel_size=3, dropout=0.2):
        super().__init__()
        # RF calculation: 1 + 2*(1+2+4) = 15
        self.blocks = nn.Sequential(
            TemporalBlock(input_dim, hidden_dim, kernel_size, dilation=1, dropout=dropout),
            TemporalBlock(hidden_dim, hidden_dim, kernel_size, dilation=2, dropout=dropout),
            TemporalBlock(hidden_dim, output_dim, kernel_size, dilation=4, dropout=dropout),
        )
        self.out_proj = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        # x: (B, seq_len, input_dim) -> (B, input_dim, seq_len)
        x = x.transpose(1, 2)
        out = self.blocks(x)
        # -> (B, seq_len, output_dim)
        out = out.transpose(1, 2)
        return self.out_proj(out)

class V10TCNModel(nn.Module):
    def __init__(self, weather_features, agronomic_features, gru_units=64, seq_len=14, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.tcn = TCNEncoder(weather_features, hidden_dim=gru_units, output_dim=gru_units * 2, dropout=dropout)
        
        self.attention_w = nn.Linear(gru_units * 2, 1)
        
        # Reduced g-layer for crop proxies
        self.agronomic_mlp = nn.Sequential(
            nn.Linear(agronomic_features, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(gru_units * 2 + 4, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, weather_seq, agronomic_state):
        # weather_seq: (B, 14, weather_features)
        tcn_out = self.tcn(weather_seq)  # (B, 14, gru_units*2)
        
        # Calculate attention scores
        attn_scores = self.attention_w(tcn_out)  # (B, 14, 1)
        
        # Temporal Bias: emphasize the entire buildup, not just t-1
        # Use torch.linspace directly to shift scores
        time_weights = torch.linspace(0.8, 1.2, steps=self.seq_len, device=attn_scores.device)
        time_weights = time_weights.unsqueeze(0).unsqueeze(-1)  # (1, 14, 1)
        attn_scores = attn_scores * time_weights
        
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # Context vector
        weather_ctx = torch.sum(attn_weights * tcn_out, dim=1)  # (B, gru_units*2)
        
        # Agro state
        agro_ctx = self.agronomic_mlp(agronomic_state)  # (B, 4)
        
        # Fusion
        combined = torch.cat([weather_ctx, agro_ctx], dim=1)
        logits = self.fusion_layer(combined)
        
        return logits, attn_weights

if __name__ == "__main__":
    model = V10TCNModel(14, 5)
    w = torch.randn(2, 14, 14)
    a = torch.randn(2, 5)
    logits, attn = model(w, a)
    print("Logits shape:", logits.shape)
    print("Attention shape:", attn.shape)
