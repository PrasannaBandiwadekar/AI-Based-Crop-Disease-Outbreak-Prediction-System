import torch
import torch.nn as nn
import torch.nn.functional as F

class V9FusionModel(nn.Module):
    def __init__(self, weather_features, agronomic_features, gru_units=64, dropout_rate=0.3):
        super(V9FusionModel, self).__init__()
        
        # 1. Environmental Pressure Model (f)
        self.gru = nn.GRU(
            input_size=weather_features,
            hidden_size=gru_units,
            batch_first=True,
            bidirectional=True
        )
        
        # Temporal Attention
        self.attention_w = nn.Linear(gru_units * 2, 1)
        
        # 2. Agronomic Modulation Model (g)
        self.agronomic_mlp = nn.Sequential(
            nn.Linear(agronomic_features, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        
        # 3. Fusion / Interaction Layer (h)
        # Input: [Attention Output (gru*2), Agronomic Output (8)]
        self.fusion_layer = nn.Sequential(
            nn.Linear(gru_units * 2 + 8, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
    def forward(self, weather_seq, agronomic_state):
        # weather_seq: (batch, seq_len, weather_features)
        # agronomic_state: (batch, agronomic_features)
        
        # 1. Weather Dynamics
        gru_out, _ = self.gru(weather_seq) # (batch, seq_len, units*2)
        
        # Temporal Attention
        attn_scores = self.attention_w(gru_out) # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        
        weather_context = torch.sum(attn_weights * gru_out, dim=1) # (batch, units*2)
        
        # 2. Agronomic Modulation
        agro_context = self.agronomic_mlp(agronomic_state) # (batch, 8)
        
        # 3. Fusion
        combined = torch.cat((weather_context, agro_context), dim=1)
        logits = self.fusion_layer(combined)
        
        return logits, attn_weights

if __name__ == "__main__":
    # Test
    model = V9FusionModel(weather_features=10, agronomic_features=5)
    x_weather = torch.randn(8, 14, 10)
    x_agro = torch.randn(8, 5)
    out, attn = model(x_weather, x_agro)
    print(f"Output Shape: {out.shape}")
    print(f"Attention Shape: {attn.shape}")
