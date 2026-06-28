"""
V11 KG-CTCN Model Architecture
--------------------------------------------------
Knowledge-Guided Causal Temporal Convolutional Network.

Components:
  - WeatherTCNEncoder   : dilated causal 1D convolutions over a 28-day window
  - AgronomicMLPEncoder : static vulnerability embedding
  - Cross-modal attention fusion : agronomy gates the weather signal
  - RiskHead            : sigmoid risk probability
  - ConfidenceHead      : auxiliary calibration head (see training note below)

CHANGELOG (v11.1):
  - [DESIGN]  Corrected attention fusion. Previously both gates were
              independent (att_w from w_embed, att_a from a_embed), meaning
              the agronomic branch could not suppress the weather signal.
              Now att_w is conditioned on a_embed so a resistant variety
              actively suppresses weather-driven risk — matching the
              architecture specification. att_a is removed; the agronomic
              embedding enters the fusion layer unscaled.
  - [DESIGN]  ConfidenceHead converted from an unsupervised sigmoid output to
              a calibrated auxiliary head trained against |risk_prob - label|
              as a regression target. The forward() method now returns the raw
              confidence logit (pre-sigmoid) so the training loop can apply
              BCELoss with the soft calibration target. Inference callers
              should apply torch.sigmoid() to the third return value.
              Training loop change required: add confidence calibration loss
              (see docstring in KGCTCN.forward).
  - [BUG]     get_risk_class vectorised. Previous per-element .item() loop was
              fragile to (B,1) vs (B,) shape variants. Now uses .tolist() on
              a squeezed tensor.
  - [BUG]     tcn_channels mutable default argument fixed. Python shares
              mutable defaults across calls; replaced with None sentinel.
  - [MINOR]   Removed post-residual ReLU in TemporalBlock. ReLU after the
              skip-add can deaden gradients in early layers of a deep stack.
              Activation is now applied only inside the block's sequential
              (conv -> chomp -> relu -> dropout), matching the original TCN
              paper convention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# TCN building blocks
# ---------------------------------------------------------------------------

class Chomp1d(nn.Module):
    """
    Remove the trailing `chomp_size` time steps introduced by causal padding.
    Ensures Conv1d output at position t depends only on inputs at t and earlier.
    """
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    One dilated causal residual block: two (Conv1d → Chomp → ReLU → Dropout)
    layers with a skip connection.

    Causality is guaranteed by left-only padding = (kernel_size - 1) * dilation,
    then chomping the right side off with Chomp1d.

    Post-residual ReLU is intentionally omitted (see CHANGELOG). Activation
    lives inside the sequential so gradients flow cleanly through the skip path.
    """
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
        )
        self.chomp1   = Chomp1d(padding)
        self.relu1    = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
        )
        self.chomp2   = Chomp1d(padding)
        self.relu2    = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )

        # 1x1 conv to match channel dimensions when input ≠ output channels
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        # No ReLU after the addition — see CHANGELOG
        return out + res


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class WeatherTCNEncoder(nn.Module):
    """
    Stack of TemporalBlocks with exponentially increasing dilation.
    Receptive field grows as 2^num_levels, covering the full 28-day window
    with 4 levels (dilation 1, 2, 4, 8 → field of 16; kernel_size=2 doubles
    it to 32, encompassing SEQ_LEN=28).

    Input:  (B, seq_len, num_features)  — time-last from the dataloader
    Output: (B, tcn_channels[-1])       — embedding at final timestep t
    """
    def __init__(
        self,
        num_inputs: int,
        num_channels: list,
        kernel_size: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            dilation  = 2 ** i
            in_ch     = num_inputs if i == 0 else num_channels[i - 1]
            padding   = (kernel_size - 1) * dilation
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size,
                              stride=1, dilation=dilation,
                              padding=padding, dropout=dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, F) → transpose to (B, F, seq_len) for Conv1d
        out = self.network(x.transpose(1, 2))
        return out[:, :, -1]  # causal: only the last timestep


class AgronomicMLPEncoder(nn.Module):
    """
    Two-layer MLP for static agronomic vulnerability features.
    Input:  (B, num_agro_features)  — already StandardScaler-normalised
    Output: (B, out_dim)
    """
    def __init__(self, num_features: int, hidden_dim: int = 32, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class KGCTCN(nn.Module):
    """
    Knowledge-Guided Causal Temporal Convolutional Network.

    Fusion logic (cross-modal attention):
      att_w = sigmoid(Linear(a_embed))   ← agronomy gates the weather signal
      fused = concat(w_embed * att_w, a_embed)

      A highly resistant variety produces a small att_w, suppressing the
      weather encoder's contribution to the risk prediction regardless of
      how extreme the meteorological conditions are. This matches the
      biological prior: resistant varieties tolerate high-humidity conditions
      that would devastate susceptible crops.

    Outputs:
      logits     : (B, 1)  raw pre-sigmoid score (use for BCE / Focal loss)
      risk_prob  : (B, 1)  sigmoid probability    (use for causal loss, inference)
      conf_logit : (B, 1)  pre-sigmoid confidence  (apply sigmoid for [0,1] output)

    Confidence head — training note:
      conf_logit has no intrinsic meaning until trained with a calibration
      target. In the training loop, compute:

          conf_target = 1.0 - (risk_prob.detach() - label).abs()  # soft [0,1]
          conf_loss   = F.binary_cross_entropy_with_logits(conf_logit, conf_target)

      Add conf_loss (weight ~0.05) to the total loss. This teaches the head
      to output high confidence when the risk prediction is close to the true
      label, and low confidence when it is far off.

      At inference time:  confidence = torch.sigmoid(conf_logit)
    """

    def __init__(
        self,
        num_weather_features: int,
        num_agro_features: int,
        tcn_channels: list = None,   # None avoids mutable default argument
        tcn_kernel_size: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        if tcn_channels is None:
            tcn_channels = [32, 64, 64, 64]

        # ── Encoders ────────────────────────────────────────────────────────
        self.weather_encoder = WeatherTCNEncoder(
            num_weather_features, tcn_channels, tcn_kernel_size, dropout
        )
        self.agro_encoder = AgronomicMLPEncoder(
            num_agro_features, hidden_dim=32, out_dim=32
        )

        weather_embed_dim = tcn_channels[-1]   # 64
        agro_embed_dim    = 32

        # ── Cross-modal attention gate ───────────────────────────────────────
        # att_w is conditioned on the agronomic embedding, not the weather
        # embedding. This allows a resistant variety (low agro risk score) to
        # suppress the weather signal, implementing the biological prior.
        self.attention_w = nn.Linear(agro_embed_dim, 1)

        # ── Fusion MLP ───────────────────────────────────────────────────────
        fusion_in_dim = weather_embed_dim + agro_embed_dim  # 64 + 32 = 96
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # ── Prediction heads ─────────────────────────────────────────────────
        self.risk_head = nn.Linear(32, 1)

        # Confidence head: outputs a pre-sigmoid logit. Train with a soft
        # calibration target (see class docstring). Apply sigmoid at inference.
        self.confidence_head = nn.Linear(32, 1)

    def forward(
        self,
        x_weather: torch.Tensor,
        x_agro: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_weather : (B, seq_len, num_weather_features)
            x_agro    : (B, num_agro_features)

        Returns:
            logits     : (B, 1)
            risk_prob  : (B, 1)
            conf_logit : (B, 1)  — apply torch.sigmoid() for [0,1] confidence
        """
        # 1. Encode each modality independently
        w_embed = self.weather_encoder(x_weather)   # (B, weather_embed_dim)
        a_embed = self.agro_encoder(x_agro)         # (B, agro_embed_dim)

        # 2. Cross-modal attention: agronomy gates weather
        #    sigmoid output in (0, 1); low value suppresses weather branch
        att_w = torch.sigmoid(self.attention_w(a_embed))   # (B, 1)
        gated_weather = w_embed * att_w                    # (B, weather_embed_dim)

        # 3. Concatenate and fuse
        fused      = torch.cat([gated_weather, a_embed], dim=1)  # (B, 96)
        unified    = self.fusion_fc(fused)                        # (B, 32)

        # 4. Dual heads
        logits     = self.risk_head(unified)        # (B, 1) — raw logit
        risk_prob  = torch.sigmoid(logits)          # (B, 1) — probability
        conf_logit = self.confidence_head(unified)  # (B, 1) — pre-sigmoid

        return logits, risk_prob, conf_logit

    # ── Utility ──────────────────────────────────────────────────────────────

    @staticmethod
    def get_risk_class(risk_prob: torch.Tensor) -> list:
        """
        Map a batch of sigmoid probabilities to risk category strings.

        Args:
            risk_prob : (B,) or (B, 1) tensor of probabilities in [0, 1]

        Returns:
            List of strings: "Low" (< 0.3), "Medium" (0.3 – 0.7), "High" (≥ 0.7)
        """
        p = risk_prob.squeeze(-1).tolist()  # handles both (B,) and (B,1)
        return [
            "Low" if v < 0.3 else "High" if v >= 0.7 else "Medium"
            for v in p
        ]