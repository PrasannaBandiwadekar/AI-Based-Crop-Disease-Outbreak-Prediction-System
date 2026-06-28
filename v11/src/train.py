"""
V11 KG-CTCN Training Pipeline
--------------------------------------------------
Implements end-to-end training with composite loss:
  Loss = BCE (weighted sampler) + 0.5 * Focal

Split: Train(2005-2018), Val(2019-2021), Test(2022-2024)
Scheduler: CosineAnnealingLR (T_max=120, eta_min=1e-5)

CHANGELOG (v11.1):
  - [CRITICAL]  Removed 'RH2M_z90', 'T2M_z90', 'PRECTOTCORR_z90' from
                WEATHER_FEATURES. These columns no longer exist after the
                dataset_pipeline.py update. The base columns (RH2M, T2M,
                PRECTOTCORR) are already the 90-day rolling Z-scores.
  - [CRITICAL]  Removed StandardScaler on X_w (weather sequences). The
                pipeline already produces Z-scored weather features. Applying
                StandardScaler again created a Z-score of a Z-score, and
                miscalibrated all causal_consistency_loss thresholds.
                StandardScaler is still applied to X_a (agronomic features),
                which arrive in natural units from the pipeline.
  - [CRITICAL]  warmup_mask filter applied before build_sequences(). The first
                90 rows of the dataset have NaN/unstable Z-scores and must not
                appear in any sequence window.
  - [BUG]       Moved 'import torch.nn.functional as F' to top-level imports.
                Previously defined after FocalLoss class, which would fail if
                the class were ever instantiated at module load time.
  - [BUG]       causal_consistency_loss thresholds recalibrated to natural
                units. RH_persist_7d is a 0-7 accumulation (daily 0-1 soft
                flags); threshold raised from 1.0 to 3.5 (half-maximum).
                Rain_sum_7d is raw mm; threshold raised from 1.0 to 10.0 mm
                (a meaningful weekly accumulation for Red Rot pressure).
  - [DESIGN]    Removed temporal_stability_loss dead stub. The function always
                returned 0.0 because WeightedRandomSampler breaks consecutive
                batch ordering. Removed rather than left as silent dead code.
  - [DESIGN]    best_val_loss now tracks mean loss per batch (divided by
                len(val_loader)) rather than raw summed loss. Summed loss
                changes with dataset size, making cross-run checkpoint
                comparisons unreliable.
  - [DESIGN]    Removed pos_weight from BCEWithLogitsLoss. WeightedRandomSampler
                already corrects for class imbalance at the batch level.
                Using both mechanisms double-corrects: sampler ensures balanced
                batches, so BCE sees roughly equal class frequency naturally.
  - [DESIGN]    Model metadata saved as JSON instead of flat .txt. Enables
                reliable parsing in the inference engine and carries a
                pipeline_version field for experiment tracking.

CHANGELOG (v11.2):
  - [TRAIN]     Epochs increased from 40 to 80.
  - [TRAIN]     Learning rate lowered from 1e-3 to 5e-4.
  - [TRAIN]     CosineAnnealingLR scheduler added (T_max=80, eta_min=1e-5).
  - [TRAIN]     scheduler.step() called after each epoch (epoch-level).
  - [TRAIN]     PIPELINE_VERSION bumped to v11.2.
  - [VERIFY]    assign_causal_labels comparison logic confirmed correct.
                Vectorised [O-7, O-3] window is equivalent to original loop.
                49 train positives consistent with 9 GT events + overlap removal.

CHANGELOG (v11.3):
  - [TRAIN]     augment_positives() added. With only 49 training positives
                the model memorises training outbreak sequences (train loss
                → 0.014) but fails to generalise (val AP stuck ~0.025).
                Gaussian noise augmentation (std=0.05 on Z-scored features)
                expands the positive pool from 49 → 245 sequences (4× copies
                + original) without generating biologically implausible data.
  - [TRAIN]     Augmentation applied AFTER train/val/test split and ONLY to
                training positives. Augmenting before splitting would leak
                perturbed versions of val-adjacent sequences into training.
  - [TRAIN]     KG-derived natural-unit features (RH_high_flag, RH_persist_7d,
                Rain_sum_7d, Rain_sum_14d, Monsoon_ind) excluded from noise.
                Adding noise to a 0/1 monsoon flag or a bounded 0-7 accumulation
                is not biologically meaningful. Noise applied to Z-scored
                weather columns and latent windows only.
  - [TRAIN]     WeightedRandomSampler rebuilt after augmentation so sample
                weights reflect the expanded positive pool size.
  - [TRAIN]     PIPELINE_VERSION bumped to v11.3.
  - [TRAIN]     augmentation config written to metadata JSON for
                experiment reproducibility.

CHANGELOG (v11.6):
  - [TRAIN]     Refined monotonicity training strategy: delayed start (epoch 20),
                subsampling (1/4 batches), and reduced weight (0.1). This
                prevents the monotonicity loss from interfering with the
                primary weather-signal learning in early epochs.
  - [TRAIN]     Shortened training to 60 epochs with T_max=60 scheduler.
  - [TRAIN]     PIPELINE_VERSION bumped to v11.6.
"""



import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
import joblib

from model import KGCTCN

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

# NOTE: RH2M, T2M, PRECTOTCORR etc. are already 90-day rolling Z-scores
# produced by dataset_pipeline.py. The _z90 aliases were removed in v11.1.
# KG-derived features (RH_high_flag, RH_persist_7d, Rain_sum_*) remain in
# natural units and must NOT be re-scaled (see StandardScaler note below).
WEATHER_FEATURES = [
    "WS10M", "T2M", "RH2M", "T2M_MIN", "T2M_MAX", "PRECTOTCORR",
    "T2M_MIN_lag_15d",
    "RH_high_flag",       # natural unit: 0.0 – 1.0 soft intensity
    "RH_persist_7d",      # natural unit: 0.0 – 7.0 accumulated daily flags
    "Rain_sum_7d",        # natural unit: mm over 7 days
    "Rain_sum_14d",       # natural unit: mm over 14 days
    "Monsoon_ind",        # natural unit: 0 / 1
    "RH2M_latent_window",
    "T2M_latent_window",
]
AGRO_FEATURES = ["variety_susceptibility", "is_ratoon", "crop_age_days"]

SEQ_LEN = 28  # TCN receptive field window (days)

AUGMENT_FACTOR    = 4     # was 7
AUGMENT_NOISE_STD = 0.05  # was 0.05

PIPELINE_VERSION = "v11.6"

# ---------------------------------------------------------------------------
# Custom losses
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal loss for hard-example mining on imbalanced binary classification.
    Downweights easy negatives so the model focuses on ambiguous outbreak
    boundary cases.
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1.0 - pt) ** self.gamma * bce
        return focal.mean()


def causal_consistency_loss(
    probs: torch.Tensor,
    weather_batch: torch.Tensor,
    feature_names: list,
) -> torch.Tensor:
    """
    Penalise predictions that violate explicit KG rules.

    Rule: when weekly RH persistence AND weekly rainfall are both high,
    predicted risk must not be near zero.

    Thresholds are calibrated to NATURAL UNITS (weather features are NOT
    re-scaled in this pipeline — see CHANGELOG):
      RH_persist_7d > 3.5  →  more than half the week had meaningful humidity
      Rain_sum_7d   > 10.0 →  at least 10 mm accumulated over 7 days

    weather_batch shape: (B, seq_len, num_features)
    probs shape: (B, 1) – sigmoid probabilities from the model
    """
    rh_idx   = feature_names.index("RH_persist_7d")
    rain_idx = feature_names.index("Rain_sum_7d")

    # Use the final timestep in each sequence (current observation)
    rh_val   = weather_batch[:, -1, rh_idx]    # 0.0 – 7.0
    rain_val = weather_batch[:, -1, rain_idx]  # mm

    condition = (rh_val > 3.5) & (rain_val > 10.0)
    if not condition.any():
        return torch.tensor(0.0, device=probs.device)

    # Violation: condition met but model predicts very low risk (< 0.3)
    violation = F.relu(0.3 - probs[condition])
    return violation.mean()


def agronomic_monotonicity_loss(
    model: torch.nn.Module,
    weather_batch: torch.Tensor,
    agro_batch: torch.Tensor,
    agro_feature_names: list,
    a_sc_scales: torch.Tensor,
    probs_orig: torch.Tensor,
) -> torch.Tensor:
    """
    Enforce biological monotonicity over agronomic inputs:
      (1) susceptible variety (2) must predict >= moderate variety (1) risk
      (2) ratoon crop (1) must predict >= plant crop (0) risk

    Implementation: for each sample in the batch, construct a "downgraded"
    copy of agro_batch (susceptible->moderate, ratoon->plant) and a
    "downgraded" agro for ratoon. The loss penalises whenever the model
    predicts higher risk for the less-vulnerable agronomic state.

    Shift is applied as -1.0 / scale to ensure a 1-unit raw shift in
    scaled space.

    agro_batch shape: (B, num_agro_features) — scaled values as fed to model.
    a_sc_scales shape: (num_agro_features,) — the .scale_ from StandardScaler.
    probs_orig: probabilities from the main forward pass (reused for speed).
    Weight recommendation: 0.3 (subordinate to BCE+Focal primary loss).
    """
    v_idx = agro_feature_names.index("variety_susceptibility")
    r_idx = agro_feature_names.index("is_ratoon")

    # -- Variety monotonicity: prob(susceptible) >= prob(moderate) --
    # Downgrade variety: shift down 1 raw unit in scaled space.
    agro_variety_down = agro_batch.clone()
    agro_variety_down[:, v_idx] = agro_variety_down[:, v_idx] - (1.0 / a_sc_scales[v_idx])
    _, probs_variety_down, _ = model(weather_batch, agro_variety_down)

    # Violation: original (more vulnerable) risk < downgraded (less vulnerable) risk
    variety_violation = F.relu(probs_variety_down - probs_orig)

    # -- Ratoon monotonicity: prob(ratoon=1) >= prob(ratoon=0) --
    agro_ratoon_down = agro_batch.clone()
    agro_ratoon_down[:, r_idx] = agro_ratoon_down[:, r_idx] - (1.0 / a_sc_scales[r_idx])
    _, probs_ratoon_down, _ = model(weather_batch, agro_ratoon_down)

    ratoon_violation = F.relu(probs_ratoon_down - probs_orig)

    return (variety_violation.mean() + ratoon_violation.mean()) / 2.0


# ---------------------------------------------------------------------------
# Positive-class augmentation
# ---------------------------------------------------------------------------

# Feature indices that are Z-scored and safe to perturb with Gaussian noise.
# KG-derived natural-unit features are excluded: adding noise to a bounded
# 0-7 accumulation or a 0/1 flag is not biologically meaningful.
_NOISE_FEATURE_NAMES = [
    "WS10M", "T2M", "RH2M", "T2M_MIN", "T2M_MAX", "PRECTOTCORR",
    "T2M_MIN_lag_15d", "RH2M_latent_window", "T2M_latent_window",
]

def augment_positives(
    X_w: np.ndarray,
    X_a: np.ndarray,
    y: np.ndarray,
    augment_factor: int = 4,
    noise_std: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Expand the positive-labelled training sequences by Gaussian perturbation.

    For each of the `augment_factor` copies, independent noise is sampled and
    added to the Z-scored weather channels only. KG-derived natural-unit
    features (RH_high_flag, RH_persist_7d, Rain_sum_*, Monsoon_ind) are held
    fixed — perturbing a bounded accumulation or a binary flag is not
    biologically defensible.

    noise_std=0.05 means each perturbed value shifts by ±0.05 standard
    deviations relative to the local 90-day baseline — small enough to stay
    within the biologically plausible envelope of the original event.

    Args:
        X_w            : (N, seq_len, num_weather_features) — training weather
        X_a            : (N, num_agro_features) — training agro (scaled)
        y              : (N,) — training labels
        augment_factor : number of noisy copies per positive sequence
        noise_std      : Gaussian noise std dev (applied to Z-scored channels)
        seed           : RNG seed for reproducibility

    Returns:
        X_w_aug, X_a_aug, y_aug — original arrays with augmented positives
        appended. Negatives are unchanged; only positive rows are duplicated.

    NOTE: Call this function AFTER the train/val/test split. Never augment
    before splitting — perturbed copies of training positives must not appear
    in the validation or test sets.
    """
    pos_mask = (y == 1)
    n_pos    = pos_mask.sum()
    if n_pos == 0:
        print("  augment_positives: no positive samples found — skipping.")
        return X_w, X_a, y

    # Identify which feature indices are safe to perturb
    noise_idx = [WEATHER_FEATURES.index(f) for f in _NOISE_FEATURE_NAMES]

    X_w_pos = X_w[pos_mask]   # (n_pos, seq_len, num_features)
    X_a_pos = X_a[pos_mask]   # (n_pos, num_agro_features)
    y_pos   = y[pos_mask]     # (n_pos,)

    rng     = np.random.default_rng(seed)
    aug_w_list = []
    aug_a_list = []
    aug_y_list = []

    for copy_idx in range(augment_factor):
        X_w_copy = X_w_pos.copy()
        noise    = rng.normal(0.0, noise_std, X_w_pos[:, :, noise_idx].shape).astype(np.float32)
        X_w_copy[:, :, noise_idx] += noise
        aug_w_list.append(X_w_copy)
        aug_a_list.append(X_a_pos)   # agro features unchanged
        aug_y_list.append(y_pos)

    X_w_aug = np.concatenate([X_w] + aug_w_list, axis=0)
    X_a_aug = np.concatenate([X_a] + aug_a_list, axis=0)
    y_aug   = np.concatenate([y]   + aug_y_list, axis=0)

    print(f"  Augmented positives: {n_pos} -> {n_pos * (1 + augment_factor)}  "
          f"(factor={augment_factor}, noise_std={noise_std})")
    print(f"  Total training sequences: {len(y)} -> {len(y_aug)}")
    return X_w_aug, X_a_aug, y_aug

# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_sequences(df: pd.DataFrame, seq_len: int):
    """
    Build sliding-window sequences of length seq_len.
    For row i, the weather window covers rows [i-seq_len+1, i] (inclusive).
    Label and agronomic features are taken at position i.

    Returns:
        X_w     : (N, seq_len, len(WEATHER_FEATURES))  float32
        X_a     : (N, len(AGRO_FEATURES))              float32
        y       : (N,)                                  float32
        dates   : (N,)                                  DatetimeIndex
    """
    fv = df[WEATHER_FEATURES].values.astype(np.float32)
    av = df[AGRO_FEATURES].values.astype(np.float32)
    lv = df["risk_label"].values.astype(np.float32)
    dv = df["date"].values

    X_w, X_a, y_arr, dates_arr = [], [], [], []
    for i in range(seq_len, len(df)):
        X_w.append(fv[i - seq_len + 1 : i + 1])
        X_a.append(av[i])
        y_arr.append(lv[i])
        dates_arr.append(dv[i])

    return (
        np.array(X_w, dtype=np.float32),
        np.array(X_a, dtype=np.float32),
        np.array(y_arr, dtype=np.float32),
        pd.to_datetime(np.array(dates_arr)),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    # ── Load & filter ───────────────────────────────────────────────────────
    df = pd.read_csv(os.path.join(DATA_DIR, "v11_features.csv"))
    df["date"] = pd.to_datetime(df["date"])

    # Exclude warm-up rows (first 90 days of unstable rolling Z-scores).
    # warmup_mask=1 means the row is in the warm-up window.
    n_before = len(df)
    df = df[df["warmup_mask"] == 0].reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} warm-up rows. Remaining: {len(df)}")

    # ── Labeling (v2) ────────────────────────────────────────────────────────
    # Assign labels based on literature-anchored GT peaks and actionable window.
    # Must happen before NaN audit because 'risk_label' is included in the audit list.
    from assign_causal_labels_v2 import assign_labels
    GT_PATH = os.path.join(BASE_DIR, "research_comp", "evidence_base",
                           "outbreak_events", "sangli_gt_v2.csv")
    df = assign_labels(df, gt_path=GT_PATH)

    # ── NaN audit ────────────────────────────────────────────────────────────
    # Some KG features (RH2M_latent_window, T2M_latent_window) use shift(7) +
    # rolling(21) and can produce NaN beyond row 90 at the boundary. Any NaN
    # entering a sequence window will produce NaN activations → NaN loss →
    # NaN gradients → corrupted model weights after the first optimizer step.
    # Drop any remaining NaN rows and report which columns caused them.
    all_feature_cols = WEATHER_FEATURES + AGRO_FEATURES + ["risk_label"]
    nan_counts = df[all_feature_cols].isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if not nan_cols.empty:
        print(f"\nWARNING: NaN values found after warmup filter — dropping affected rows.")
        for col, count in nan_cols.items():
            print(f"  {col}: {count} NaN rows")
        n_before_nan = len(df)
        df = df.dropna(subset=all_feature_cols).reset_index(drop=True)
        print(f"  Dropped {n_before_nan - len(df)} rows. Remaining: {len(df)}\n")
    else:
        print("NaN audit passed — no NaN values in feature columns.")

    # ── Build sequences ──────────────────────────────────────────────────────
    X_w, X_a, y, dates = build_sequences(df, SEQ_LEN)

    # ── Chronological splits ─────────────────────────────────────────────────
    train_mask = (dates.year >= 2005) & (dates.year <= 2018)
    val_mask   = (dates.year >= 2019) & (dates.year <= 2021)
    test_mask  = (dates.year >= 2022) & (dates.year <= 2024)

    print(f"Train: {train_mask.sum():>5d} samples  (Pos: {y[train_mask].sum():.0f})")
    print(f"Val:   {val_mask.sum():>5d} samples  (Pos: {y[val_mask].sum():.0f})")
    print(f"Test:  {test_mask.sum():>5d} samples  (Pos: {y[test_mask].sum():.0f})")

    # ── Scaling ──────────────────────────────────────────────────────────────
    # Weather features (X_w): NOT scaled here. Base weather columns are already
    # 90-day rolling Z-scores from the pipeline. KG-derived features
    # (RH_persist_7d, Rain_sum_*, etc.) are in natural units and the
    # causal_consistency_loss thresholds depend on those units.
    #
    # Agronomic features (X_a): scaled here because they arrive in raw units
    # (susceptibility 1-5, age 0-365) and have no pipeline normalization.
    a_sc = StandardScaler()
    X_a_sc = X_a.copy()
    X_a_sc[train_mask] = a_sc.fit_transform(X_a[train_mask])
    X_a_sc[val_mask]   = a_sc.transform(X_a[val_mask])
    X_a_sc[test_mask]  = a_sc.transform(X_a[test_mask])

    # ── Positive-class augmentation ────────────────────────────────────────────
    # Applied AFTER split extraction, ONLY to training data.
    # Val and test arrays (X_w[val_mask] etc.) are untouched.
    print("\nAugmenting training positives...")
    X_w_train, X_a_train, y_train = augment_positives(
        X_w[train_mask],
        X_a_sc[train_mask],
        y[train_mask],
        augment_factor=AUGMENT_FACTOR,
        noise_std=AUGMENT_NOISE_STD,
    )


    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── DataLoaders ──────────────────────────────────────────────────────────
    a_sc_scales_torch = torch.FloatTensor(a_sc.scale_).to(device)
    
    train_ds = TensorDataset(
        torch.FloatTensor(X_w_train),
        torch.FloatTensor(X_a_train),
        torch.FloatTensor(y_train).view(-1, 1),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_w[val_mask]),
        torch.FloatTensor(X_a_sc[val_mask]),
        torch.FloatTensor(y[val_mask]).view(-1, 1),
    )

    class_counts = np.bincount(y_train.astype(int))
    sample_weights = (1.0 / class_counts)[y_train.astype(int)]
    sampler = WeightedRandomSampler(
        torch.DoubleTensor(sample_weights), num_samples=len(y_train), replacement=True
    )
    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler)
    val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False)

    # ── Model & optimiser ────────────────────────────────────────────────────
    model = KGCTCN(len(WEATHER_FEATURES), len(AGRO_FEATURES), dropout=0.4).to(device)
    
    # Loss functions
    bce_loss_fn   = nn.BCEWithLogitsLoss()
    focal_loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
    optimizer     = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    
    # Warmup + Cosine schedule: starts at 1e-5, peaks at 5e-4 at epoch 5, then decays.
    scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=55, eta_min=1e-5
    )

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_ap  = 0.0
    patience     = 10   # stop when val AP hasn't improved for this many epochs
    patience_ctr = 0
    best_epoch   = 0
    print("\nStarting training...")
    print(f"{'Epoch':>6}  {'Train loss':>11}  {'Val loss':>9}  {'Val AP':>7}  {'LR':>9}")
    print("-" * 54)

    MONO_WARMUP_EPOCHS = 5

    for epoch in range(1, 101):  # cap at 100; early stopping fires much sooner
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for batch_idx, (bw, ba, by) in enumerate(train_loader):
            bw, ba, by = bw.to(device), ba.to(device), by.to(device)
            optimizer.zero_grad()

            logits, probs, conf_logit = model(bw, ba)

            # Composite loss:
            #   BCE        — class balance via WeightedRandomSampler
            #   Focal      — hard-example mining on outbreak boundary cases
            #   Conf       — calibrates confidence head against |prob - label|
            #   Mono       — agronomic monotonicity (susceptible >= moderate, ratoon >= plant)
            #                Delayed start and subsampled to avoid interfering with early signal.
            conf_target = 1.0 - (probs.detach() - by).abs()
            conf_loss   = F.binary_cross_entropy_with_logits(conf_logit, conf_target)

            if epoch > MONO_WARMUP_EPOCHS and batch_idx % 4 == 0:
                mono_loss = agronomic_monotonicity_loss(
                    model, bw, ba, AGRO_FEATURES, a_sc_scales_torch, probs
                )
            else:
                mono_loss = torch.tensor(0.0, device=device)

            loss = (bce_loss_fn(logits, by)
                    + 0.5 * focal_loss_fn(logits, by)
                    + 0.05 * conf_loss
                    + 0.1 * mono_loss)
            loss.backward()

            # Clip gradients — prevents NaN/Inf weights if any boundary NaN
            # survives into a sequence window and reaches the loss.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Abort immediately if loss is NaN — better than silent corruption.
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}: {loss.item():.4f}. "
                    "Check for NaN in input features or exploding gradients."
                )

            optimizer.step()
            epoch_loss += loss.item()

        # Step scheduler once per epoch (cosine schedule is epoch-level)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        mean_train_loss = epoch_loss / len(train_loader)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_preds, val_y = [], []
        with torch.no_grad():
            for bw, ba, by in val_loader:
                bw, ba, by = bw.to(device), ba.to(device), by.to(device)
                logits, probs, _ = model(bw, ba)
                
                # Validation loss match
                val_loss_sum += bce_loss_fn(logits, by).item()
                val_preds.extend(probs.cpu().numpy().flatten())
                val_y.extend(by.cpu().numpy().flatten())

        # Use mean loss per batch so checkpointing is comparable across runs
        # with different val set sizes or batch sizes.
        mean_val_loss = val_loss_sum / len(val_loader)

        # Guard against NaN predictions before calling sklearn.
        # NaN in val_preds means weights were corrupted by a NaN loss.
        val_preds_arr = np.array(val_preds)
        if not np.isfinite(val_preds_arr).all():
            nan_n = (~np.isfinite(val_preds_arr)).sum()
            raise RuntimeError(
                f"Epoch {epoch}: {nan_n} non-finite values in val_preds. "
                "Model weights are NaN — check training batches for NaN features."
            )
        val_ap = average_precision_score(val_y, val_preds_arr) if sum(val_y) > 0 else 0.0

        if epoch % 5 == 0 or epoch <= 10:
            print(f"{epoch:>6}  {mean_train_loss:>11.4f}  {mean_val_loss:>9.4f}  {val_ap:>7.4f}  {current_lr:>9.2e}")

        # Checkpoint on val AP — val BCE rewards predicting zero for everything.
        if val_ap > best_val_ap:
            best_val_ap  = val_ap
            best_epoch   = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "v11_kg_ctcn.pth"))
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best Val AP {best_val_ap:.4f} at epoch {best_epoch})")
                break

    print("\nTraining complete.")
    print(f"Best Val AP: {best_val_ap:.4f}  (epoch {best_epoch})")


    # ── Save scalers ─────────────────────────────────────────────────────────
    # Only the agronomic scaler is saved — weather features are not re-scaled
    # at inference time either. The inference engine must not apply a weather
    # scaler to incoming data.
    joblib.dump(a_sc, os.path.join(MODEL_DIR, "agro_scaler.pkl"))
    print("Agronomic scaler saved. (No weather scaler — pipeline handles normalization.)")

    # ── Temperature Scaling (post-hoc calibration on val set) ────────────────
    # The model outputs bimodal scores (near-0 or near-1) because pos_weight
    # and focal loss together push logits to extremes. Temperature scaling
    # divides logits by a learned scalar T > 1 to spread the distribution.
    # T is fit on val set logits only — no model weights are changed.
    print("\nFitting temperature scaling on val set...")

    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "v11_kg_ctcn.pth")))
    model.eval()

    # Collect raw logits (pre-sigmoid) on val set
    val_logits_list, val_labels_list = [], []
    val_ds_ts = TensorDataset(
        torch.FloatTensor(X_w[val_mask]),
        torch.FloatTensor(X_a_sc[val_mask]),
        torch.FloatTensor(y[val_mask]).view(-1, 1),
    )
    val_loader_ts = DataLoader(val_ds_ts, batch_size=128, shuffle=False)

    with torch.no_grad():
        for bw, ba, by in val_loader_ts:
            bw, ba, by = bw.to(device), ba.to(device), by.to(device)
            logits, _, _ = model(bw, ba)
            val_logits_list.append(logits.cpu())
            val_labels_list.append(by.cpu())

    val_logits_ts = torch.cat(val_logits_list)
    val_labels_ts = torch.cat(val_labels_list)

    # Fit temperature T by minimising BCE on val logits/T
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    t_optimizer = optim.LBFGS([temperature], lr=0.1, max_iter=100)
    ts_loss_fn  = nn.BCEWithLogitsLoss()

    def ts_eval():
        t_optimizer.zero_grad()
        scaled = val_logits_ts / temperature.clamp(min=0.1)
        loss   = ts_loss_fn(scaled, val_labels_ts)
        loss.backward()
        return loss

    t_optimizer.step(ts_eval)
    T = temperature.item()
    print(f"  Learned temperature T = {T:.4f}")
    print(f"  (T > 1 spreads scores toward 0.5; T < 1 sharpens them)")

    # Save temperature alongside model
    joblib.dump(T, os.path.join(MODEL_DIR, "temperature.pkl"))
    print(f"  Temperature saved to models/temperature.pkl")

    # ── Optimal Threshold Finding (Val Set) ──────────────────────────────────
    # model is already loaded (best checkpoint) and in eval mode from the
    # temperature scaling step above — no need to reload weights here.
    print("\nFinding optimal threshold on validation set (max F2)...")

    val_ds = TensorDataset(
        torch.FloatTensor(X_w[val_mask]),
        torch.FloatTensor(X_a_sc[val_mask]),
        torch.FloatTensor(y[val_mask]).view(-1, 1),
    )
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    v_preds, v_y = [], []
    with torch.no_grad():
        for bw, ba, by in val_loader:
            bw, ba, by = bw.to(device), ba.to(device), by.to(device)
            logits, _, _ = model(bw, ba)
            probs = torch.sigmoid(logits / T)
            v_preds.extend(probs.cpu().numpy().flatten())
            v_y.extend(by.cpu().numpy().flatten())

    v_preds_arr = np.array(v_preds)
    v_y_arr     = np.array(v_y)

    val_precision, val_recall, val_thresholds = precision_recall_curve(v_y_arr, v_preds_arr)
    # F2 appropriate here because missing an outbreak is worse than false alarm
    f2_scores = (5 * val_precision * val_recall) / (4 * val_precision + val_recall + 1e-8)
    best_f2_idx = np.argmax(f2_scores)
    optimal_threshold = val_thresholds[best_f2_idx]

    print(f"Optimal threshold (max F2 on val): {optimal_threshold:.4f}")
    print(f"  Val precision at threshold: {val_precision[best_f2_idx]:.3f}")
    print(f"  Val recall at threshold:    {val_recall[best_f2_idx]:.3f}")

    # ── Test evaluation ──────────────────────────────────────────────────────
    print("\nRunning test set evaluation...")
    # model already loaded and in eval mode
    test_ds = TensorDataset(
        torch.FloatTensor(X_w[test_mask]),
        torch.FloatTensor(X_a_sc[test_mask]),
        torch.FloatTensor(y[test_mask]).view(-1, 1),
    )
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    test_preds, test_y = [], []
    with torch.no_grad():
        for bw, ba, by in test_loader:
            bw, ba, by = bw.to(device), ba.to(device), by.to(device)
            logits, _, _ = model(bw, ba)
            # Apply temperature scaling before sigmoid
            scaled_probs = torch.sigmoid(logits / T)
            test_preds.extend(scaled_probs.cpu().numpy().flatten())
            test_y.extend(by.cpu().numpy().flatten())

    test_preds_arr = np.array(test_preds)
    test_y_arr     = np.array(test_y)
 
    if test_y_arr.sum() > 0:
        test_ap  = average_precision_score(test_y_arr, test_preds_arr)
        test_auc = roc_auc_score(test_y_arr, test_preds_arr)
 
        # Re-calculate P/R curve on test
        test_precision, test_recall, test_thresholds = precision_recall_curve(test_y_arr, test_preds_arr)
 
        # Use optimal_threshold from Val
        t_idx = np.argmin(np.abs(test_thresholds - optimal_threshold))
        p_at_opt = test_precision[t_idx]
        r_at_opt = test_recall[t_idx]
 
        print(f"\n-- Test Set Evaluation --")
        print(f"  Test AP:               {test_ap:.4f}")
        print(f"  Test AUC-ROC:          {test_auc:.4f}")
        print(f"  Positives in test:     {int(test_y_arr.sum())} / {len(test_y_arr)}")
        print(f"  At optimal threshold {optimal_threshold:.4f}:")
        print(f"    Precision:           {p_at_opt:.3f}")
        print(f"    Recall:              {r_at_opt:.3f}")

        # What threshold actually captures something on test?
        for i, (p, r, t) in enumerate(zip(test_precision, test_recall, test_thresholds)):
            if r >= 0.5:
                print(f"\nFirst test threshold with recall >= 0.5: {t:.4f}")
                print(f"  Precision at that point: {p:.4f}")
                break
    else:
        print(f"\n-- Test Set Evaluation --")
        print(f"  Positives in test:     0 / {len(test_y_arr)}")
        print(f"  (Skipping AP/AUC calculation as test set has no positive samples)")
 
    print("\nTest score distribution:")
    print(f"  Max:    {test_preds_arr.max():.4f}")
    print(f"  Mean:   {test_preds_arr.mean():.4f}")
    print(f"  Median: {np.median(test_preds_arr):.4f}")
    print(f"  >0.05:  {(test_preds_arr > 0.05).sum()}")
    print(f"  >0.1:   {(test_preds_arr > 0.1).sum()}")
    print(f"  >0.2:   {(test_preds_arr > 0.2).sum()}")

    # ── Save metadata (JSON) ─────────────────────────────────────────────────
    metadata = {
        "pipeline_version": PIPELINE_VERSION,
        "weather_features": WEATHER_FEATURES,
        "agro_features": AGRO_FEATURES,
        "seq_len": SEQ_LEN,
        "num_epochs": 60,
        "optimal_threshold": float(optimal_threshold),
        "augmentation": {
            "factor": AUGMENT_FACTOR,
            "noise_std": AUGMENT_NOISE_STD,
            "noise_features": _NOISE_FEATURE_NAMES,
        },
        "loss_weights": {
            "bce": 1.0,
            "focal": 0.5,
            "agronomic_monotonicity": 0.1,
            "mono_warmup_epochs": 20,
            "mono_batch_subsample": 4,
            "causal_consistency": 0.0,
        },
        "train_years": [2005, 2018],
        "val_years": [2019, 2021],
        "test_years": [2022, 2024],
    }
    meta_path = os.path.join(MODEL_DIR, "v11_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_path}")


 
    print("\nTest score distribution:")
    print(f"  Max:    {test_preds_arr.max():.4f}")
    print(f"  Mean:   {test_preds_arr.mean():.4f}")
    print(f"  Median: {np.median(test_preds_arr):.4f}")
    print(f"  >0.05:  {(test_preds_arr > 0.05).sum()}")
    print(f"  >0.1:   {(test_preds_arr > 0.1).sum()}")
    print(f"  >0.2:   {(test_preds_arr > 0.2).sum()}")


if __name__ == "__main__":
    train()