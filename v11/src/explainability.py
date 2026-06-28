"""
V11 KG-CTCN Explainability Module
--------------------------------------------------
Provides natural language explanations, attention weights,
and feature importance for risk predictions.

CHANGELOG (v11.1):
  - [CRITICAL]  load_metadata() switched from .txt parser to json.load().
                train.py now writes v11_metadata.json; the old .txt file no
                longer exists.
  - [CRITICAL]  Removed weather StandardScaler from main(). Weather features
                are already 90-day rolling Z-scores from the pipeline — applying
                StandardScaler again double-scales them. Agro scaler is now
                loaded from the saved agro_scaler.pkl (fitted on train split
                in train.py) rather than refitted here, ensuring exact
                reproducibility.
  - [CRITICAL]  warmup_mask filter applied before any indexing. Sequences
                whose windows overlap the first 90 warm-up rows contain NaN
                or garbage Z-scores and must not enter the model.
  - [BUG]       confidence output is now passed through torch.sigmoid() before
                .item(). model.py v11.1 returns conf_logit (pre-sigmoid);
                comparing a raw logit to 0.6 as if it were a probability is
                incorrect.
  - [BUG]       RH_persist_7d description corrected. The feature is a 0-7
                soft-flag accumulation (daily clip((RH-75)/15, 0, 1) summed
                over 7 days), not a count of days where RH > 85%. Threshold
                kept at > 4.0 (more than half-maximum) but description updated.
  - [BUG]       T2M_MIN_lag_15d threshold corrected from absolute °C range
                (22-28) to Z-score scale (> 0.5). The pipeline normalises
                T2M_MIN to a rolling Z-score; values of 22-28 are physically
                impossible on that scale and the condition never fired.
  - [DESIGN]    agro_data indexed by name via agro_features.index() rather
                than positional integer. Positional indexing silently breaks
                if AGRO_FEATURES order changes.
  - [DESIGN]    generate_explanation() now returns a structured dict instead
                of printing directly. The caller (main() or any other consumer)
                decides how to render it. This makes the function testable and
                usable in API / dashboard contexts.

CHANGELOG (v11.2):
  - [BUG]       main() fallback for missing target_date referenced
                df["risk_label"] which does not exist in v11_features.csv
                (risk_label is computed at runtime, not persisted). The fallback
                now calls assign_labels() from assign_causal_labels_v2 to
                derive labels on the fly, then selects the first positive date.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import joblib

from model import KGCTCN

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_metadata() -> dict:
    """
    Load model metadata from v11_metadata.json (written by train.py v11.1).
    Returns the full metadata dict including feature lists, seq_len, and
    pipeline_version.
    """
    meta_path = os.path.join(MODEL_DIR, "v11_metadata.json")
    with open(meta_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Explanation generation
# ---------------------------------------------------------------------------

def generate_explanation(
    date: pd.Timestamp,
    risk_prob: float,
    confidence: float,
    weather_seq: np.ndarray,
    agro_row: np.ndarray,
    weather_features: list,
    agro_features: list,
) -> dict:
    """
    Build a structured explanation for a single prediction.

    All feature values are expected in their pipeline-normalised units:
      - Base weather columns (T2M, RH2M, …): 90-day rolling Z-scores
      - KG-derived features (RH_persist_7d, Rain_sum_*, …): natural units
      - Agro features: raw units (variety_susceptibility 1-5, crop_age days)

    Args:
        date            : prediction date
        risk_prob       : sigmoid risk probability in [0, 1]
        confidence      : sigmoid confidence score in [0, 1]
        weather_seq     : (seq_len, num_weather_features) — unscaled pipeline output
        agro_row        : (num_agro_features,) — unscaled pipeline output
        weather_features: ordered list of weather feature names
        agro_features   : ordered list of agro feature names

    Returns:
        dict with keys: date, risk_class, risk_prob, confidence_class,
        confidence, drivers, lead_time_days
    """
    # ── Risk and confidence classification ───────────────────────────────────
    risk_class = (
        "HIGH"   if risk_prob >= 0.7 else
        "MEDIUM" if risk_prob >= 0.3 else
        "LOW"
    )
    conf_class = "High" if confidence >= 0.6 else "Low"

    # ── Feature extraction at time t (last step in sequence) ─────────────────
    def _get(name: str) -> float:
        return float(weather_seq[-1, weather_features.index(name)])

    rh_persist  = _get("RH_persist_7d")     # natural unit: 0.0 – 7.0
    rain_7d     = _get("Rain_sum_7d")        # natural unit: mm
    t2m_lag_z   = _get("T2M_MIN_lag_15d")   # Z-score (normalised by pipeline)
    monsoon     = _get("Monsoon_ind")        # 0 / 1

    # Agro features accessed by name — safe against future reordering
    variety_susc  = float(agro_row[agro_features.index("variety_susceptibility")])
    crop_age_days = float(agro_row[agro_features.index("crop_age_days")])

    # ── Driver assessment ─────────────────────────────────────────────────────
    drivers = []

    # RH_persist_7d: 0-7 accumulation of daily soft RH intensity flags
    # (each day contributes clip((RH-75)/15, 0, 1); max = 7.0 at RH ≥ 90% all week)
    # Threshold > 4.0 = more than half the week had meaningful humidity pressure.
    if rh_persist > 4.0:
        drivers.append(
            f"Sustained high humidity pressure: RH persistence score {rh_persist:.1f}/7.0 "
            f"(threshold 4.0) — indicates elevated fungal colonisation conditions "
            f"for more than half the past week."
        )

    # Rain_sum_7d: raw mm accumulated over 7 days
    if rain_7d > 10.0:
        drivers.append(
            f"High weekly rainfall: {rain_7d:.1f} mm accumulated over 7 days — "
            f"promotes inoculum dispersal and field saturation."
        )

    # T2M_MIN_lag_15d: Z-score of 15-day lagged minimum temperature.
    # Positive Z-score = warmer-than-normal nights 15 days ago (favours pathogen growth).
    # Negative Z-score = cooler-than-normal (potential immunity stress trigger).
    # Both extremes are biologically relevant; threshold ±0.5 std.
    if t2m_lag_z > 0.5:
        drivers.append(
            f"Warm night temperatures 15 days ago: T2M_MIN lag Z-score {t2m_lag_z:+.2f} "
            f"— above-normal minimum temperatures favour Colletotrichum falcatum activity."
        )
    elif t2m_lag_z < -0.5:
        drivers.append(
            f"Cool night temperatures 15 days ago: T2M_MIN lag Z-score {t2m_lag_z:+.2f} "
            f"— below-normal minimum temperatures may have stressed crop immunity."
        )

    # Variety susceptibility: 0 (resistant) – 2 (susceptible)
    if variety_susc >= 2:
        drivers.append(
            f"Highly susceptible variety (score {int(variety_susc)}/2) — "
            f"agronomic vulnerability amplifies weather-driven risk."
        )
    elif variety_susc == 1:
        drivers.append(
            f"Moderately susceptible variety (score {int(variety_susc)}/2) — "
            f"intermediate host vulnerability."
        )

    # Crop age: peak Red Rot susceptibility typically 4-8 months after planting
    if 120 <= crop_age_days <= 240:
        drivers.append(
            f"Crop age {int(crop_age_days)} days — within peak susceptibility window "
            f"(120-240 days post-planting)."
        )

    if monsoon:
        drivers.append("Active monsoon period (Jun–Sep) — baseline risk elevated.")

    if not drivers:
        drivers.append("No individual driver exceeded alert threshold.")

    return {
        "date":             date.date().isoformat(),
        "risk_class":       risk_class,
        "risk_prob":        round(risk_prob, 4),
        "confidence_class": conf_class,
        "confidence":       round(confidence, 4),
        "drivers":          drivers,
        "lead_time_days":   "3-7",
    }


def print_explanation(exp: dict) -> None:
    """Render a structured explanation dict to stdout."""
    print("=" * 60)
    print(f"Prediction for {exp['date']}")
    print(f"Risk:       {exp['risk_class']} ({exp['risk_prob']:.4f})")
    print(f"Confidence: {exp['confidence_class']} ({exp['confidence']:.4f})")
    print(f"Lead time:  {exp['lead_time_days']} days")
    print("\nPrimary drivers:")
    for driver in exp["drivers"]:
        print(f"  - {driver}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ── Load metadata ────────────────────────────────────────────────────────
    meta = load_metadata()
    weather_features = meta["weather_features"]
    agro_features    = meta["agro_features"]
    seq_len          = int(meta["seq_len"])

    # ── Load dataset and apply warmup filter ─────────────────────────────────
    df = pd.read_csv(os.path.join(DATA_DIR, "v11_features.csv"))
    df["date"] = pd.to_datetime(df["date"])

    # Exclude warm-up rows — sequences overlapping these contain NaN Z-scores
    df = df[df["warmup_mask"] == 0].reset_index(drop=True)

    # ── Select target date ───────────────────────────────────────────────────
    target_date = pd.to_datetime("2019-06-20")
    if target_date not in df["date"].values:
        # risk_label is NOT stored in v11_features.csv — it is computed at
        # runtime. Assign labels now so we can find the first positive day.
        import json as _json
        _meta_path = os.path.join(MODEL_DIR, "v11_metadata.json")
        with open(_meta_path) as _f:
            _meta = _json.load(_f)
        _gt_path = os.path.join(
            os.path.dirname(BASE_DIR),
            "research_comp", "evidence_base", "outbreak_events", "sangli_gt_v2.csv",
        )
        from assign_causal_labels_v2 import assign_labels
        _df_labeled = assign_labels(df, gt_path=_gt_path)
        pos_rows = _df_labeled[_df_labeled["risk_label"] == 1]
        if pos_rows.empty:
            raise ValueError(
                "Target date not found and no risk_label=1 rows exist after labeling. "
                "Check that sangli_gt_v2.csv covers the loaded date range."
            )
        target_date = pos_rows["date"].iloc[0]
        print(f"Target date not found; using first risk-positive day: {target_date.date()}")

    idx = df[df["date"] == target_date].index[0]

    # Guard: ensure the sequence window does not reach into warm-up territory
    if idx < seq_len:
        raise ValueError(
            f"Index {idx} for date {target_date.date()} is too close to the start of "
            f"the filtered dataset to build a {seq_len}-day sequence. "
            f"Choose a later date."
        )

    # ── Load model ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KGCTCN(len(weather_features), len(agro_features)).to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, "v11_kg_ctcn.pth"),
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    # ── Prepare features ─────────────────────────────────────────────────────
    # Weather: pipeline already Z-scored — NO additional StandardScaler.
    # Passing through a second scaler would double-normalise and corrupt the
    # causal_consistency_loss thresholds calibrated to natural units.
    w_raw = df[weather_features].values   # already normalised by pipeline
    a_raw = df[agro_features].values      # raw units — scaler applied below

    # Agro: load the scaler fitted on the training split in train.py.
    # Do NOT refit here — refitting on a different random state breaks
    # exact reproducibility with the saved model.
    a_sc = joblib.load(os.path.join(MODEL_DIR, "agro_scaler.pkl"))
    a_scaled = a_sc.transform(a_raw)

    # Build tensors for the target date
    X_w = torch.FloatTensor(
        w_raw[idx - seq_len + 1 : idx + 1]
    ).unsqueeze(0).to(device)             # (1, seq_len, num_weather_features)

    X_a = torch.FloatTensor(
        a_scaled[idx]
    ).unsqueeze(0).to(device)             # (1, num_agro_features)

    # ── Run inference ────────────────────────────────────────────────────────
    with torch.no_grad():
        _, risk_prob_t, conf_logit_t = model(X_w, X_a)

    # model.py v11.1 returns conf_logit (pre-sigmoid) — apply sigmoid here
    risk_prob  = risk_prob_t.item()
    confidence = torch.sigmoid(conf_logit_t).item()

    # ── Generate and print explanation ────────────────────────────────────────
    # Raw (pipeline-normalised) values are passed to generate_explanation so
    # it can apply feature-appropriate thresholds (Z-score for base weather,
    # natural units for KG-derived and agro features).
    raw_weather_seq = w_raw[idx - seq_len + 1 : idx + 1]
    raw_agro_row    = a_raw[idx]

    explanation = generate_explanation(
        date=target_date,
        risk_prob=risk_prob,
        confidence=confidence,
        weather_seq=raw_weather_seq,
        agro_row=raw_agro_row,
        weather_features=weather_features,
        agro_features=agro_features,
    )

    print_explanation(explanation)


if __name__ == "__main__":
    main()