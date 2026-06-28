"""
build_domain_rules.py
---------------------
Converts the unified red rot knowledge model into a domain validation
rule system for the Crop Disease Outbreak Prediction System (v5).

Input : research_comp/knowledge_graph/red_rot/unified_model.json
Output: v5/evaluation/domain_rules.json

Run:
  python scratch/build_domain_rules.py
"""

import os
import json

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNIFIED_SRC = os.path.join(BASE, "research_comp", "knowledge_graph",
                            "red_rot", "unified_model.json")
OUT_DIR     = os.path.join(BASE, "v5", "evaluation")
OUT_PATH    = os.path.join(OUT_DIR, "domain_rules.json")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load unified model ────────────────────────────────────────────────────────
with open(UNIFIED_SRC, encoding="utf-8") as f:
    model = json.load(f)

thresholds = model["final_outbreak_logic"]["key_thresholds"]

# ── Domain Rules ──────────────────────────────────────────────────────────────
domain_rules = {

    # ── Meta ──────────────────────────────────────────────────────────────────
    "disease": model["disease"],
    "pathogen": model["pathogen"],
    "source": "research_comp/knowledge_graph/red_rot/unified_model.json",
    "evidence_basis": model["evidence_consensus"],

    # ── 1. Seasonal Rules ─────────────────────────────────────────────────────
    "season_rules": {
        "active_months": [7, 8, 9, 10],          # July–October
        "peak_fortnight": "August (2nd fortnight)",
        "onset_indicator_month": 7,               # Monsoon onset = July
        "valid_prediction_window": {
            "start_month": 6,                     # June (pre-monsoon buildup)
            "end_month": 11                       # November (post-monsoon tail)
        },
        "description": (
            "Red rot outbreaks are restricted to the monsoon and post-monsoon season. "
            "Disease incidence is 0 % in May-June, rises sharply from July, peaks in "
            "the 2nd fortnight of August, and tapers through October-November. "
            "Predictions outside months 6-11 should be treated as low-confidence."
        )
    },

    # ── 2. Lag Rules ─────────────────────────────────────────────────────────
    "lag_rules": {
        "primary_lag_feature": "T2M_MIN",
        "lag_days": int(thresholds["key_lag_window_days"]),
        "direction": "negative",
        "r_squared_range": thresholds["smra_r2_T2M_MIN"],
        "secondary_lag_feature": "RH2M",
        "secondary_lag_days": 15,
        "description": (
            "T2M_MIN (minimum temperature) in one 15-day window is NEGATIVELY correlated "
            "with disease incidence in the NEXT 15-day window (R² 0.82-0.87, SMRA validated). "
            "A LOWER T2M_MIN now predicts HIGHER disease incidence 15 days later. "
            "RH2M lag of 15 days is a validated secondary predictor. "
            "Models must include lag_15d versions of T2M_MIN and RH2M as mandatory features."
        ),
        "validation_rule": (
            "For a HIGH-risk prediction to be domain-consistent, "
            "T2M_MIN_lag_15d must be >= 24 deg C (warm monsoon minimum). "
            "If T2M_MIN_lag_15d < 20 deg C, the prediction is likely a false positive."
        )
    },

    # ── 3. Accumulation Rules ─────────────────────────────────────────────────
    "accumulation_rules": {
        "humidity": {
            "threshold_pct": 85,
            "severe_threshold_pct": 100,
            "min_days": 7,
            "feature": "RH2M",
            "rolling_features": ["RH2M_rolling_mean_7d", "RH2M_rolling_mean_14d"],
            "rule": "RH2M must exceed 85 % for at least 7 consecutive days to support outbreak risk."
        },
        "rainfall": {
            "threshold_mm_per_15d": float(thresholds["rainfall_peak_mm_15d"]),
            "threshold_mm_per_day": 10.0,
            "min_days": 5,
            "feature": "PRECTOTCORR",
            "rolling_features": [
                "PRECTOTCORR_rolling_sum_14d",
                "PRECTOTCORR_rolling_sum_28d"
            ],
            "rule": (
                "PRECTOTCORR must exceed 10 mm/day for at least 5 consecutive days, "
                "or 115 mm accumulated over 15 days, to confirm rain-driven inoculum dispersal."
            )
        },
        "temperature": {
            "range_C": thresholds["temperature_optimum_C"],
            "min_C": 24.0,
            "max_C": 33.1,
            "min_days": 14,
            "feature": "T2M",
            "rolling_features": [
                "T2M_rolling_mean_7d",
                "T2M_rolling_mean_14d",
                "T2M_rolling_mean_28d"
            ],
            "rule": (
                "T2M must remain within 24-33 deg C for at least 14 consecutive days "
                "to maintain conditions favorable for pathogen growth and infection."
            )
        }
    },

    # ── 4. Threshold Rules ────────────────────────────────────────────────────
    "threshold_rules": {
        "temperature": {
            "optimum_range_C": thresholds["temperature_optimum_C"],
            "absolute_max_C": float(thresholds["temperature_max_C"]),
            "absolute_min_C": float(thresholds["temperature_min_C"]),
            "feature_T2M": "29.4 - 31.0 deg C optimal",
            "feature_T2M_MAX": "< 36.9 deg C (disease present); peak ~33.1 deg C",
            "feature_T2M_MIN": ">= 26.4 deg C at peak incidence",
            "note": "Use ranges, not exact thresholds, to avoid overfitting."
        },
        "humidity": {
            "severe_outbreak_RH_pct": "85-100",
            "good_development_RH_pct": "79-92",
            "morning_RH_pct": "80-90",
            "evening_RH_pct": "50-76",
            "feature": "RH2M",
            "note": "Morning RH is the more reliable indicator than evening RH."
        },
        "rainfall": {
            "peak_15d_mm": float(thresholds["rainfall_peak_mm_15d"]),
            "daily_trigger_mm": 10.0,
            "active_season": "July to September (monsoon)",
            "feature": "PRECTOTCORR",
            "note": "Unseasonal heavy rain outside monsoon is also a risk flag."
        },
        "soil_inoculum": {
            "survival_days": thresholds["soil_inoculum_survival_days"],
            "note": "Inoculum from previous season remains viable for 60-90 days."
        }
    },

    # ── 5. Temporal Window Rules ──────────────────────────────────────────────
    "temporal_window_rules": {
        "rolling_windows_days": [int(d) for d in thresholds["key_rolling_windows_days"]],
        "lag_days": int(thresholds["key_lag_window_days"]),
        "windows": {
            "7d": {
                "features": ["T2M_rolling_mean_7d", "RH2M_rolling_mean_7d"],
                "role": "Short-term humidity and temperature persistence check."
            },
            "14d": {
                "features": [
                    "T2M_rolling_mean_14d",
                    "RH2M_rolling_mean_14d",
                    "PRECTOTCORR_rolling_sum_14d"
                ],
                "role": (
                    "Primary outbreak buildup window. "
                    "14-day accumulation of temp + humidity is the minimum "
                    "sustained condition needed before visible symptoms appear."
                )
            },
            "28d": {
                "features": [
                    "T2M_rolling_mean_28d",
                    "PRECTOTCORR_rolling_sum_28d"
                ],
                "role": (
                    "Epidemic onset predictor. "
                    "28-day climate trend captures the full monsoon buildup phase "
                    "and correlates with epidemic-level disease development."
                )
            },
            "15d_lag": {
                "features": ["T2M_MIN_lag_15d", "RH2M_lag_15d"],
                "role": (
                    "Primary prediction lag window (SMRA validated R² 0.82-0.87). "
                    "T2M_MIN 15 days ago is the strongest single predictor of "
                    "current disease incidence."
                )
            }
        }
    },

    # ── 6. Early Warning Rules ────────────────────────────────────────────────
    "early_warning_rules": {
        "lead_time_days": [3, 7],
        "latent_period_days": {
            "standard": thresholds["latent_period_days_standard"],
            "elevated_CO2": thresholds["latent_period_days_elevated_CO2"]
        },
        "trigger_conditions": [
            {
                "id": "EW-01",
                "name": "Temperature-in-range sustained",
                "condition": "T2M rolling_mean_14d >= 29.4 AND T2M rolling_mean_14d <= 31.0",
                "lead_days": 7,
                "source": "SMRA models; Minnatullah 2025"
            },
            {
                "id": "EW-02",
                "name": "High humidity sustained",
                "condition": "RH2M rolling_mean_7d >= 85",
                "lead_days": 7,
                "source": "Red rot mgmt 2022; Duttamajumder 2008"
            },
            {
                "id": "EW-03",
                "name": "Monsoon onset detected",
                "condition": (
                    "month == 7 AND PRECTOTCORR_rolling_sum_14d >= 50 "
                    "AND T2M_MIN >= 24"
                ),
                "lead_days": 14,
                "source": "Saharan 1992; Minnatullah 2025"
            },
            {
                "id": "EW-04",
                "name": "Rainfall pulse — inoculum dispersal risk",
                "condition": (
                    "PRECTOTCORR > 10 mm/day for >= 5 consecutive days "
                    "AND month in [7,8,9,10]"
                ),
                "lead_days": 5,
                "source": "Minnatullah 2025; Red rot mgmt 2022"
            },
            {
                "id": "EW-05",
                "name": "T2M_MIN lag alert",
                "condition": (
                    "T2M_MIN_lag_15d >= 26 "
                    "AND RH2M_lag_15d >= 80"
                ),
                "lead_days": 15,
                "source": "Saharan 1992 SMRA (R² 0.82-0.87)"
            },
            {
                "id": "EW-06",
                "name": "Combined high-risk composite",
                "condition": (
                    "EW-01 AND EW-02 AND month in [7,8,9,10]"
                ),
                "lead_days": 7,
                "source": "Consensus — all 5 red rot papers"
            }
        ],
        "description": (
            "Early warning is triggered when 2+ conditions from EW-01 to EW-05 are met. "
            "EW-06 (composite) alone is sufficient for a HIGH-risk alert. "
            "Lead time of 3-7 days aligns with the shortest validated latent period (7 days spindle). "
            "The 15-day lag rule (EW-05) provides the longest lead for field intervention."
        )
    },

    # ── 7. Domain Validation Logic ────────────────────────────────────────────
    "validation_logic": {
        "description": (
            "Rules to validate ML model predictions against domain knowledge. "
            "A model output is domain_consistent only if ALL mandatory checks pass."
        ),
        "high_risk_checks": [
            {
                "id": "CHECK-01",
                "name": "season_check",
                "rule": "current_month must be in [6, 7, 8, 9, 10, 11]",
                "mandatory": True,
                "failure_action": "mark_as_domain_inconsistent",
                "reason": "Red rot outbreaks are biologically impossible outside monsoon season."
            },
            {
                "id": "CHECK-02",
                "name": "lag_check",
                "rule": (
                    "T2M_MIN_lag_15d >= 24 deg C "
                    "AND RH2M_lag_15d >= 79 %"
                ),
                "mandatory": True,
                "failure_action": "downgrade_to_medium_risk",
                "reason": "SMRA validated: low T2M_MIN or RH2M 15 days ago contradicts high-risk prediction."
            },
            {
                "id": "CHECK-03",
                "name": "accumulation_check",
                "rule": (
                    "At least ONE of: "
                    "(RH2M_rolling_mean_7d >= 85) OR "
                    "(PRECTOTCORR_rolling_sum_14d >= 50 mm) OR "
                    "(T2M_rolling_mean_14d in [29, 31])"
                ),
                "mandatory": True,
                "failure_action": "mark_as_domain_inconsistent",
                "reason": "High-risk without any accumulated environmental stress is biologically implausible."
            },
            {
                "id": "CHECK-04",
                "name": "temporal_window_check",
                "rule": (
                    "At least ONE rolling window feature must exceed its threshold: "
                    "RH2M_rolling_mean_14d >= 79 OR "
                    "T2M_rolling_mean_14d >= 29.4 OR "
                    "PRECTOTCORR_rolling_sum_28d >= 100 mm"
                ),
                "mandatory": False,
                "failure_action": "flag_for_review",
                "reason": "Sustained conditions over 14-28 days are required for epidemic-level risk."
            }
        ],
        "low_risk_override": {
            "rule": (
                "If month NOT in [6,7,8,9,10,11] AND T2M < 24 AND RH2M < 70, "
                "override model prediction to LOW regardless of model output."
            ),
            "reason": "Biologically impossible outbreak conditions."
        },
        "failure_action": "mark_as_domain_inconsistent",
        "output_flags": [
            "domain_consistent",
            "domain_inconsistent",
            "domain_downgraded",
            "domain_flagged_for_review"
        ]
    },

    "confidence_level": "high"
}

# ── Write output ──────────────────────────────────────────────────────────────
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(domain_rules, f, indent=4, ensure_ascii=False)

print(f"\n[DONE] Domain rules saved -> {OUT_PATH}")
print(f"       Disease      : {domain_rules['disease']}")
print(f"       Season months: {domain_rules['season_rules']['active_months']}")
print(f"       Lag days     : {domain_rules['lag_rules']['lag_days']}d (feature: {domain_rules['lag_rules']['primary_lag_feature']})")
print(f"       Rolling wins : {domain_rules['temporal_window_rules']['rolling_windows_days']}")
print(f"       EW triggers  : {len(domain_rules['early_warning_rules']['trigger_conditions'])}")
print(f"       Val checks   : {len(domain_rules['validation_logic']['high_risk_checks'])}")
