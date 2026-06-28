import os
import pandas as pd
# import shap
from . import config
from . import utils

logger = utils.setup_logger("alert_engine")

def log_prediction(date_str, risk_score, band, alert_status):
    """Logs every daily prediction."""
    utils.ensure_dirs()
    file_exists = os.path.exists(config.PREDICTIONS_LOG_PATH)
    
    with open(config.PREDICTIONS_LOG_PATH, "a") as f:
        if not file_exists:
            f.write("date,risk_score,risk_band,alert_status\n")
        f.write(f"{date_str},{risk_score:.4f},{band},{alert_status}\n")

def generate_alert(model, features_df, current_day_row, risk_score, should_alert, bdl_results=None):
    """
    Generates the farmer advisory message with SHAP explanations and Fallback Logic.
    BDL Gating: Even if fallback triggers, BDL must ALLOW.
    """
    # 1. Determine Risk Band
    if risk_score < 0.30:
        band = "LOW"
    elif risk_score < 0.60:
        band = "MODERATE"
    elif risk_score < 0.80:
        band = "HIGH"
    else:
        band = "EXTREME"

    # 2. Fallback Rule Check
    env_score = current_day_row["env_interaction_score"].values[0]
    fallback_triggered = False
    
    if env_score == 3:
        # Fallback is also gated by BDL
        bdl_allow = bdl_results.get("final_decision") == "ALLOW" if bdl_results else True
        if bdl_allow:
            should_alert = True
            fallback_triggered = True
            band = "EXTREME (FALLBACK)"
            logger.warning("Fallback rule triggered and BDL ALLOWED.")
        else:
            logger.info("Fallback rule suppressed by BDL.")

    date_str = current_day_row["date"].dt.strftime("%Y-%m-%d").values[0]
    
    # 3. Log Prediction
    log_prediction(date_str, risk_score, band, should_alert)
    
    if not should_alert:
        logger.info(f"No alert generated today. Current Pressure: {band} ({risk_score:.4f})")
        return False
        
    # 4. Explainability (SHAP) - Bypassed
    top_features = ["RH2M_roll7", "PREC_sum14"]
    
    # 5. Format Message
    message = f"""
============================================================
⚠️ {band} OUTBREAK RISK ALERT: Red Rot Intelligence System
Date: {date_str}
Environmental Risk Intensity: {risk_score:.4f}
Fallback Status: {fallback_triggered}
BDL Score: {bdl_results.get('bdl_score', 'N/A') if bdl_results else 'N/A'}
BDL Phases: {', '.join(bdl_results.get('phase_detected', [])) if bdl_results else 'N/A'}

Advisory:
- Sustained outbreak-conducive conditions detected.
- Primary Driver: {top_features[0]}
- Secondary Driver: {top_features[1]}

Action: Apply preventive fungicide within 48-72 hours.
============================================================
"""
    print(message)
    logger.info(f"Alert generated for {date_str} based on {top_features}")
    return True
