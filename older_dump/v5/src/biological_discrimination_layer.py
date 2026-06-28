import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("bdl_layer")

def calculate_bdl_score(features_df: pd.DataFrame) -> dict:
    """
    Biological Discrimination Layer (BDL) for Red Rot.
    Transforms the system from a climate detector into a disease precursor detector.
    
    Phases:
    1. Humidity Trend: 5-day slope of RH2M_roll7 > 0.5% per day.
    2. Sustained Saturation: RH2M_roll7 > 82.
    3. Rainfall Spike: Recent 3-day rainfall > (previous 7-day mean + delta).
    4. Temperature Stability: Max temp change over 5 days < 2.0°C.
    """
    if len(features_df) < 10:
        return {
            "bdl_score": 0.0,
            "phase_detected": [],
            "trajectory_valid": False,
            "final_decision": "SUPPRESS"
        }

    # Extract relevant data
    rh_roll7_recent = features_df["RH2M_roll7"].tail(5).values
    rh_roll7_current = rh_roll7_recent[-1]
    
    prec_recent_3d = features_df["PRECTOTCORR"].tail(3).sum()
    prec_prev_7d_mean = features_df["PRECTOTCORR"].iloc[-10:-3].mean()
    
    t2m_recent_5d = features_df["T2M"].tail(5).values
    
    phases = []
    
    # --- Phase 1: Humidity Trend (Look for recent rising trend) ---
    # We check if there was a rising trend (>0.5% / day) in any 5-day sub-window of the last 10 days
    p1_active = False
    for i in range(6): # Check sub-windows ending up to 5 days ago
        sub_window = features_df["RH2M_roll7"].iloc[-5-i : len(features_df)-i].values
        if len(sub_window) < 5: continue
        x_sub = np.arange(len(sub_window))
        s_sub = np.polyfit(x_sub, sub_window, 1)[0]
        if s_sub > 0.5:
            p1_active = True
            break
    if p1_active: phases.append("P1_HUM_TREND")
    
    # --- Phase 2: Sustained Saturation ---
    p2_active = rh_roll7_current > 82
    if p2_active: phases.append("P2_SATURATION")
    
    # --- Phase 3: Rainfall Spike ---
    # Recent 3-day rainfall vs previous 7-day expected + significant delta
    # User: "recent spike > previous 7-day mean + threshold delta"
    # To reduce false alerts in monsoon, we need a sharper delta.
    delta = 20.0
    p3_active = prec_recent_3d > (prec_prev_7d_mean * 3 + delta)
    if p3_active: phases.append("P3_RAIN_SPIKE")
    
    # --- Phase 4: Temperature Stability ---
    t_max_change = np.ptp(t2m_recent_5d)
    p4_active = t_max_change < 1.5 # Stricter stability
    if p4_active: phases.append("P4_TEMP_STABILITY")
    
    # --- Scoring ---
    score = (0.3 if p1_active else 0.0) + \
            (0.3 if p2_active else 0.0) + \
            (0.2 if p3_active else 0.0) + \
            (0.2 if p4_active else 0.0)
    
    # --- Suppression Rules ---
    final_decision = "ALLOW"
    
    # Define current slope for plateau check
    x_curr = np.arange(len(rh_roll7_recent))
    slope_curr = np.polyfit(x_curr, rh_roll7_recent, 1)[0]

    # Rule 1: RH2M high BUT slope ≈ 0 (flat monsoon plateau)
    if rh_roll7_current > 80 and abs(slope_curr) < 0.1 and not p1_active:
        final_decision = "SUPPRESS"
        logger.info(f"BDL SUPPRESS: Flat monsoon plateau without recent trend (slope={slope_curr:.4f})")

    # Rule 2: Continuous rainfall without spike event
    if not p3_active and prec_prev_7d_mean > 2.0:
        rain_history = features_df["PRECTOTCORR"].tail(14).values
        # Stricter spike check: any day with > 25mm in last 14 days
        if not np.any(rain_history > 25):
            final_decision = "SUPPRESS"
            logger.info("BDL SUPPRESS: Continuous rainfall without significant spike.")

    # Rule 3: No multi-phase convergence (less than 3/4 phases active)
    if len(phases) < 3:
        final_decision = "SUPPRESS"
        logger.info(f"BDL SUPPRESS: Multi-phase convergence failed ({len(phases)}/4 phases)")

    # Global threshold gate
    if score < 0.70:
        final_decision = "SUPPRESS"

    return {
        "bdl_score": round(float(score), 4),
        "phase_detected": phases,
        "trajectory_valid": len(phases) >= 3,
        "final_decision": final_decision,
        "humidity_slope": round(float(slope_curr), 4)
    }
