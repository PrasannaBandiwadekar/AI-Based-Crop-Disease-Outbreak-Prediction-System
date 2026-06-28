import pandas as pd
from . import config
from . import utils
from . import biological_discrimination_layer as bdl

logger = utils.setup_logger("decision_engine")

def evaluate_risk(recent_features_df):
    """
    Evaluates risk based on the V5.1 Signal Confirmation Gate:
    1. ML_SIGNAL: 2-of-3 HIGH risk rule (>= 0.60)
    2. BDL_SIGNAL: bdl_score >= 0.70
    3. FINAL_ALERT: ML_SIGNAL AND BDL_SIGNAL
    """
    threshold = 0.60

    if len(recent_features_df) < 3:
        logger.warning("Not enough data for 2-of-3 rule.")
        return False
        
    # 1. Biological Discrimination Layer (BDL)
    bdl_results = bdl.calculate_bdl_score(recent_features_df)
    bdl_score = bdl_results["bdl_score"]
    bdl_allow = bdl_results["final_decision"] == "ALLOW"
    bdl_signal = bdl_score >= 0.70
    
    # 2. ML Signal (2-of-3 rule)
    last_3_days = recent_features_df.tail(3).copy()
    last_3_days["is_high"] = (last_3_days["risk_score"] >= threshold).astype(int)
    ml_signal = bool(last_3_days["is_high"].sum() >= 2)
    
    # 3. Final Decision
    final_alert = ml_signal and bdl_signal and bdl_allow
    
    logger.info(
        f"Decision Engine: ML_Signal={ml_signal}, BDL_Score={bdl_score}, "
        f"BDL_Allow={bdl_allow}, Final={final_alert}"
    )
    
    return {
        "should_alert": final_alert,
        "bdl_results": bdl_results
    }
