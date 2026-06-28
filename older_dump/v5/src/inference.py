import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from . import config
from . import utils
from . import features
from . import decision_engine
from . import alert_engine

logger = utils.setup_logger("inference")

def run_inference():
    logger.info("Starting inference pipeline...")
    
    # 1. Load Model & Threshold
    try:
        import joblib
        model = joblib.load(config.MODEL_PATH.replace('.json', '.pkl'))
        threshold = utils.load_threshold()
        logger.info(f"Loaded XGBoost model and threshold ({threshold:.4f})")
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}. Run --mode train first.")
        return
        
    # 2. Load latest 45-day window
    # In production, this would fetch from a live database/API.
    # For this simulation, we load the raw dataset and take the last 45 days.
    raw_df = utils.load_raw_data()
    last_45_days = raw_df.tail(45).copy()
    
    # 3. Generate Features (IDENTICAL to training)
    logger.info("Applying feature engineering pipeline to 45-day window...")
    features_df = features.build_features(last_45_days)
    
    # 3.5 Runtime Guard System (CRITICAL)
    # Ensure system enforces no prediction on incomplete temporal state
    calendar_days = (last_45_days["date"].max() - last_45_days["date"].min()).days + 1
    if calendar_days < 28:
        logger.error(f"ABORT PREDICTION: Temporal span too short ({calendar_days} days). Requires >= 28 continuous days for safe lag computation.")
        return
        
    if features_df[config.ENGINEERED_FEATURES].tail(3).isnull().values.any():
        logger.error("ABORT PREDICTION: Irrecoverable NaNs detected in recent inference window. Sensor blackout exceeded safe interpolation limits.")
        return

    
    # The last row represents 'today'
    current_day_row = features_df.tail(1)
    date_str = current_day_row["date"].dt.strftime("%Y-%m-%d").values[0]
    
    # We need the last 3 days with features computed for the decision engine
    last_3_days_features = features_df.tail(3).copy()
    
    # 4. Predict Raw Risk Scores
    X_infer = last_3_days_features[config.ENGINEERED_FEATURES]
    raw_scores = model.predict_proba(X_infer)[:, 1]
    
    last_3_days_features["risk_score"] = raw_scores
    current_risk_score = raw_scores[-1]
    
    logger.info(f"Prediction for {date_str} - Risk Score: {current_risk_score:.4f}")
    
    # 5. Decision Engine (with 14-day BDL context)
    decision_payload = decision_engine.evaluate_risk(features_df.tail(14))
    should_alert = decision_payload["should_alert"]
    bdl_results = decision_payload["bdl_results"]
    
    # 6. Alert Engine
    alert_engine.generate_alert(
        model, 
        features_df, 
        current_day_row, 
        current_risk_score, 
        should_alert,
        bdl_results=bdl_results
    )
    
    logger.info("Inference pipeline completed.")
