import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src import config, utils, features, decision_engine, alert_engine

# Need to handle XGBoost vs HistGradientBoosting for testing
try:
    from xgboost import XGBClassifier
    MODEL_TYPE = "xgboost"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    import joblib
    MODEL_TYPE = "sklearn"

def run_backtest(start_year=2019, end_year=2024):
    """Simulates daily inference from start_year to end_year."""
    print(f"Starting V5 Backtest ({start_year}-{end_year})...")
    
    # 1. Load Model & Threshold
    if MODEL_TYPE == "xgboost":
        model = XGBClassifier()
        model.load_model(config.MODEL_PATH)
    else:
        model = joblib.load(config.MODEL_PATH.replace(".json", ".pkl"))
    
    # Use fixed threshold 0.60 as per final spec
    threshold = 0.60 
    
    # 2. Load Raw Data
    raw_df = utils.load_raw_data()
    
    # Filter to required range + 45 days warmup
    start_date = pd.Timestamp(year=start_year, month=1, day=1)
    end_date = pd.Timestamp(year=end_year, month=12, day=31)
    
    # Ensure we have enough data
    backtest_data = raw_df[(raw_df["date"] >= start_date - pd.Timedelta(days=45)) & 
                           (raw_df["date"] <= end_date)].copy()
    
    dates_to_test = backtest_data[backtest_data["date"] >= start_date]["date"].unique()
    results = []

    # 3. Daily Simulation Loop
    for current_date in tqdm(dates_to_test):
        # Slice last 45 days
        window = backtest_data[backtest_data["date"] <= current_date].tail(45).copy()
        
        # Guard: Need at least 28 days
        if len(window) < 28:
            continue
            
        # Feature Engineering
        feat_df = features.build_features(window)
        
        # Check for NaNs in last 3 days
        if feat_df[config.ENGINEERED_FEATURES].tail(3).isnull().values.any():
            results.append({"date": current_date, "risk_score": np.nan, "risk_band": "ABORTED", "alert": False})
            continue
            
        # Inference
        X_infer = feat_df.tail(14)[config.ENGINEERED_FEATURES]
        # We only need the model probabilities for the rows we want to test
        # However, to be consistent with 2-of-3 rule, we need at least last 3 days probabilities
        # Let's predict for the whole 14 day window to be safe
        probs = model.predict_proba(X_infer)[:, 1]
        
        last_14_days = feat_df.tail(14).copy()
        last_14_days["risk_score"] = probs
        
        current_score = probs[-1]
        
        # Decision
        decision_payload = decision_engine.evaluate_risk(last_14_days)
        should_alert = decision_payload["should_alert"]
        
        # Risk Band
        if current_score < 0.30: band = "LOW"
        elif current_score < 0.60: band = "MODERATE"
        elif current_score < 0.80: band = "HIGH"
        else: band = "EXTREME"
        
        results.append({
            "date": current_date,
            "risk_score": current_score,
            "risk_band": band,
            "alert": should_alert
        })
    
    # 4. Save Results
    results_df = pd.DataFrame(results)
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(config.OUTPUTS_DIR, "backtest_results.csv"), index=False)
    print(f"Backtest completed. Results saved to {config.OUTPUTS_DIR}/backtest_results.csv")
    return results_df

if __name__ == "__main__":
    run_backtest()
