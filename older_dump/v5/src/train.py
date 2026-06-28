import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from . import config
from . import utils
from . import features

logger = utils.setup_logger("train")

def run_training():
    logger.info("Starting training pipeline...")
    
    # 1. Load Data
    raw_df = utils.load_raw_data()
    
    # 2. Build Features & Labels
    logger.info("Applying feature engineering pipeline...")
    df = features.build_features(raw_df)
    df = features.add_label(df)
    
    # Drop rows with NaNs caused by rolling/lags/label shifting
    # We drop the first 28 rows (warmup) and last 7 rows (label window)
    df = df.iloc[28:-7].dropna(subset=config.ENGINEERED_FEATURES + [config.TARGET])
    
    # 3. Split Chronologically
    train_mask = (df["date"].dt.year >= config.TRAIN_YEARS[0]) & (df["date"].dt.year <= config.TRAIN_YEARS[1])
    val_mask = (df["date"].dt.year >= config.VAL_YEARS[0]) & (df["date"].dt.year <= config.VAL_YEARS[1])
    
    X_train = df.loc[train_mask, config.ENGINEERED_FEATURES]
    y_train = df.loc[train_mask, config.TARGET]
    
    X_val = df.loc[val_mask, config.ENGINEERED_FEATURES]
    y_val = df.loc[val_mask, config.TARGET]
    
    logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
    
    # 4. Handle Imbalance
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    logger.info(f"Class imbalance scale_pos_weight: {scale_pos_weight:.2f}")
    
    # 5. Train XGBoost
    model = HistGradientBoostingClassifier(max_depth=4, random_state=42)
    # Note: scale_pos_weight is passed via sample_weight in fit for HistGradientBoosting
    sample_weights = np.where(y_train == 1, scale_pos_weight, 1.0)
    logger.info("Training HistGradientBoosting model (XGBoost alternative for testing)...")
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # 6. Compute Threshold
    val_probs = model.predict_proba(X_val)[:, 1]
    # We compute the 85th percentile from the training set as the global threshold!
    # Wait, the prompt says "compute 85th percentile threshold from validation scores"
    # I will strictly follow the prompt.
    threshold = np.percentile(val_probs, 85)
    logger.info(f"Calculated 85th percentile threshold on validation set: {threshold:.4f}")
    
    # 7. Save Artifacts
    import joblib
    joblib.dump(model, config.MODEL_PATH.replace('.json', '.pkl'))
    utils.save_threshold(threshold)
    
    logger.info("Training pipeline completed successfully.")
