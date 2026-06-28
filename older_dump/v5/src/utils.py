import os
import pandas as pd
import json
import logging
from . import config

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

logger = setup_logger("utils")

def ensure_dirs():
    """Ensure all required directories exist."""
    dirs = [config.DATA_DIR, config.MODELS_DIR, config.LOGS_DIR, config.OUTPUTS_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def load_raw_data(path=config.RAW_DATA_PATH):
    """Load raw dataset and parse initial dates."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data not found at {path}")
    
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path, skiprows=14)
    
    # Parse dates
    df["date"] = pd.to_datetime(df["YEAR"].astype(str) + df["DOY"].astype(str).str.zfill(3), format="%Y%j")
    df = df.sort_values("date").drop_duplicates(subset="date")
    
    return df

def save_threshold(threshold_value, path=config.THRESHOLD_PATH):
    ensure_dirs()
    with open(path, "w") as f:
        f.write(str(threshold_value))
    logger.info(f"Saved threshold {threshold_value} to {path}")

def load_threshold(path=config.THRESHOLD_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError("Threshold file not found. Run train mode first.")
    with open(path, "r") as f:
        val = float(f.read().strip())
    return val

def save_model_json(model, path=config.MODEL_PATH):
    ensure_dirs()
    model.save_model(path)
    logger.info(f"Saved XGBoost model to {path}")
