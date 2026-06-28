import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# File paths
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "POWER_Point_Daily_20050101_20241231_016d54N_069d78E_LST.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.json")
THRESHOLD_PATH = os.path.join(MODELS_DIR, "threshold.txt")
PREDICTIONS_LOG_PATH = os.path.join(LOGS_DIR, "predictions.csv")

# Train/Val/Test Splits
TRAIN_YEARS = (2005, 2018)
VAL_YEARS = (2019, 2021)
TEST_YEARS = (2022, 2024)

# Feature lists
RAW_FEATURES = ["T2M", "T2M_MIN", "T2M_MAX", "RH2M", "PRECTOTCORR"]

ENGINEERED_FEATURES = [
    "T2M_roll7",
    "PREC_sum14",
    "RH2M_roll7",
    "T2M_MIN_lag15",
    "PREC_lag15",
    "monsoon_flag",
    "temperature_suitability_flag",
    "env_interaction_score"
]

TARGET = "outbreak_risk"

# XGBoost Parameters
XGB_PARAMS = {
    "objective": "binary:logistic",
    "max_depth": 4,
    "eval_metric": "aucpr",
    "random_state": 42
}
