import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
LOGS_DIR   = os.path.join(BASE_DIR, "logs")

RAW_DATA_PATH  = os.path.join(
    BASE_DIR, "..", "..", "..", "..", "v5", "data", "raw",
    "POWER_Point_Daily_20050101_20241231_016d54N_069d78E_LST.csv"
)
OUTBREAK_EVENTS_PATH = (
    "e:/crop-disease-outbreak-prediction-system-feature-zip-changes"
    "/crop-disease-outbreak-prediction-system-feature-zip-changes"
    "/model_train/research_comp/evidence_base/outbreak_events"
    "/red_rot_outbreak_events.csv"
)
MODEL_PATH     = os.path.join(MODELS_DIR, "v6_gru_model.keras")
THRESHOLD_PATH = os.path.join(MODELS_DIR, "v6_threshold.txt")

# ─── Chronological Splits ─────────────────────────────────────────────────────
TRAIN_YEARS = (2005, 2018)
VAL_YEARS   = (2019, 2021)
TEST_YEARS  = (2022, 2024)

# ─── Sequence Configuration ───────────────────────────────────────────────────
SEQUENCE_LENGTH = 14          # 14-day sliding window per sample

# ─── Label Configuration (TRUE Onset Labels) ─────────────────────────────────
# A day t is labelled 1 if an outbreak peak_start falls in [t+0, t+7] to match evaluator
LEAD_MIN_DAYS = 0
LEAD_MAX_DAYS = 7

# ─── Feature Set (biologically validated, 5-paper consensus) ──────────────────
# Raw columns required from NASA POWER data
RAW_FEATURES = ["T2M", "T2M_MIN", "T2M_MAX", "RH2M", "PRECTOTCORR"]

# Engineered sequence features (computed per-day BEFORE windowing)
# All rolling features use .shift(1) to prevent look-ahead leakage.
SEQUENCE_FEATURES = [
    "T2M",                  # Raw daily temp
    "T2M_MIN",              # Raw daily min temp
    "RH2M",                 # Raw daily humidity
    "PRECTOTCORR",          # Raw daily rainfall
    "T2M_roll7",            # 7-day rolling mean temp
    "T2M_MIN_lag15",        # 15-day lag of T2M_MIN (SMRA validated R²=0.82–0.87)
    "RH2M_roll7",           # 7-day rolling mean humidity
    "RH2M_roll14",          # 14-day rolling mean humidity
    "RH2M_trend_slope",     # 5-day linear slope of RH2M (rising humidity phase)
    "PREC_sum14",           # 14-day accumulated rainfall
    "PREC_sum3",            # 3-day recent rainfall (spike event)
    "delta_RH2M",           # Daily change in humidity (phase transition signal)
    "monsoon_day",          # Days since monsoon onset (July 1 = day 1)
    "temp_suitability",     # 1 if T2M_roll7 in [29, 33] else 0
]

# ─── Model Hyperparameters ────────────────────────────────────────────────────
GRU_UNITS      = 64
DROPOUT_RATE   = 0.3
LEARNING_RATE  = 1e-3
BATCH_SIZE     = 32
MAX_EPOCHS     = 100
PATIENCE       = 10          # Early stopping patience

TARGET = "outbreak_onset"

# ─── Ensure Directories ───────────────────────────────────────────────────────
for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)
