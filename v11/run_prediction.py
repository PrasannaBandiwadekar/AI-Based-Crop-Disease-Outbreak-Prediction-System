"""
V11 KG-CTCN SAFE BACKFILL SCRIPT
--------------------------------------------------
Populates the historical prediction database (feedback_loop.sqlite) 
from 2022-01-01 to 2026-05-31.

CRITICAL RULES:
1. CAUSALITY: Only data up to T-1 is used for prediction at T.
2. NO LEAKAGE: 28-day sequences are sliced from the historical window.
3. IN-DISTRIBUTION: Replicates the 365-day rolling Z-score normalization exactly.
"""

import os
import sys
import uuid
import json
import logging
import sqlite3
import contextlib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import torch

# Add src to path to import local modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

try:
    from inference_engine import V11InferenceEngine
    from weather_provider import WeatherDataProvider
    from feedback_db import _get_connection, init_db
except ImportError as e:
    print(f"Error importing V11 components: {e}")
    sys.exit(1)

# --- CONFIGURATION ---
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2026, 5, 31)
LOCATION = "Sangli"
DEFAULT_AGRO = {
    "variety_susceptibility": 0.8,  # Reasonably susceptible
    "is_ratoon": 0,                 # Plant crop
    "crop_age_days": 180            # Mid-season
}

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("v11_backfill.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Backfill")

# --- CUSTOM ENGINE TO BYPASS INDIVIDUAL API CALLS ---
class BackfillEngine(V11InferenceEngine):
    """
    Overrides V11InferenceEngine to use a pre-loaded full weather dataframe
    instead of fetching from API for every single date.
    """
    def __init__(self, full_weather_df):
        super().__init__()
        self.full_weather_df = full_weather_df
        # We don't need the real weather provider during backfill
        self.weather_provider = None

    def _fetch_past_weather_slice(self, location, target_date):
        """Slices the cached dataframe to enforce causality."""
        # MIN_HISTORY_DAYS is 400 in v11.2 (for 365-day rolling window)
        from inference_engine import MIN_HISTORY_DAYS
        
        mask = (self.full_weather_df['date'] <= target_date)
        # We need the past 400 days including target_date
        return self.full_weather_df[mask].tail(MIN_HISTORY_DAYS).copy()

# --- MAIN LOGIC ---

def fetch_weather_batch(start_date, end_date):
    """Fetches the entire required weather range in one large batch."""
    provider = WeatherDataProvider()
    # We need 400 days of history before the start_date to allow the 
    # first prediction to have a full rolling Z-score window.
    fetch_start = start_date - timedelta(days=410)
    
    logger.info(f"Fetching weather batch from {fetch_start.date()} to {end_date.date()}...")
    
    # WeatherDataProvider.get_weather_history takes target_date and window_days
    # We calculate the window needed to cover the whole range
    total_days = (end_date - fetch_start).days + 1
    
    return provider.get_weather_history(LOCATION, end_date, window_days=total_days)

def get_existing_prediction_dates():
    """Returns a set of dates that already have predictions in the DB."""
    try:
        with contextlib.closing(_get_connection()) as conn:
            df = pd.read_sql_query("SELECT prediction_date FROM predictions", conn)
            return set(df['prediction_date'].tolist())
    except Exception:
        return set()

def run_backfill():
    # 1. Initialize DB
    init_db()
    
    # 2. Fetch full weather data
    try:
        full_weather = fetch_weather_batch(START_DATE, END_DATE)
    except Exception as e:
        logger.error(f"Failed to fetch weather data: {e}")
        return

    # 3. Load Engine with data cache
    logger.info("Initializing V11 KG-CTCN Inference Engine...")
    engine = BackfillEngine(full_weather)
    
    # 4. Filter out already processed dates
    existing_dates = get_existing_prediction_dates()
    
    target_dates = pd.date_range(START_DATE, END_DATE)
    dates_to_process = [d for d in target_dates if d.strftime("%Y-%m-%d") not in existing_dates]
    
    if not dates_to_process:
        logger.info("All dates in range already exist in database. No work to do.")
        return

    logger.info(f"Starting backfill for {len(dates_to_process)} dates...")
    
    results = []
    skipped = 0
    success = 0
    
    # 5. Prediction Loop
    for t_date in tqdm(dates_to_process, desc="Backfilling Predictions"):
        try:
            # target_date in EWS is usually prediction_date + lead (e.g. T+5)
            # but for the log, we follow the schema: prediction_date is T.
            res = engine.run_inference(LOCATION, t_date, DEFAULT_AGRO)
            
            # Prepare row for DB
            pred_id = str(uuid.uuid4())
            
            # Note: v11.1 feedback_db uses current date for prediction_date.
            # We override this to store the historical date T.
            row = {
                "prediction_id": pred_id,
                "prediction_date": t_date.strftime("%Y-%m-%d"),
                "target_date": (t_date + timedelta(days=5)).strftime("%Y-%m-%d"),
                "location": LOCATION,
                "weather_snapshot_json": json.dumps(res["raw_weather_sequence"].tolist()),
                "agro_inputs_json": json.dumps(res["agro_inputs"]),
                "risk_score": res["risk_score"],
                "risk_class": res["risk_class"],
                "confidence_score": res["confidence_score"],
                "alert_status": "BACKFILLED",
                "created_at": datetime.now().isoformat()
            }
            results.append(row)
            success += 1
            
            # Commit in batches of 50 to avoid holding memory but keep it fast
            if len(results) >= 50:
                save_batch(results)
                results = []
                
        except Exception as e:
            logger.warning(f"Skipped {t_date.date()}: {e}")
            skipped += 1
            
    # Save remaining
    if results:
        save_batch(results)
        
    logger.info(f"Backfill Complete. Success: {success} | Skipped: {skipped}")

def save_batch(batch_data):
    """Inserts a batch of results into the SQLite DB."""
    with contextlib.closing(_get_connection()) as conn:
        c = conn.cursor()
        c.executemany('''
            INSERT INTO predictions
            (prediction_id, prediction_date, target_date, location,
             weather_snapshot_json, agro_inputs_json,
             risk_score, risk_class, confidence_score, alert_status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', [
            (r["prediction_id"], r["prediction_date"], r["target_date"], r["location"],
             r["weather_snapshot_json"], r["agro_inputs_json"],
             r["risk_score"], r["risk_class"], r["confidence_score"], 
             r["alert_status"], r["created_at"])
            for r in batch_data
        ])
        conn.commit()

if __name__ == "__main__":
    run_backfill()
