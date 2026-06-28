"""
V6 Utils -- Standalone data loader (no relative imports).
Replaces V5's utils.py which used relative package imports.
"""

import os
import logging
import pandas as pd

# ── Resolved path to the NASA POWER raw data (shared with V5) ────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_V5_RAW_DATA = os.path.normpath(os.path.join(
    _HERE, "..", "..", "v5", "data", "raw",
    "POWER_Point_Daily_20050101_20241231_016d54N_069d78E_LST.csv"
))


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(ch)
    return logger


logger = setup_logger("v6.utils")


def load_raw_data(path: str = _V5_RAW_DATA) -> pd.DataFrame:
    """
    Load and parse the NASA POWER raw daily weather CSV.
    Skips the 14-line NASA header and parses YEAR+DOY into a proper date column.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw data not found at:\n  {path}\n"
            "Ensure the V5 data directory is present."
        )

    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path, skiprows=14)

    # Parse dates from NASA POWER YEAR + DOY columns
    df["date"] = pd.to_datetime(
        df["YEAR"].astype(str) + df["DOY"].astype(str).str.zfill(3),
        format="%Y%j"
    )
    df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} daily records  ({df['date'].min().date()} -> {df['date'].max().date()})")
    return df


def save_threshold(value: float, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(str(value))
    logger.info(f"Threshold {value:.6f} saved -> {path}")


def load_threshold(path: str) -> float:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Threshold file not found: {path}. Run training first.")
    with open(path) as f:
        return float(f.read().strip())
