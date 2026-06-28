"""
build_dataset_v5.py
--------------------
Builds the fully processed v5 time-series dataset for the
Crop Disease Outbreak Prediction System.

Input : v5/data/raw/POWER_Point_Daily_20050101_20241231_016d54N_069d78E_LST.csv
Output: v5/data/processed/dataset_v5.csv

Run:
  python scratch/build_dataset_v5.py
"""

import os
import sys
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(BASE, "v5", "data", "raw",
                       "POWER_Point_Daily_20050101_20241231_016d54N_069d78E_LST.csv")
OUT_DIR = os.path.join(BASE, "v5", "data", "processed")
OUT_CSV = os.path.join(OUT_DIR, "dataset_v5.csv")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("  v5 Dataset Builder — Crop Disease Outbreak Prediction")
print("=" * 60)

# ── STEP 1: Load raw CSV ──────────────────────────────────────────────────────
print("\n[1/9] Loading raw CSV...")

# NASA POWER CSV has a multi-line header ending with -END HEADER-
# Find the row where actual data starts
with open(RAW_CSV, encoding="utf-8") as f:
    lines = f.readlines()

skip = 0
for i, line in enumerate(lines):
    if line.strip().startswith("YEAR"):
        skip = i
        break

df = pd.read_csv(RAW_CSV, skiprows=skip)
print(f"       Raw rows loaded: {len(df)}")
print(f"       Columns: {list(df.columns)}")

# Build proper date from YEAR + DOY
df["date"] = pd.to_datetime(df["YEAR"].astype(str) + df["DOY"].astype(str).str.zfill(3),
                             format="%Y%j")
df = df.sort_values("date").reset_index(drop=True)

# Keep only needed columns
NEEDED = ["date", "T2M", "T2M_MIN", "T2M_MAX", "RH2M", "PRECTOTCORR"]
df = df[NEEDED].copy()

# Replace NASA missing value sentinel (-999) with NaN
df.replace(-999, np.nan, inplace=True)

# Remove duplicates and enforce strict daily calendar grid
df = df.drop_duplicates(subset="date")
df = df.set_index("date").asfreq("D").reset_index()

print(f"       After temporal canonicalization: {len(df)} rows ({df['date'].min().date()} to {df['date'].max().date()})")

# ── STEP 2: Handle missing values ─────────────────────────────────────────────
print("\n[2/9] Handling missing values...")
missing_before = df.isnull().sum().sum()
df[["T2M","T2M_MIN","T2M_MAX","RH2M","PRECTOTCORR"]] = (
    df[["T2M","T2M_MIN","T2M_MAX","RH2M","PRECTOTCORR"]]
    .ffill(limit=3)
)
# DO NOT dropna() here! We must preserve the calendar grid for rolling/lag operations.
print(f"       Missing cells handled (ffill). Before: {missing_before}, Remaining NaNs: {df.isnull().sum().sum()}")


# ── STEP 3: Rolling means (use only past data via min_periods=1) ──────────────
print("\n[3/9] Creating rolling and lag features...")

def roll_mean(series, w):
    return series.rolling(w, min_periods=1).mean()

def roll_sum(series, w):
    return series.rolling(w, min_periods=1).sum()

# T2M rolling means
df["T2M_roll3"]  = roll_mean(df["T2M"], 3)
df["T2M_roll5"]  = roll_mean(df["T2M"], 5)
df["T2M_roll7"]  = roll_mean(df["T2M"], 7)
df["T2M_roll14"] = roll_mean(df["T2M"], 14)
df["T2M_roll28"] = roll_mean(df["T2M"], 28)

# T2M_MIN rolling means
df["T2M_MIN_roll3"]  = roll_mean(df["T2M_MIN"], 3)
df["T2M_MIN_roll7"]  = roll_mean(df["T2M_MIN"], 7)
df["T2M_MIN_roll14"] = roll_mean(df["T2M_MIN"], 14)

# RH2M rolling means
df["RH2M_roll3"]  = roll_mean(df["RH2M"], 3)
df["RH2M_roll5"]  = roll_mean(df["RH2M"], 5)
df["RH2M_roll7"]  = roll_mean(df["RH2M"], 7)
df["RH2M_roll14"] = roll_mean(df["RH2M"], 14)
df["RH2M_roll28"] = roll_mean(df["RH2M"], 28)

# PREC rolling sums
df["PREC_sum3"]  = roll_sum(df["PRECTOTCORR"], 3)
df["PREC_sum5"]  = roll_sum(df["PRECTOTCORR"], 5)
df["PREC_sum7"]  = roll_sum(df["PRECTOTCORR"], 7)
df["PREC_sum14"] = roll_sum(df["PRECTOTCORR"], 14)
df["PREC_sum28"] = roll_sum(df["PRECTOTCORR"], 28)

# Lag features — .shift() = past data, fully leakage-free
df["T2M_MIN_lag15"] = df["T2M_MIN"].shift(15)   # PRIMARY (SMRA validated)
df["T2M_lag15"]     = df["T2M"].shift(15)
df["RH2M_lag15"]    = df["RH2M"].shift(15)
df["PREC_lag15"]    = df["PRECTOTCORR"].shift(15)
df["T2M_MIN_lag7"]  = df["T2M_MIN"].shift(7)
df["RH2M_lag7"]     = df["RH2M"].shift(7)
df["T2M_MIN_lag28"] = df["T2M_MIN"].shift(28)

print("       Rolling and lag features created.")

# ── STEP 4: Domain features ───────────────────────────────────────────────────
print("\n[4/9] Creating domain-guided biological features...")

# monsoon_flag — based on month
df["month"]      = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear
df["monsoon_flag"] = df["month"].isin([6, 7, 8, 9, 10]).astype(int)

# temperature_suitability_flag — optimum pathogen range 29-33 C
df["temperature_suitability_flag"] = ((df["T2M"] >= 29.0) & (df["T2M"] <= 33.0)).astype(int)

# temp_suitable_days_14d — rolling count of suitable days
df["temp_suitable_days_14d"] = (
    df["temperature_suitability_flag"].rolling(14, min_periods=1).sum()
)

# humidity_streak_days — consecutive days where RH2M >= 82
def streak(series, threshold, above=True):
    """Count consecutive days meeting threshold."""
    mask = (series >= threshold) if above else (series <= threshold)
    result = []
    count = 0
    for val in mask:
        if val:
            count += 1
        else:
            count = 0
        result.append(count)
    return result

df["humidity_streak_days"]  = streak(df["RH2M"], 82)
df["rainfall_streak_days"]  = streak(df["PRECTOTCORR"], 5)

print("       Domain features created.")

# ── STEP 4b: New engineered features (FIX 1 + FIX 2) ─────────────────────────
print("\n[4b/9] Adding improved signal features...")

# FIX 1: Rainfall accumulation signal (explicit domain feature)
df["rainfall_accumulation_14d"] = df["PREC_sum14"]   # alias — PREC_sum14 already computed

# FIX 2: Environmental interaction score (0-3, uses ONLY past rolling data)
# Each component = 1 if past-7-day rolling value meets threshold, else 0
rh_ok   = (df["RH2M_roll7"]  >= 82).astype(int)
temp_ok = ((df["T2M_roll7"] >= 29) & (df["T2M_roll7"] <= 33)).astype(int)
rain_ok = (df["PREC_sum7"]   >= 15).astype(int)
df["env_interaction_score"] = rh_ok + temp_ok + rain_ok   # range 0-3

print("       rainfall_accumulation_14d and env_interaction_score created.")


# ── STEP 5: Create label (LEAKAGE-FREE — uses only future data) ───────────────
print("\n[5/9] Computing outbreak_risk label (future window t+3 to t+7)...")

# Shift backward to bring future values to current row (leakage-free: only in label)
T2M_f = pd.DataFrame({
    "t3": df["T2M"].shift(-3),
    "t4": df["T2M"].shift(-4),
    "t5": df["T2M"].shift(-5),
    "t6": df["T2M"].shift(-6),
    "t7": df["T2M"].shift(-7),
})
RH2M_f = pd.DataFrame({
    "t3": df["RH2M"].shift(-3),
    "t4": df["RH2M"].shift(-4),
    "t5": df["RH2M"].shift(-5),
    "t6": df["RH2M"].shift(-6),
    "t7": df["RH2M"].shift(-7),
})
PREC_f = pd.DataFrame({
    "t3": df["PRECTOTCORR"].shift(-3),
    "t4": df["PRECTOTCORR"].shift(-4),
    "t5": df["PRECTOTCORR"].shift(-5),
    "t6": df["PRECTOTCORR"].shift(-6),
    "t7": df["PRECTOTCORR"].shift(-7),
})

temp_ok     = (T2M_f.mean(axis=1) >= 29.0) & (T2M_f.mean(axis=1) <= 33.0)
humidity_ok = RH2M_f.mean(axis=1) >= 82.0
rain_ok     = PREC_f.sum(axis=1) >= 20.0

# Label = 1 if at least 2 of 3 conditions met (softer than strict AND)
risk_score = temp_ok.astype(int) + humidity_ok.astype(int) + rain_ok.astype(int)
df["outbreak_risk"] = (risk_score >= 2).astype(int)

print("       Label computed.")

# ── STEP 6: Drop warmup and tail rows ─────────────────────────────────────────
print("\n[6/9] Dropping warmup (first 28) and tail (last 7) rows...")
n_before = len(df)
df = df.iloc[28:-7].reset_index(drop=True)
print(f"       Rows before: {n_before}, after clean: {len(df)}")

# Also drop any remaining NaN rows (from lag features)
df = df.dropna().reset_index(drop=True)
print(f"       Rows after dropna: {len(df)}")

# ── STEP 7: Final column order ────────────────────────────────────────────────
print("\n[7/9] Ordering final columns...")
FINAL_COLS = [
    "date",
    # Raw weather
    "T2M", "T2M_MIN", "T2M_MAX", "RH2M", "PRECTOTCORR",
    # Rolling — T2M
    "T2M_roll3", "T2M_roll5", "T2M_roll7", "T2M_roll14", "T2M_roll28",
    # Rolling — T2M_MIN
    "T2M_MIN_roll3", "T2M_MIN_roll7", "T2M_MIN_roll14",
    # Rolling — RH2M
    "RH2M_roll3", "RH2M_roll5", "RH2M_roll7", "RH2M_roll14", "RH2M_roll28",
    # Rolling sums — PREC
    "PREC_sum3", "PREC_sum5", "PREC_sum7", "PREC_sum14", "PREC_sum28",
    # Lag features
    "T2M_MIN_lag15", "T2M_lag15", "RH2M_lag15", "PREC_lag15",
    "T2M_MIN_lag7", "RH2M_lag7", "T2M_MIN_lag28",
    # Domain features
    "month", "day_of_year", "monsoon_flag",
    "temperature_suitability_flag", "temp_suitable_days_14d",
    "humidity_streak_days", "rainfall_streak_days",
    # Improved signal features
    "rainfall_accumulation_14d",
    "env_interaction_score",
    # Label
    "outbreak_risk",
]
df = df[FINAL_COLS]

# ── STEP 8: Save ──────────────────────────────────────────────────────────────
print("\n[8/9] Saving dataset...")
df.to_csv(OUT_CSV, index=False)
print(f"       Saved -> {OUT_CSV}")

# ── STEP 9: Debug summary ─────────────────────────────────────────────────────
print("\n[9/9] Dataset Summary")
print("-" * 60)
print(f"  Shape          : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  Date range     : {df['date'].min().date()} to {df['date'].max().date()}")
print(f"  Feature count  : {df.shape[1] - 2}  (excl. date + label)")

label_counts = df["outbreak_risk"].value_counts().sort_index()
n_pos = label_counts.get(1, 0)
n_neg = label_counts.get(0, 0)
total = n_pos + n_neg
print(f"\n  Label distribution:")
print(f"    outbreak_risk=1 (positive): {n_pos:,}  ({100*n_pos/total:.1f} %)")
print(f"    outbreak_risk=0 (negative): {n_neg:,}  ({100*n_neg/total:.1f} %)")
print(f"    Imbalance ratio            : 1:{n_neg//max(n_pos,1)}")

print(f"\n  First 5 rows (selected cols):")
print(df[["date","T2M","RH2M","PRECTOTCORR","monsoon_flag","outbreak_risk"]].head(5).to_string(index=False))

print(f"\n  Last 5 rows (selected cols):")
print(df[["date","T2M","RH2M","PRECTOTCORR","monsoon_flag","outbreak_risk"]].tail(5).to_string(index=False))

print(f"  Null check: {df.isnull().sum().sum()} nulls remaining")

# Validation checks
const_cols = [c for c in df.columns if c != "date" and df[c].nunique() <= 1]
print(f"\n  Constant columns check: {len(const_cols)} constant cols {const_cols if const_cols else '— none (good)'}")

eis_min = int(df["env_interaction_score"].min())
eis_max = int(df["env_interaction_score"].max())
print(f"  env_interaction_score range: {eis_min} to {eis_max}  (expected 0-3)")

ra14_min = df["rainfall_accumulation_14d"].min()
ra14_max = df["rainfall_accumulation_14d"].max()
print(f"  rainfall_accumulation_14d range: {ra14_min:.2f} to {ra14_max:.2f} mm")

print("\n[DONE] dataset_v5.csv updated and ready for model training.")
print(f"       Location: {OUT_CSV}")
