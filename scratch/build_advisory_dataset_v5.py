"""
build_advisory_dataset_v5.py  (UPDATED)
----------------------------------------
Advisory Dataset Layer for Crop Disease Outbreak Prediction System v5.
Red Rot of Sugarcane — decision-support layer.

Input : v5/data/processed/dataset_v5.csv
Output: v5/data/processed/advisory_dataset_v5.csv

Run:
  python scratch/build_advisory_dataset_v5.py
"""

import os
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV  = os.path.join(BASE, "v5", "data", "processed", "dataset_v5.csv")
OUT_CSV = os.path.join(BASE, "v5", "data", "processed", "advisory_dataset_v5.csv")

print("=" * 60)
print("  v5 Advisory Dataset Builder (UPDATED)")
print("  Red Rot of Sugarcane")
print("=" * 60)

# ── STEP 1: Load & validate ───────────────────────────────────────────────────
print("\n[1/9] Loading dataset_v5.csv...")
df = pd.read_csv(IN_CSV, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

REQUIRED = ["env_interaction_score", "monsoon_flag", "RH2M_roll7",
            "humidity_streak_days", "rainfall_streak_days"]
missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

print(f"       Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"       Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"       Null check: {df[REQUIRED].isnull().sum().sum()} nulls in critical columns")

# ── STEP 2: Agronomic placeholder inputs ─────────────────────────────────────
print("\n[2/9] Adding agronomic features (simulated defaults)...")
df["variety_susceptibility"] = 0.7
df["crop_age_months"]        = 6
df["infected_sett_flag"]     = 0

# ── STEP 3: Agronomic modifier factors ───────────────────────────────────────
print("\n[3/9] Computing agronomic modifier factors...")

# Susceptibility factor
df["susceptibility_factor"] = df["variety_susceptibility"].apply(
    lambda vs: 1.2 if vs >= 0.7 else (1.0 if vs >= 0.4 else 0.8)
)

# Crop stage factor
df["crop_stage_factor"] = df["crop_age_months"].apply(
    lambda age: 1.2 if 4 <= age <= 8 else 0.9
)

# Infection source factor
df["infection_factor"] = df["infected_sett_flag"].apply(
    lambda x: 1.3 if x == 1 else 1.0
)

# ── STEP 4: Improved base risk score ─────────────────────────────────────────
print("\n[4/9] Computing improved base_risk_score...")
# Smoother formulation: blends discrete env_interaction_score with
# continuous RH2M_roll7 signal — both use only past data
df["base_risk_score"] = (
    df["env_interaction_score"] + (df["RH2M_roll7"] / 100.0)
) / 4.0
print(f"       base_risk_score range: {df['base_risk_score'].min():.4f} to {df['base_risk_score'].max():.4f}")

# ── STEP 5: Final risk score with monsoon gating ─────────────────────────────
print("\n[5/9] Computing final_risk_score with monsoon gating...")
df["final_risk_score"] = (
    df["base_risk_score"]
    * df["susceptibility_factor"]
    * df["crop_stage_factor"]
    * df["infection_factor"]
)

# Monsoon gating — suppress risk outside monsoon season
df["final_risk_score"] = np.where(
    df["monsoon_flag"] == 0,
    df["final_risk_score"] * 0.3,
    df["final_risk_score"]
)

# Clip to [0, 1]
df["final_risk_score"] = df["final_risk_score"].clip(upper=1.0)
print(f"       final_risk_score range: {df['final_risk_score'].min():.4f} to {df['final_risk_score'].max():.4f}")

# ── STEP 6: Risk level classification ────────────────────────────────────────
print("\n[6/9] Classifying risk_level...")
def classify_risk(score):
    if score >= 0.75:
        return "HIGH"
    elif score >= 0.50:
        return "MEDIUM"
    else:
        return "LOW"

df["risk_level"] = df["final_risk_score"].apply(classify_risk)

# ── STEP 7: Early warning flag ────────────────────────────────────────────────
print("\n[7/9] Computing early_warning flag...")
df["early_warning"] = (
    (df["env_interaction_score"] == 3) & (df["monsoon_flag"] == 1)
).astype(int)

# ── STEP 8: Context-aware advisory message ────────────────────────────────────
print("\n[8/9] Generating context-aware advisory messages...")

def advisory_message(row):
    rl  = row["risk_level"]
    hsd = row["humidity_streak_days"]
    rsd = row["rainfall_streak_days"]
    if rl == "HIGH" and hsd >= 7:
        return ("High risk due to prolonged high humidity. "
                "Immediate monitoring and preventive fungicide application recommended.")
    elif rl == "HIGH" and rsd >= 5:
        return ("High risk due to continuous rainfall enabling pathogen spread. "
                "Ensure drainage and initiate protection measures.")
    elif rl == "HIGH":
        return ("High outbreak risk under favorable environmental conditions. "
                "Immediate field inspection required.")
    elif rl == "MEDIUM":
        return "Moderate risk. Monitor weather conditions and prepare preventive measures."
    else:
        return "Low risk. Maintain standard crop management practices."

df["advisory_message"] = df.apply(advisory_message, axis=1)

# ── STEP 9: Save ──────────────────────────────────────────────────────────────
print("\n[9/9] Saving advisory_dataset_v5.csv...")
OUT_COLS = [
    "date",
    "env_interaction_score",
    "base_risk_score",
    "monsoon_flag",
    "humidity_streak_days",
    "rainfall_streak_days",
    "variety_susceptibility",
    "crop_age_months",
    "infected_sett_flag",
    "susceptibility_factor",
    "crop_stage_factor",
    "infection_factor",
    "final_risk_score",
    "risk_level",
    "early_warning",
    "advisory_message",
]
advisory = df[OUT_COLS].copy()
advisory.to_csv(OUT_CSV, index=False)
print(f"       Saved -> {OUT_CSV}")

# ── Validation & Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Validation & Summary")
print("=" * 60)

total = len(advisory)
print(f"\n  Shape         : {advisory.shape[0]} rows x {advisory.shape[1]} columns")
print(f"  Date range    : {advisory['date'].min().date()} to {advisory['date'].max().date()}")
print(f"  Null check    : {advisory.isnull().sum().sum()} nulls")

print(f"\n  Risk level distribution:")
for lvl in ["HIGH", "MEDIUM", "LOW"]:
    cnt = (advisory["risk_level"] == lvl).sum()
    bar = "#" * int(40 * cnt / total)
    print(f"    {lvl:<8} : {cnt:>5,}  ({100*cnt/total:5.1f} %)  {bar}")

ew = advisory["early_warning"].sum()
print(f"\n  Early warnings: {ew:,} days ({100*ew/total:.1f} %)")

# Advisory message breakdown
msg_counts = advisory["advisory_message"].value_counts()
print(f"\n  Advisory message variants: {len(msg_counts)}")
for msg, cnt in msg_counts.items():
    print(f"    [{cnt:>4}]  {msg[:80]}")

print(f"\n  First 5 rows:")
cols_preview = ["date","final_risk_score","risk_level","early_warning","humidity_streak_days","advisory_message"]
pd.set_option("display.max_colwidth", 60)
print(advisory[cols_preview].head(5).to_string(index=False))

print(f"\n  Last 5 rows:")
print(advisory[cols_preview].tail(5).to_string(index=False))

print("\n[DONE] Advisory dataset (UPDATED) ready.")
print(f"       Location: {OUT_CSV}")
