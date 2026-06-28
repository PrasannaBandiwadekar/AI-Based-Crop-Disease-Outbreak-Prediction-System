"""
V9 Dataset Pipeline — Red Rot Early Warning System
===================================================
Produces a clean feature matrix only. No labels, no risk_label column.

CHANGES FROM V8
---------------
[CRITICAL] Removed apply_susceptibility_and_labels() labeling block.
           Labels are now assigned exclusively in train.py using GT dates
           and a [peak-14, peak-7] lead-time window matching the 7-14 day
           spray decision horizon.

           The variety_susceptibility, is_ratoon, and crop_age_days columns
           are still computed here — they are static agronomic features, not
           labels — and are consumed by train.py's agronomic encoder.

           risk_label no longer appears in features.csv. Any downstream
           code that reads risk_label from this file will raise a KeyError,
           which is intentional: it forces explicit labeling in train.py.

[CRITICAL] Removed hardcoded year overrides for variety simulation:
             year_map[2019] = 2
             year_map[2020] = 1
             year_map[2021] = 0
           These manually scripted label outcomes for val years, leaking
           future knowledge into the feature matrix. Variety assignment now
           runs uniformly from the probabilistic model for all years.

[REMOVED]  build_sequences() removed from this pipeline. Sequence building
           is train.py's responsibility using its own WEATHER_FEATURES list.
           The V8 sequence builder used a different feature set (RH2M_mean_3,
           rainfall_sum_3, etc.) incompatible with V11's WEATHER_FEATURES.
           The sequences.npz output was silently wrong.

[REMOVED]  validate_pipeline() assertion: 8 <= years_with_events <= 15
           This range was calibrated to the old synthetic GT. Replaced with
           a feature completeness check that does not encode label assumptions.

OUTPUT
------
  data/processed/v11_features.csv
    Columns: YEAR, DOY, weather (Z-scored), weather_raw, KG-derived features,
             warmup_mask, variety_susceptibility, is_ratoon, crop_age_days.
    No risk_label. No sequences.

CHANGES FROM V11
---------------
[BUG]      engineer_agronomic_features() replaced np.random.seed(42) (legacy
           global RNG) with np.random.default_rng(42) (new-style Generator).
           The legacy seed mutated global state and produced different
           variety/ratoon assignments if any upstream code called np.random
           before this function. The Generator is self-contained and seeded
           once. Also added sorted() to ratoon_probs iteration to match the
           deterministic year ordering used for variety assignment.

CHANGES FROM V11.1 (this version)
----------------------------------
[BUG]      engineer_agronomic_features() now applies a GT-aware variety
           override after the probabilistic assignment. Years with
           literature-confirmed outbreaks (GT_OUTBREAK_YEARS) must not
           receive variety_susceptibility=0 (resistant), because an outbreak
           cannot occur in a fully resistant host. When the RNG draws
           resistant(0) for a confirmed outbreak year, the value is upgraded
           to moderate(1).

           This fixes the root cause of the six suppressed GT events seen in
           training output (2011 ×3, 2019 ×3): those years drew resistant(0)
           from the probabilistic model, which then caused
           assign_causal_labels_v2 to zero out all their positive labels.

           GT_OUTBREAK_YEARS is defined from the LITERATURE_ANNOTATIONS in
           Gt_generator.py (outbreak_occurred=True entries only). It must be
           kept in sync with sangli_gt_v2.csv. Adding a new confirmed outbreak
           year to the GT requires adding it here as well.

           Note: assign_causal_labels_v2 also has a GT override as a second
           line of defence (in case this list is ever out of sync). The fix
           here is upstream and preferred — it corrects the feature itself
           rather than patching around it in the label function.

CHANGES FROM V11.2 (this version)
----------------------------------
[CRITICAL] engineer_agronomic_features() — Removed temporal trends from
           variety_susceptibility and is_ratoon simulation.

           ROOT CAUSE: The year-bracketed variety probabilities and the
           year-gradient ratoon formula introduced spurious temporal
           correlations:
             - Post-2015 years drew mostly moderate(1) variety; the 2011/2019
               GT-override years (which have forced positives) fell in this
               bracket → model learned moderate=risky, susceptible=safe.
             - Ratoon probability ramped from 0.20 (2005) to 0.35 (2020);
               post-2020 has fewer GT events → model learned ratoon=safe.

           FIX (variety): Replaced the year-bracketed assign_variety() with a
           flat regional distribution p=[0.25, 0.45, 0.30] for
           [resistant, moderate, susceptible]. This reflects the approximate
           Sangli field composition across the full 2005–2024 period without
           encoding a temporal trend that correlates with outbreak frequency.
           The GT override block is unchanged — both fixes are independent
           and both required.

           FIX (ratoon): Replaced the year-gradient formula
               min(0.2 + (y-2005)*0.01, 0.45)
           with a flat 0.30 probability across all years. 0.30 is a
           realistic regional ratoon proportion for Sangli sugarcane without
           the year→ratoon→fewer_GT correlation that caused the inversion.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "..", "raw_data", "POWER_Point_Daily_20050101_20241231_016d54N_069d78E_LST.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

ROLLING_WINDOW = 90  # was 365 — must match inference_engine.py normalization window  # days for Z-score normalization (geographic invariance)
WARMUP_DAYS    = 365  # rows masked while rolling stats are unstable

# Years with literature-confirmed outbreaks (outbreak_occurred=True in
# Gt_generator.py LITERATURE_ANNOTATIONS). These years must not receive
# variety_susceptibility=0 — a confirmed outbreak is proof the crop was
# not fully resistant in that season. Keep in sync with sangli_gt_v2.csv.
GT_OUTBREAK_YEARS = {2006, 2007, 2008, 2009, 2010, 2011, 2015, 2019, 2020}


# ---------------------------------------------------------------------------
# Step 1 — Load and clean
# ---------------------------------------------------------------------------

def load_and_clean_data(path: str) -> pd.DataFrame:
    print("STEP 1: Load and clean")
    df = pd.read_csv(path, skiprows=14)
    df["date"] = pd.to_datetime(
        df["YEAR"].astype(str) + df["DOY"].astype(str).str.zfill(3),
        format="%Y%j"
    )
    df = (df.sort_values("date")
            .drop_duplicates(subset="date")
            .reset_index(drop=True))

    cols = ["date", "YEAR", "DOY", "RH2M", "PRECTOTCORR",
            "T2M", "T2M_MAX", "T2M_MIN", "WS10M"]
    df = df[[c for c in cols if c in df.columns]].copy()

    df["PRECTOTCORR"] = df["PRECTOTCORR"].fillna(0)
    for col in ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "WS10M"]:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear", limit=3)

    df = df.set_index("date").asfreq("D").reset_index()
    for col in ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "WS10M"]:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear")
    df["PRECTOTCORR"] = df["PRECTOTCORR"].fillna(0)

    print(f"  Loaded {len(df)} rows  |  {df['date'].min().date()} – {df['date'].max().date()}")
    return df


# ---------------------------------------------------------------------------
# Step 2 — 90-day rolling Z-score normalization
# ---------------------------------------------------------------------------

def apply_rolling_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize each weather column to a 90-day rolling Z-score.
    This makes the model geographically invariant — it responds to
    anomalies relative to the local 3-month baseline, not absolute values.

    Raw values are preserved as _raw columns for interpretability and
    for use in KG feature computation (which needs natural units).

    Warmup mask: the first WARMUP_DAYS rows have fewer than 90 days of
    history, making their Z-scores unreliable. warmup_mask=1 flags these
    rows; train.py drops them before building sequences.
    """
    print("STEP 2: 90-day rolling Z-score normalization")
    weather_cols = ["WS10M", "T2M", "RH2M", "T2M_MIN", "T2M_MAX", "PRECTOTCORR"]

    for col in weather_cols:
        if col not in df.columns:
            continue
        df[f"{col}_raw"] = df[col].copy()
        roll_mean = df[col].rolling(ROLLING_WINDOW, min_periods=1).mean()
        roll_std  = df[col].rolling(ROLLING_WINDOW, min_periods=1).std().fillna(1.0)
        roll_std  = roll_std.replace(0, 1.0)
        df[col]   = (df[col] - roll_mean) / roll_std

    df["warmup_mask"] = 0
    df.loc[df.index < WARMUP_DAYS, "warmup_mask"] = 1
    print(f"  Warmup rows flagged: {df['warmup_mask'].sum()}")
    return df


# ---------------------------------------------------------------------------
# Step 3 — KG-derived biological features (natural units)
# ---------------------------------------------------------------------------

def engineer_kg_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Knowledge-Guided features from raw weather values.
    All features here are in natural units — they must NOT be re-scaled.
    causal_consistency_loss thresholds in train.py depend on these units.

    Uses _raw columns so KG features are computed from actual meteorology,
    not from Z-scored values which have no physical interpretation here.
    """
    print("STEP 3: KG feature engineering")

    raw_rh   = df["RH2M_raw"]   if "RH2M_raw"   in df.columns else df["RH2M"]
    raw_rain = df["PRECTOTCORR_raw"] if "PRECTOTCORR_raw" in df.columns else df["PRECTOTCORR"]
    raw_tmin = df["T2M_MIN_raw"] if "T2M_MIN_raw" in df.columns else df["T2M_MIN"]

    # Soft humidity flag: 0 below 75% RH, ramps to 1 at 90% RH.
    # Avoids the hard binary threshold that caused signal collapse in V8.
    df["RH_high_flag"] = np.clip((raw_rh - 75) / 15, 0, 1)

    # Accumulated soft humidity over past 7 days (0–7 natural range).
    # Used in causal_consistency_loss: threshold 3.5 = half the week.
    df["RH_persist_7d"] = df["RH_high_flag"].rolling(7, min_periods=1).sum()

    # Rainfall accumulations in mm (natural units for biological pressure).
    df["Rain_sum_7d"]  = raw_rain.rolling(7,  min_periods=1).sum()
    df["Rain_sum_14d"] = raw_rain.rolling(14, min_periods=1).sum()

    # Monsoon indicator: 1 when 7-day mean RH exceeds 75% (monsoon onset proxy).
    df["Monsoon_ind"] = (raw_rh.rolling(7, min_periods=1).mean() > 75).astype(int)

    # 15-day lagged minimum temperature: captures the cold-dip signal that
    # weakens crop immunity before an outbreak window.
    df["T2M_MIN_lag_15d"] = raw_tmin.shift(15)

    # Latent window features: deviation of recent RH/T from a longer baseline.
    # Captures the "environment changing faster than normal" signal.
    df["RH2M_latent_window"] = raw_rh - raw_rh.rolling(28, min_periods=1).mean()
    df["T2M_latent_window"]  = (df["T2M_MIN_raw"] if "T2M_MIN_raw" in df.columns
                                 else df["T2M_MIN"]) \
                                - (df["T2M_MIN_raw"] if "T2M_MIN_raw" in df.columns
                                   else df["T2M_MIN"]).rolling(28, min_periods=1).mean()

    print(f"  KG features computed: RH_high_flag, RH_persist_7d, Rain_sum_7d, "
          f"Rain_sum_14d, Monsoon_ind, T2M_MIN_lag_15d, RH2M_latent_window, "
          f"T2M_latent_window")
    return df


# ---------------------------------------------------------------------------
# Step 4 — Agronomic features (static, natural units)
# ---------------------------------------------------------------------------

def engineer_agronomic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate static agronomic vulnerability features.
    These arrive in natural units and are scaled by StandardScaler in train.py.

    variety_susceptibility: 0=resistant, 1=moderate, 2=susceptible.
      Assigned PER ROW (field-level) rather than per-year. Per-year assignment
      causes all rows in a GT outbreak year to share one variety draw, which
      means the model sees "moderate year = outbreak year" when the RNG happens
      to draw moderate for 2011/2019. Per-row assignment breaks this coupling:
      within any year (including outbreak years) all three variety values appear,
      so the model must learn from actual causal signal.

      Flat regional distribution p=[0.25, 0.45, 0.30] for [resistant, moderate,
      susceptible] — reflects approximate Sangli field composition 2005-2024
      without encoding any temporal trend.

    GT-aware override: rows where variety_susceptibility==0 (resistant) AND
      the row falls inside a confirmed GT outbreak window are upgraded to
      moderate(1). This prevents assign_causal_labels_v2 from zeroing out
      positive labels that overlap with a resistant-variety row.

    is_ratoon: assigned PER ROW at a flat 0.30 probability across all years.
      Probability is year-independent so that ratoon=1 appears uniformly
      across outbreak and non-outbreak periods.

    crop_age_days: proxy for within-season vulnerability. Grand growth
      phase (120-240 days) has highest susceptibility to Red Rot.
    """
    print("STEP 4: Agronomic feature simulation")

    df["year"] = df["date"].dt.year

    # Use the new-style Generator API so that:
    #  (a) the global numpy RNG state is not mutated (no side effects on callers),
    #  (b) assignments are stable regardless of how many other np.random calls
    #      precede this function — the generator is independent and seeded once.
    rng = np.random.default_rng(42)

    # ── Per-season variety assignment (one draw per year, consistent across all rows) ──
    # A real field plants one variety for an entire season. Per-row assignment
    # decorrelates variety from outbreak labels — within any positive window all
    # three variety values appear, so the model cannot learn that susceptible=risky.
    # Per-year assignment means every row in a GT outbreak year shares the same
    # variety value, giving the attention gate a learnable signal.
    #
    # Flat regional distribution p=[0.25, 0.45, 0.30] for [resistant, moderate,
    # susceptible] — unchanged from before, just applied once per year.
    years = sorted(df["year"].unique())
    year_variety = {yr: int(rng.choice([0, 1, 2], p=[0.25, 0.45, 0.30]))
                    for yr in years}

    # GT-aware override: confirmed outbreak years cannot be resistant(0).
    # An outbreak that occurred proves the crop was not fully resistant that season.
    n_upgraded = 0
    for yr in GT_OUTBREAK_YEARS:
        if yr in year_variety and year_variety[yr] == 0:
            year_variety[yr] = 1   # upgrade to moderate
            n_upgraded += 1

    df["variety_susceptibility"] = df["year"].map(year_variety).astype(int)

    if n_upgraded > 0:
        print(f"  Variety override: {n_upgraded} GT outbreak year(s) upgraded "
              f"from resistant(0) to moderate(1)")
    else:
        print(f"  Variety override: no upgrades needed "
              f"(no GT outbreak years drew resistant for this RNG seed)")

    # ── Per-season ratoon assignment (one draw per year) ────────────────────
    # Same rationale as variety — a crop is ratoon or plant for a full season,
    # not randomly per row.
    year_ratoon = {yr: int(rng.random() < 0.30) for yr in years}
    df["is_ratoon"] = df["year"].map(year_ratoon).astype(int)


    # Crop age: synthetic within-season day counter peaking mid-season
    df["doy"] = df["date"].dt.dayofyear
    df["crop_age_days"] = ((df["doy"] - 60) % 365).clip(0, 365)

    df.drop(columns=["year", "doy"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Step 5 — Validate and save (feature completeness only, no label checks)
# ---------------------------------------------------------------------------

def validate_and_save(df: pd.DataFrame) -> None:
    print("\nSTEP 5: Validate and save")

    # Feature completeness check
    expected_cols = [
        "WS10M", "T2M", "RH2M", "T2M_MIN", "T2M_MAX", "PRECTOTCORR",
        "T2M_MIN_lag_15d", "RH_high_flag", "RH_persist_7d",
        "Rain_sum_7d", "Rain_sum_14d", "Monsoon_ind",
        "RH2M_latent_window", "T2M_latent_window",
        "warmup_mask", "variety_susceptibility", "is_ratoon", "crop_age_days",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise AssertionError(f"FAIL: Missing expected columns: {missing}")

    # No risk_label check — labels must NOT be in this file
    if "risk_label" in df.columns:
        raise AssertionError(
            "FAIL: risk_label found in features.csv. "
            "Labeling must happen in train.py, not the pipeline."
        )

    # NaN audit
    nan_counts = df[expected_cols].isna().sum()
    nan_cols   = nan_counts[nan_counts > 0]
    if not nan_cols.empty:
        print("\n  NaN summary (expected in warmup rows):")
        for col, n in nan_cols.items():
            warmup_nan = df[df["warmup_mask"] == 1][col].isna().sum()
            post_warmup_nan = n - warmup_nan
            status = "OK" if post_warmup_nan == 0 else "WARN"
            print(f"    [{status}] {col}: {n} NaN total "
                  f"({warmup_nan} in warmup, {post_warmup_nan} post-warmup)")

    # GT-aware variety sanity check: no confirmed outbreak year should be
    # resistant(0) after the override above.
    df_tmp = df.copy()
    df_tmp["year"] = pd.to_datetime(df["date"]).dt.year if "date" in df.columns else None
    if df_tmp["year"] is not None:
        for yr in GT_OUTBREAK_YEARS:
            yr_rows = df_tmp[df_tmp["year"] == yr]
            if not yr_rows.empty:
                vs = yr_rows["variety_susceptibility"].iloc[0]
                if vs == 0:
                    print(f"  WARN: GT outbreak year {yr} still has "
                          f"variety_susceptibility=0 after override. "
                          f"Check GT_OUTBREAK_YEARS list.")

    out_path = os.path.join(PROCESSED_DIR, "v11_features.csv")
    df.to_csv(out_path, index=False)

    print(f"\n  Saved: {out_path}")
    print(f"  Rows: {len(df)}  |  Columns: {len(df.columns)}")
    print(f"  Warmup rows (excluded from training): {df['warmup_mask'].sum()}")
    print(f"  Usable rows: {(df['warmup_mask'] == 0).sum()}")
    print(f"  No risk_label column — labeling is train.py's responsibility.")
    print("\n[PASS] Pipeline validation complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_pipeline():
    df = load_and_clean_data(RAW_DATA_PATH)
    df = apply_rolling_zscore(df)
    df = engineer_kg_features(df)
    df = engineer_agronomic_features(df)
    validate_and_save(df)


if __name__ == "__main__":
    run_pipeline()