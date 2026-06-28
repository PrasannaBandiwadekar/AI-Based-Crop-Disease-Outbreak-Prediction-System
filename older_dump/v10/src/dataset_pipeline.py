"""
V10.6 DATASET PIPELINE — ROBUST CONTRAST EDITION
==================================================

CORE FIX FOR GENERALIZATION:
  The TCN was memorizing training sequences because raw weather values
  look different in 2019-2020 vs 2005-2015 (climate drift).

  Solution: Transform all features into CONTRAST SCORES that are
  invariant to absolute climate level. Every feature is expressed as
  "how many sigma above the rolling baseline is this today?"

  This makes pre-outbreak windows look similar across years even if
  absolute RH or temperature has shifted.

FEATURES (all expressed as z-scores vs trailing baseline):
  - RH anomaly vs 30-day rolling mean (z-score)
  - T2M anomaly vs 30-day rolling mean (z-score)
  - RH anomaly vs 90-day rolling mean (longer baseline)
  - 3-day slope of RH (standardized)
  - 5-day slope of RH (standardized)
  - Trigger pulse count over 3 days
  - Persistence: consecutive days above 80th percentile RH
  - Rainfall spike indicator
  - Seasonal anomaly (vs same calendar period in prior years)
  - Biological lag features

ALL features are shift(1) causal.
"""

import pandas as pd
import numpy as np
import os
from datetime import timedelta

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

RAW_DATA_PATH = os.path.join(
    PROJECT_ROOT, "v9", "data", "raw",
    "POWER_Point_Daily_20050101_20241231_016d54N_069d78E_LST.csv"
)
GT_PATH = os.path.join(
    PROJECT_ROOT, "research_comp", "evidence_base",
    "outbreak_events", "sangli_synthetic_gt.csv"
)
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def rolling_zscore(series, window):
    """
    Z-score of x(t-1) vs rolling mean/std over the prior `window` days.
    Uses shifted series to maintain causality.
    """
    mu  = series.rolling(window, min_periods=max(window//4, 5)).mean()
    sig = series.rolling(window, min_periods=max(window//4, 5)).std().clip(lower=0.01)
    return (series - mu) / sig


def seasonal_anomaly_fast(series, date_series, window_days=21):
    """
    Efficient seasonal anomaly:
    For each day, compute anomaly vs same DOY ± window/2 in prior years.
    Uses a vectorized approach over pre-computed DOY groups.
    """
    s    = series.values.copy()
    doy  = date_series.dt.dayofyear.values
    year = date_series.dt.year.values
    n    = len(s)
    anom = np.full(n, np.nan)
    half = window_days // 2

    # Group by DOY for fast lookup
    from collections import defaultdict
    doy_vals = defaultdict(list)  # doy -> list of (year, value)
    for i in range(n):
        doy_vals[doy[i]].append((year[i], s[i]))

    for i in range(n):
        y_i = year[i]; d_i = doy[i]
        # Collect values from same DOY±half in PRIOR years
        combined = []
        for d in range(d_i - half, d_i + half + 1):
            # Wrap DOY (1-365)
            d_w = ((d - 1) % 365) + 1
            for (yr, val) in doy_vals.get(d_w, []):
                if yr < y_i and not np.isnan(val):
                    combined.append(val)
        if len(combined) >= 7:
            mu  = np.mean(combined)
            sig = max(np.std(combined), 0.01)
            anom[i] = (s[i] - mu) / sig  # z-score

    return pd.Series(anom, index=series.index)


def generate_v10_final_features():
    print("V10.6 ROBUST CONTRAST PIPELINE: Generating features...")

    df = pd.read_csv(RAW_DATA_PATH, skiprows=14)
    df['date'] = pd.to_datetime(
        df['YEAR'].astype(str) + df['DOY'].astype(str), format='%Y%j'
    )
    df = df.sort_values('date').reset_index(drop=True)

    # ── STRICT CAUSAL SHIFT ────────────────────────────────────────────────
    s = df.shift(1)   # all raw signals are yesterday's observed values

    # ── 1. RAW CAUSAL WEATHER ─────────────────────────────────────────────
    df['RH2M']        = s['RH2M']
    df['T2M']         = s['T2M']
    df['T2M_MIN']     = s['T2M_MIN']
    df['PRECTOTCORR'] = s['PRECTOTCORR']

    # ── 2. ROLLING Z-SCORES (primary contrast signals) ────────────────────
    # These encode "how many sigma above recent average" — invariant to
    # absolute climate level changes across years
    df['RH_z14']   = rolling_zscore(s['RH2M'],        14)
    df['RH_z30']   = rolling_zscore(s['RH2M'],        30)
    df['RH_z90']   = rolling_zscore(s['RH2M'],        90)
    df['T_z14']    = rolling_zscore(s['T2M'],         14)
    df['Rain_z14'] = rolling_zscore(s['PRECTOTCORR'], 14)
    df['Rain_z30'] = rolling_zscore(s['PRECTOTCORR'], 30)

    # ── 3. RATE OF CHANGE (slope) ─────────────────────────────────────────
    df['RH_slope_3']   = (s['RH2M']  - s['RH2M'].shift(3))  / 3.0
    df['RH_slope_7']   = (s['RH2M']  - s['RH2M'].shift(7))  / 7.0
    df['T_slope_3']    = (s['T2M']   - s['T2M'].shift(3))   / 3.0

    # ── 4. TRIGGER PULSE ──────────────────────────────────────────────────
    df['Trigger_Pulse']  = (
        (s['RH2M'] > 84.0) & (s['T2M'] > 25.5) & (s['PRECTOTCORR'] > 1.0)
    ).astype(int)
    df['Trigger_3d_sum'] = df['Trigger_Pulse'].rolling(3, min_periods=1).sum()
    df['Trigger_7d_sum'] = df['Trigger_Pulse'].rolling(7, min_periods=1).sum()

    # ── 5. PERSISTENCE ABOVE THRESHOLD ───────────────────────────────────
    # Consecutive days above 80th-percentile rolling RH
    rh_p80 = s['RH2M'].rolling(90, min_periods=30).quantile(0.80)
    high_rh = (s['RH2M'] > rh_p80).astype(int)
    df['RH_persist'] = high_rh.rolling(7, min_periods=1).sum()

    # ── 6. RAINFALL SPIKE ─────────────────────────────────────────────────
    df['Rain_spike']   = (s['PRECTOTCORR'] > 5.0).astype(int)
    df['Rain_sum_7']   = s['PRECTOTCORR'].rolling(7, min_periods=1).sum()

    # ── 7. SEASONAL ANOMALY Z-SCORE ───────────────────────────────────────
    print("  Computing seasonal anomaly z-scores (this may take ~60 s)...")
    df['RH_season_z']   = seasonal_anomaly_fast(s['RH2M'],        df['date'], window_days=21)
    df['T_season_z']    = seasonal_anomaly_fast(s['T2M'],         df['date'], window_days=21)
    df['Rain_season_z'] = seasonal_anomaly_fast(s['PRECTOTCORR'], df['date'], window_days=21)

    # ── 8. BIOLOGICAL LAG ─────────────────────────────────────────────────
    df['RH2M_lag_15']    = s['RH2M'].shift(14)
    df['T2M_MIN_lag_15'] = s['T2M_MIN'].shift(14)

    # ── 9. AGRONOMIC ──────────────────────────────────────────────────────
    df['ratoon_flag']      = (df['date'].dt.year % 2 == 0).astype(int)
    df['sanitation_score'] = 0.7

    # ── 10. BINARY RISK LABELS ────────────────────────────────────────────
    df['risk_label']  = 0
    df['ignore_mask'] = 0

    gt = pd.read_csv(GT_PATH)
    gt['peak_start'] = pd.to_datetime(gt['peak_start'])

    for _, row in gt.iterrows():
        start = row['peak_start']
        df.loc[
            (df['date'] >= start - timedelta(days=7)) &
            (df['date'] <= start - timedelta(days=3)),
            'risk_label'
        ] = 1
        df.loc[
            (df['date'] >= start - timedelta(days=2)) &
            (df['date'] <= start + timedelta(days=2)),
            'ignore_mask'
        ] = 1

    df = (
        df[df['ignore_mask'] == 0]
        .drop(columns=['ignore_mask'])
        .dropna()
        .reset_index(drop=True)
    )

    out_path = os.path.join(PROCESSED_DIR, "features.csv")
    df.to_csv(out_path, index=False)

    pos = int(df['risk_label'].sum())
    neg = int((df['risk_label'] == 0).sum())
    print(f"V10.6 DATA READY | Samples={len(df)} | Positives={pos} | Negatives={neg}")
    print(f"  Class ratio (neg/pos): {neg/max(pos,1):.1f}")

    # Quick separability check
    feat_check = ['RH2M', 'RH_z14', 'RH_z30', 'RH_z90', 'RH_season_z',
                  'Trigger_Pulse', 'Trigger_3d_sum', 'RH_persist']
    df_pos = df[df['risk_label'] == 1]
    df_neg = df[df['risk_label'] == 0]
    print("\n  Feature separability (Cohen's d):")
    for f in feat_check:
        if f in df.columns:
            pm = df_pos[f].mean(); nm = df_neg[f].mean()
            ps = df_pos[f].std();  ns = df_neg[f].std()
            d  = abs(pm - nm) / max((ps + ns) / 2, 0.001)
            print(f"    {f:22s}: pos={pm:6.2f} neg={nm:6.2f} d={d:.3f}")


if __name__ == "__main__":
    generate_v10_final_features()
