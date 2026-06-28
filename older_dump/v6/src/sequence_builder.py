"""
V6 Sequence Builder -- 14-Day Sliding Window Feature Engineering

Root Cause Fixed: V5 used 8 scalar aggregated features per single day.
This destroyed all phase-ordering information critical to Red Rot onset:
    Phase 1 -> Rising humidity trend
    Phase 2 -> Sustained saturation
    Phase 3 -> Rainfall spike
    Phase 4 -> Temperature stabilization

V6 represents each sample as a 14-day trajectory:
    X[t].shape = (SEQUENCE_LENGTH, len(SEQUENCE_FEATURES))

The GRU can then learn the causal ordering of these phases.
All rolling features use strict .shift(1) -- zero look-ahead leakage.
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(__file__))
from config import SEQUENCE_LENGTH, SEQUENCE_FEATURES, TARGET


def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all per-day engineered features (biologically validated).
    STRICT RULE: every rolling/lag uses .shift(1) to prevent leakage.

    Features are derived from 5-paper consensus in unified_model.json.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    # ── Calendar alignment: 1 row = 1 day ────────────────────────────────────
    df = df.set_index("date").asfreq("D").reset_index()
    df.rename(columns={"index": "date"}, errors="ignore")

    # ── Missing data rules (same as V5, biologically correct) ─────────────────
    df["PRECTOTCORR"] = df["PRECTOTCORR"].fillna(0)
    for col in ["T2M", "T2M_MIN", "T2M_MAX", "RH2M"]:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear", limit=3)

    # ── Rolling features (all SHIFTED to prevent leakage) ─────────────────────

    # 7-day rolling mean temperature
    df["T2M_roll7"] = df["T2M"].rolling(7, min_periods=1).mean().shift(1)

    # CRITICAL: T2M_MIN 15-day lag -- SMRA validated R²=0.82–0.87 (best predictor)
    df["T2M_MIN_lag15"] = df["T2M_MIN"].shift(15)

    # 7-day and 14-day rolling mean humidity
    df["RH2M_roll7"]  = df["RH2M"].rolling(7,  min_periods=1).mean().shift(1)
    df["RH2M_roll14"] = df["RH2M"].rolling(14, min_periods=1).mean().shift(1)

    # 5-day linear slope of RH2M -- captures RISING HUMIDITY PHASE (Phase 1)
    def rolling_slope(series, window=5):
        slopes = []
        arr = series.values
        for i in range(len(arr)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                y = arr[i - window + 1 : i + 1]
                x = np.arange(window)
                slope = np.polyfit(x, y, 1)[0]
                slopes.append(slope)
        return pd.Series(slopes, index=series.index)

    df["RH2M_trend_slope"] = rolling_slope(df["RH2M"], window=5).shift(1)

    # 14-day accumulated rainfall -- primary outbreak trigger (~115mm/15d)
    df["PREC_sum14"] = df["PRECTOTCORR"].rolling(14, min_periods=1).sum().shift(1)

    # 3-day recent rainfall -- captures RAINFALL SPIKE EVENT (Phase 3)
    df["PREC_sum3"] = df["PRECTOTCORR"].rolling(3, min_periods=1).sum().shift(1)

    # Daily change in humidity -- captures PHASE TRANSITION signals
    df["delta_RH2M"] = df["RH2M"].diff().shift(1)

    # Monsoon day counter -- days since July 1 of current year (0 outside monsoon)
    def monsoon_day(date):
        monsoon_start = pd.Timestamp(year=date.year, month=7, day=1)
        monsoon_end   = pd.Timestamp(year=date.year, month=10, day=31)
        if monsoon_start <= date <= monsoon_end:
            return (date - monsoon_start).days + 1
        return 0

    df["monsoon_day"] = df["date"].apply(monsoon_day)

    # Temperature suitability flag -- pathogen optimum 29–33°C
    df["temp_suitability"] = ((df["T2M_roll7"] >= 29) & (df["T2M_roll7"] <= 33)).astype(float)

    return df


def build_sequences(
    feature_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Creates 14-day sliding window sequences for the GRU model.

    Returns:
        X: np.ndarray shape (N, SEQUENCE_LENGTH, n_features)
        y: np.ndarray shape (N,)
        dates: DatetimeIndex of the LAST day in each window (the prediction day)
    """
    # Merge labels into feature_df on 'date'
    df = feature_df.merge(labeled_df[["date", TARGET]], on="date", how="left")
    df[TARGET] = df[TARGET].fillna(0).astype(int)

    # Drop rows with insufficient history (first SEQUENCE_LENGTH rows)
    # and rows where critical features are NaN
    df = df.dropna(subset=SEQUENCE_FEATURES + [TARGET]).reset_index(drop=True)

    X_list, y_list, date_list = [], [], []

    for i in range(SEQUENCE_LENGTH, len(df)):
        window = df.iloc[i - SEQUENCE_LENGTH : i]
        target_row = df.iloc[i]

        # Ensure no NaNs in this window
        if window[SEQUENCE_FEATURES].isnull().any().any():
            continue

        X_list.append(window[SEQUENCE_FEATURES].values.astype(np.float32))
        y_list.append(int(target_row[TARGET]))
        date_list.append(target_row["date"])

    X = np.array(X_list)     # (N, 14, n_features)
    y = np.array(y_list)     # (N,)
    dates = pd.DatetimeIndex(date_list)

    pos_rate = y.mean() * 100
    print(f"[SequenceBuilder] Built {len(X)} sequences  |  shape: {X.shape}")
    print(f"[SequenceBuilder] Label balance: {y.sum()} positive ({pos_rate:.1f}%), "
          f"{(y==0).sum()} negative ({100-pos_rate:.1f}%)")

    return X, y, dates


def chronological_split(X, y, dates):
    """
    Split sequences chronologically by year (no data shuffling).
    Train: 2005-2018  |  Val: 2019-2021  |  Test: 2022-2024
    """
    from config import TRAIN_YEARS, VAL_YEARS, TEST_YEARS

    years = dates.year

    tr_mask   = (years >= TRAIN_YEARS[0]) & (years <= TRAIN_YEARS[1])
    val_mask  = (years >= VAL_YEARS[0])   & (years <= VAL_YEARS[1])
    test_mask = (years >= TEST_YEARS[0])  & (years <= TEST_YEARS[1])

    splits = {
        "train": (X[tr_mask],   y[tr_mask],   dates[tr_mask]),
        "val":   (X[val_mask],  y[val_mask],  dates[val_mask]),
        "test":  (X[test_mask], y[test_mask], dates[test_mask]),
    }

    for name, (Xs, ys, _) in splits.items():
        print(f"[SequenceBuilder] {name:5s}: {len(Xs):5d} samples  "
              f"|  positives: {ys.sum():4d} ({ys.mean()*100:.1f}%)")

    return splits


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "v5", "src"))
    import utils
    from label_engine import load_outbreak_events, build_onset_labels

    raw_df = utils.load_raw_data()
    raw_df["date"] = pd.to_datetime(raw_df["date"])

    peaks      = load_outbreak_events()
    labeled_df = build_onset_labels(raw_df, peaks)
    feature_df = build_daily_features(raw_df)

    X, y, dates = build_sequences(feature_df, labeled_df)
    splits      = chronological_split(X, y, dates)

    print(f"\nX_train shape: {splits['train'][0].shape}")
    print(f"Features per step: {len(SEQUENCE_FEATURES)}")
    print(f"Feature names: {SEQUENCE_FEATURES}")
