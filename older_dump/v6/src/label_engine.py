"""
V6 Label Engine -- TRUE Outbreak Onset Labels

Root Cause Fixed: V5 used proxy labels (future env_interaction_score >= 2),
which trained the model to predict "will tomorrow also be humid?" -- not disease onset.

Correct Definition:
    label[t] = 1  if any peak_start falls in [t + LEAD_MIN, t + LEAD_MAX]
    label[t] = 0  otherwise

This directly targets the biological event we care about:
"Is an outbreak going to start in the next 3–14 days?"
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(__file__))
from config import (
    OUTBREAK_EVENTS_PATH, LEAD_MIN_DAYS, LEAD_MAX_DAYS, TARGET
)


def load_outbreak_events() -> pd.DataFrame:
    """Load and deduplicate outbreak events by unique peak_start date."""
    df = pd.read_csv(OUTBREAK_EVENTS_PATH)
    df["peak_start"] = pd.to_datetime(df["peak_start"])
    # We only care about unique peak dates (one biological event per date)
    unique_peaks = df["peak_start"].drop_duplicates().sort_values().reset_index(drop=True)
    print(f"[LabelEngine] Loaded {len(df)} outbreak records -> {len(unique_peaks)} unique peak dates")
    return unique_peaks


def build_onset_labels(weather_df: pd.DataFrame, peak_dates: pd.Series) -> pd.DataFrame:
    """
    Annotates each day t in weather_df with:
        outbreak_onset = 1  if any peak_start ∈ [t + LEAD_MIN, t + LEAD_MAX]
        outbreak_onset = 0  otherwise

    Args:
        weather_df: Daily weather DataFrame with a 'date' column.
        peak_dates:  Series of unique outbreak peak_start dates.

    Returns:
        weather_df with an added 'outbreak_onset' column.
    """
    df = weather_df.copy()
    df[TARGET] = 0

    positive_days = 0
    for peak in peak_dates:
        # The window of days from which the peak is "detectable within lead window"
        earliest_alert = peak - pd.Timedelta(days=LEAD_MAX_DAYS)
        latest_alert   = peak - pd.Timedelta(days=LEAD_MIN_DAYS)
        mask = (df["date"] >= earliest_alert) & (df["date"] <= latest_alert)
        new_positives = mask.sum() - df.loc[mask, TARGET].sum()  # avoid double counting
        df.loc[mask, TARGET] = 1
        positive_days += new_positives

    total = len(df)
    pct   = positive_days / total * 100
    print(f"[LabelEngine] Onset Label distribution:")
    print(f"  Positive (outbreak_onset=1): {positive_days:4d} days  ({pct:.1f}%)")
    print(f"  Negative (outbreak_onset=0): {total - positive_days:4d} days  ({100-pct:.1f}%)")

    return df


def build_biological_labels(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Annotates each day t in weather_df with:
        outbreak_onset = 1  if t falls in [Aug 1, Aug 14] of any year.
        outbreak_onset = 0  otherwise

    This allows the model to learn 'pre-peak' environmental conditions across all 20 years,
    compensating for the fact that we only have confirmed peak dates for 2019-2021.
    """
    df = weather_df.copy()
    # Month 8 is August. Day between 8 and 15 covers the [peak-7, peak] window for the typical Aug 15 peak.
    df[TARGET] = ((df["date"].dt.month == 8) & (df["date"].dt.day >= 8) & (df["date"].dt.day <= 15)).astype(int)

    positive_days = df[TARGET].sum()
    total = len(df)
    pct   = positive_days / total * 100
    print(f"[LabelEngine] Biological Label distribution (Option A):")
    print(f"  Positive (outbreak_onset=1): {positive_days:4d} days  ({pct:.1f}%)")
    print(f"  Negative (outbreak_onset=0): {total - positive_days:4d} days  ({100-pct:.1f}%)")

    return df


def validate_label_alignment(labeled_df: pd.DataFrame, peak_dates: pd.Series):
    """
    Sanity check: verify every known peak_start is reachable from at least
    one positive label day within the configured lead window.
    """
    print("\n[LabelEngine] Validating label-to-peak alignment...")
    covered = 0
    for peak in peak_dates:
        earliest = peak - pd.Timedelta(days=LEAD_MAX_DAYS)
        latest   = peak - pd.Timedelta(days=LEAD_MIN_DAYS)
        window = labeled_df[
            (labeled_df["date"] >= earliest) &
            (labeled_df["date"] <= latest) &
            (labeled_df[TARGET] == 1)
        ]
        if not window.empty:
            covered += 1
        else:
            print(f"  WARNING: peak {peak.date()} has NO positive label in data range.")

    print(f"  Coverage: {covered}/{len(peak_dates)} peaks have positive labels in data.")
    if covered == len(peak_dates):
        print("  PASS: All outbreak peaks are correctly reachable.")
    else:
        print("  WARN: Some peaks fall outside training data date range (expected for test set).")


if __name__ == "__main__":
    from utils import load_raw_data

    raw_df = load_raw_data()
    raw_df["date"] = pd.to_datetime(raw_df["date"])

    peaks = load_outbreak_events()
    labeled = build_onset_labels(raw_df, peaks)
    validate_label_alignment(labeled, peaks)
    print(f"\nSample positives:\n{labeled[labeled[TARGET]==1][['date',TARGET]].head(20)}")
