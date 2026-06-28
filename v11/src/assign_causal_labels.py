"""
assign_causal_labels_v2.py
--------------------------
Drop-in replacement for the labeling block in train.py.

CHANGE: Window is now [peak-10, peak-7] — 4 days.

Why [peak-10, peak-7]:
  - peak-7 to peak:    outbreak is imminent, spray too late to protect
  - peak-10 to peak-7: biological signal is building, 7-10 days remain
                        — exactly the actionable spray window
  - peak-14 to peak-11: signal too weak / too far out for reliable detection

This replaces the previous [O-7, O-3] window in train.py which was
misaligned with the V8 pipeline's [peak-10, peak-2] window, causing
the two label sets to disagree.

Usage in train.py — replace the labeling loop with this function:

    from assign_causal_labels_v2 import assign_labels
    df = assign_labels(df, gt_path)
"""

import pandas as pd

# Lead-time window in days before outbreak peak.
# Both values are the number of days BEFORE peak_start.
LABEL_WINDOW_FAR  = 10   # furthest day from peak that gets label=1
LABEL_WINDOW_NEAR = 7    # closest day to peak that gets label=1
                          # (days peak-6 through peak are label=0 — too late)


def assign_labels(df: pd.DataFrame, gt_path: str) -> pd.DataFrame:
    """
    Assign risk_label=1 to rows in the actionable spray window.

    For each GT outbreak event at peak_start:
        label=1 for dates in [peak_start - 10d, peak_start - 7d]
        label=0 everywhere else

    Rows where variety_susceptibility==0 (resistant variety) keep label=0
    regardless of weather — resistant varieties do not develop Red Rot
    regardless of environmental pressure.

    Args:
        df       : feature dataframe with 'date' and 'variety_susceptibility'
        gt_path  : path to GT CSV with 'peak_start' column

    Returns:
        df with 'risk_label' column added (int, 0 or 1)
    """
    gt_df = pd.read_csv(gt_path)
    gt_df["peak_start"] = pd.to_datetime(gt_df["peak_start"])

    df = df.copy()
    df["risk_label"] = 0

    for _, gt in gt_df.iterrows():
        peak      = gt["peak_start"]
        win_start = peak - pd.Timedelta(days=LABEL_WINDOW_FAR)   # peak - 10
        win_end   = peak - pd.Timedelta(days=LABEL_WINDOW_NEAR)  # peak - 7

        mask = (df["date"] >= win_start) & (df["date"] <= win_end)
        df.loc[mask, "risk_label"] = 1

    # Resistant variety override: weather cannot trigger outbreak in resistant host
    resistant_mask = df["variety_susceptibility"] == 0
    n_suppressed   = df.loc[resistant_mask, "risk_label"].sum()
    df.loc[resistant_mask, "risk_label"] = 0

    n_pos = df["risk_label"].sum()
    n_gt  = len(gt_df)

    print(f"  GT events loaded:          {n_gt}")
    print(f"  Window: [peak-{LABEL_WINDOW_FAR}d, peak-{LABEL_WINDOW_NEAR}d]"
          f" = {LABEL_WINDOW_FAR - LABEL_WINDOW_NEAR + 1} days per event")
    print(f"  Expected positives (max):  {n_gt * (LABEL_WINDOW_FAR - LABEL_WINDOW_NEAR + 1)}")
    print(f"  Positives suppressed (resistant variety): {n_suppressed}")
    print(f"  Final risk_label=1 count:  {n_pos}")

    # Sanity check: every GT event should produce at least 1 positive row
    for _, gt in gt_df.iterrows():
        peak      = gt["peak_start"]
        win_start = peak - pd.Timedelta(days=LABEL_WINDOW_FAR)
        win_end   = peak - pd.Timedelta(days=LABEL_WINDOW_NEAR)
        event_pos = df.loc[
            (df["date"] >= win_start) & (df["date"] <= win_end),
            "risk_label"
        ].sum()
        if event_pos == 0:
            print(f"  WARNING: GT event {peak.date()} produced 0 positive rows "
                  f"(all suppressed by resistant variety or date out of range).")

    return df