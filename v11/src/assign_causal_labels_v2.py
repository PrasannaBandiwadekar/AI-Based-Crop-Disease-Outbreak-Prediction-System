"""
assign_causal_labels_v2.py
--------------------------
CANONICAL labeling module. Supersedes assign_causal_labels.py (FAR=10).

CHANGE: Window is now [peak-14, peak-7] — 8 days.

Why [peak-14, peak-7]:
  - peak-7 to peak:    outbreak is imminent, spray too late to protect
  - peak-14 to peak-7: biological signal is building, 7-14 days remain
                        — actionable window for fungicide procurement + spray
  - peak-21 to peak-15: signal too weak / too far out for reliable detection

NOTE: The previous assign_causal_labels.py used LABEL_WINDOW_FAR=10
([peak-10, peak-7], 4-day window). That file is now superseded. Delete it
to avoid any consumer accidentally importing the old constants. All modules
(train.py, validate_pipeline.py, inference_engine.py, explainability.py)
must import from this file only.

Usage in train.py — replace the labeling loop with this function:

    from assign_causal_labels_v2 import assign_labels
    df = assign_labels(df, gt_path)

CHANGELOG (v2.1):
  - [BUG]    GT-authoritative override added after resistant-variety
             suppression. If a GT event window is completely zeroed out
             by the resistant-variety rule, the GT is reinstated as
             risk_label=1 for that window.

             Rationale: the GT file records literature-confirmed outbreaks.
             An outbreak that actually occurred is proof the variety in that
             field was not fully resistant — the probabilistic variety
             simulator in dataset_pipeline.py occasionally draws resistant(0)
             for years with known events (e.g. 2011, 2019), which is a
             simulation error, not biology. The GT is the authoritative source;
             the simulated variety is a feature approximation. When they
             conflict, GT wins.

             Without this fix, six GT events produced zero positive rows,
             making those outbreaks invisible to the model during training
             and validation. The fix restores those rows to risk_label=1
             and prints an explicit override message for audit purposes.
"""

import pandas as pd

# Lead-time window in days before outbreak peak.
# Both values are the number of days BEFORE peak_start.
LABEL_WINDOW_FAR  = 14   # furthest day from peak that gets label=1
LABEL_WINDOW_NEAR = 7    # closest day to peak that gets label=1
                          # (days peak-6 through peak are label=0 — too late)


def assign_labels(df: pd.DataFrame, gt_path: str) -> pd.DataFrame:
    """
    Assign risk_label=1 to rows in the actionable spray window.

    For each GT outbreak event at peak_start:
        label=1 for dates in [peak_start - 14d, peak_start - 7d]
        label=0 everywhere else

    Rows where variety_susceptibility==0 (resistant variety) keep label=0
    unless the GT confirms an outbreak occurred in that window — in that
    case the GT overrides the simulated variety (see GT override block).

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
        win_start = peak - pd.Timedelta(days=LABEL_WINDOW_FAR)   # peak - 14
        win_end   = peak - pd.Timedelta(days=LABEL_WINDOW_NEAR)  # peak - 7

        mask = (df["date"] >= win_start) & (df["date"] <= win_end)
        df.loc[mask, "risk_label"] = 1

    # Resistant variety suppression: weather cannot trigger outbreak in a
    # fully resistant host under normal circumstances.
    resistant_mask = df["variety_susceptibility"] == 0
    n_suppressed   = df.loc[resistant_mask, "risk_label"].sum()
    df.loc[resistant_mask, "risk_label"] = 0

    # ── GT-authoritative override ────────────────────────────────────────────
    # If a GT event window was completely zeroed by the resistant-variety rule,
    # restore it. The GT records a literature-confirmed outbreak — that is proof
    # the crop in that season was not fully resistant. The probabilistic variety
    # simulator in dataset_pipeline.py is an approximation; when it conflicts
    # with a confirmed event, the GT takes precedence.
    #
    # This does NOT mean resistant varieties can have outbreaks in general —
    # only that when the GT explicitly confirms one, we trust the GT over the
    # simulated variety draw.
    n_overrides = 0
    for _, gt in gt_df.iterrows():
        peak      = gt["peak_start"]
        win_start = peak - pd.Timedelta(days=LABEL_WINDOW_FAR)
        win_end   = peak - pd.Timedelta(days=LABEL_WINDOW_NEAR)
        window_mask = (df["date"] >= win_start) & (df["date"] <= win_end)

        if df.loc[window_mask, "risk_label"].sum() == 0:
            # Window exists in the dataframe (date range is not out of bounds)
            if window_mask.sum() > 0:
                df.loc[window_mask, "risk_label"] = 1
                n_overrides += 1
                print(f"  GT override: event {peak.date()} restored "
                      f"(was fully suppressed by resistant variety simulation)")

    n_pos = df["risk_label"].sum()
    n_gt  = len(gt_df)

    print(f"  GT events loaded:          {n_gt}")
    print(f"  Window: [peak-{LABEL_WINDOW_FAR}d, peak-{LABEL_WINDOW_NEAR}d]"
          f" = {LABEL_WINDOW_FAR - LABEL_WINDOW_NEAR + 1} days per event")
    print(f"  Expected positives (max):  {n_gt * (LABEL_WINDOW_FAR - LABEL_WINDOW_NEAR + 1)}")
    print(f"  Positives suppressed (resistant variety): {n_suppressed}")
    print(f"  GT overrides applied:      {n_overrides}")
    print(f"  Final risk_label=1 count:  {n_pos}")

    # Sanity check: every GT event must now produce at least 1 positive row.
    # After the override block above, the only remaining zero-positive case
    # is a GT event whose date window falls entirely outside the dataframe
    # (e.g. peak is too close to the start of the dataset).
    for _, gt in gt_df.iterrows():
        peak      = gt["peak_start"]
        win_start = peak - pd.Timedelta(days=LABEL_WINDOW_FAR)
        win_end   = peak - pd.Timedelta(days=LABEL_WINDOW_NEAR)
        event_pos = df.loc[
            (df["date"] >= win_start) & (df["date"] <= win_end),
            "risk_label"
        ].sum()
        if event_pos == 0:
            print(f"  WARNING: GT event {peak.date()} still has 0 positive rows "
                  f"— window is outside the dataset date range.")

    return df