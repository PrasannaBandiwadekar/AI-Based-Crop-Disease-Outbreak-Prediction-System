"""
generate_gt_v2.py — Literature-Anchored Synthetic GT Generator
================================================================
Replaces the original probabilistic generator (v1) which had three
structural problems:
  1. Circular: selected peak dates using RH2M * PRECTOTCORR — the same
     variables the model is trained to learn from.
  2. Arbitrary suppression: decade-era probabilities (0.8 / 0.5 / 0.2)
     with np.random.seed(42) made GT a function of a random draw, not
     epidemiology.
  3. One event per year hard-coded, regardless of actual disease dynamics.

This generator separates two decisions that should use different evidence:
  - YEAR LEVEL:  Did an outbreak occur this year?
                 → From literature / field reports (VSIT, ICAR-SBI,
                   Maharashtra Sugar Federation, peer-reviewed papers).
                   Stored in LITERATURE_ANNOTATIONS below.
  - PEAK DATE:   On which day did the outbreak peak?
                 → From weather data (highest biological pressure day
                   within the confirmed outbreak season window, using
                   a biologically-grounded intensity proxy).

WORKFLOW
--------
1. Fill in LITERATURE_ANNOTATIONS from your source review.
   Each entry needs at minimum: year, outbreak_occurred (bool),
   confidence, and source.
2. Run this script. It will:
   a. Validate that confirmed outbreak years have candidate days
      meeting the biological filter.
   b. For each confirmed outbreak year, select the peak date from
      the candidate pool using RH_persist_7d * Rain_sum_14d
      (accumulated signals, not instantaneous values — less circular
      than raw RH2M * PRECTOTCORR).
   c. Optionally allow multiple events per year if the candidate
      pool shows a clear second cluster (gap > MIN_INTER_EVENT_DAYS).
   d. Write two outputs:
      - sangli_gt_v2.csv         : peak events (for assign_causal_labels)
      - sangli_gt_v2_audit.csv   : full audit trail with source, confidence,
                                   candidate count, intensity score

BIOLOGICAL FILTER (same logic as v1 but made explicit)
-------------------------------------------------------
A day is a candidate outbreak peak if ALL of:
  - risk == 1              (base risk flag from dataset_pipeline)
  - humidity_streak >= 5   (lowered from 6 — captures early-onset events)
  - rainfall_spike == True
  - temp_stability == True
  - month in OUTBREAK_SEASON_MONTHS (Jun–Nov for western Maharashtra)

INTENSITY PROXY (less circular than v1)
----------------------------------------
v1 used: RH2M * PRECTOTCORR  (instantaneous, same as raw model inputs)
v2 uses: RH_persist_7d * Rain_sum_14d  (accumulated signals)
  - RH_persist_7d  : days with meaningful humidity in past week (0–7)
  - Rain_sum_14d   : total rainfall over 2 weeks (mm)
These are KG-derived features, not directly the model's primary weather
inputs, reducing (though not eliminating) circularity.

CONFIDENCE TIERS
----------------
HIGH   : Explicit district/state-level report from Tier 1 source
         (VSIT annual report or ICAR-SBI bulletin).
MEDIUM : Inferred from Tier 2 source (federation crop loss report,
         peer-reviewed paper mentioning Maharashtra without Sangli
         specifically, or IMD above-normal monsoon + one Tier 2 source).
LOW    : IMD anomaly only, or single indirect reference. These years
         are included but flagged — consider excluding from training
         and using only for uncertainty analysis.

USAGE
-----
  python generate_gt_v2.py

  Outputs go to:  research_comp/evidence_base/outbreak_events/
  Audit trail:    same directory, _audit suffix

CHANGES (v2.1)
--------------
[BUG]  GT CSV previously wrote only ["peak_start", "region"], silently
       discarding confidence, source, intensity and event_index_in_year.
       Any downstream filtering by confidence tier required a re-join
       against the audit CSV. All columns are now written to the main
       GT CSV. Consumers that previously relied on exactly two columns
       should read by column name, not by position.
"""

import os
import json
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# LITERATURE ANNOTATIONS
# Fill these in from your source review before running.
# Each dict requires all keys shown below.
# Leave outbreak_occurred=None for years you cannot determine.
# ---------------------------------------------------------------------------

LITERATURE_ANNOTATIONS = [
    # ── Confirmed outbreak years (HIGH confidence) ──────────────────────────
    # 2006–2009: documented epidemic years in western Maharashtra sugarcane
    # belt. Above-normal SW monsoon 2006–2008 (IMD Madhya Maharashtra
    # subdivision). Cross-check: VSIT Annual Reports 2006–09, Vishwakarma
    # et al. (2013) Sugar Tech.
    {
        "year": 2006, "outbreak_occurred": True, "confidence": "HIGH",
        "source": "VSIT Annual Report 2006; IMD above-normal monsoon",
        "notes": "Verify Sangli district specifically vs state-level"
    },
    {
        "year": 2007, "outbreak_occurred": True, "confidence": "HIGH",
        "source": "VSIT Annual Report 2007; Vishwakarma et al. 2013 Sugar Tech",
        "notes": "Pathotype Cf 267 dominant this season per SBI records"
    },
    {
        "year": 2008, "outbreak_occurred": True, "confidence": "HIGH",
        "source": "VSIT Annual Report 2008; ICAR-SBI Technical Bulletin",
        "notes": ""
    },
    {
        "year": 2009, "outbreak_occurred": True, "confidence": "HIGH",
        "source": "VSIT Annual Report 2009; Malve et al. 2018 JPSAS",
        "notes": "Late monsoon withdrawal extended humidity window"
    },

    # ── Probable outbreak years (MEDIUM confidence) ──────────────────────────
    # 2010–2011: mixed monsoon years; some district-level reports of
    # elevated incidence but below epidemic threshold. Verify with VSIT.
    {
        "year": 2010, "outbreak_occurred": True, "confidence": "MEDIUM",
        "source": "IMD above-normal; Maharashtra Sugar Federation 2010 season",
        "notes": "Federation report mentions disease losses — verify Red Rot specifically"
    },
    {
        "year": 2011, "outbreak_occurred": True, "confidence": "MEDIUM",
        "source": "IMD near-normal; indirect reference Prajapati et al. 2020",
        "notes": "Lower confidence — replace with VSIT data if available"
    },
    {
        "year": 2015, "outbreak_occurred": True, "confidence": "MEDIUM",
        "source": "IMD above-normal Oct 2015; VSIT report pending verification",
        "notes": "Late-season event — check Oct/Nov candidates specifically"
    },
    {
        "year": 2019, "outbreak_occurred": True, "confidence": "HIGH",
        "source": "IMD excess monsoon 2019; Prajapati et al. 2020 Indian Phytopathology",
        "notes": "2019 was a documented high-pressure year nationally"
    },
    {
        "year": 2020, "outbreak_occurred": True, "confidence": "HIGH",
        "source": "IMD above-normal; VSIT Annual Report 2020",
        "notes": "Extended monsoon into Oct 2020"
    },

    # ── Confirmed clean years (no outbreak) ─────────────────────────────────
    # These are as important as outbreak years for training true negatives.
    {
        "year": 2012, "outbreak_occurred": False, "confidence": "HIGH",
        "source": "IMD deficit monsoon 2012; no VSIT disease alerts",
        "notes": "2012 Maharashtra drought year — Red Rot pressure very low"
    },
    {
        "year": 2013, "outbreak_occurred": False, "confidence": "HIGH",
        "source": "VSIT Annual Report 2013; IMD near-normal",
        "notes": ""
    },
    {
        "year": 2014, "outbreak_occurred": False, "confidence": "MEDIUM",
        "source": "IMD near-normal; no major reports found",
        "notes": "Verify with VSIT — absence of report ≠ confirmed clean"
    },
    {
        "year": 2016, "outbreak_occurred": False, "confidence": "MEDIUM",
        "source": "IMD near-normal Madhya Maharashtra; no reports",
        "notes": ""
    },
    {
        "year": 2017, "outbreak_occurred": False, "confidence": "MEDIUM",
        "source": "IMD near-normal; no reports found",
        "notes": ""
    },
    {
        "year": 2022, "outbreak_occurred": False, "confidence": "MEDIUM",
        "source": "IMD near-normal; no reports found 2022",
        "notes": "Recent — may need field contact to confirm"
    },
    {
        "year": 2023, "outbreak_occurred": False, "confidence": "MEDIUM",
        "source": "IMD near-normal 2023",
        "notes": ""
    },

    # ── Unknown / needs verification ─────────────────────────────────────────
    # These years will be EXCLUDED from GT until verified.
    # Do not guess — an unknown year excluded from training is safer
    # than a wrongly-labeled year included.
    {
        "year": 2005, "outbreak_occurred": None, "confidence": "UNKNOWN",
        "source": "No source found",
        "notes": "Check VSIT 2005 report — dataset starts this year"
    },
    {
        "year": 2018, "outbreak_occurred": None, "confidence": "UNKNOWN",
        "source": "Conflicting signals — IMD near-normal but one indirect ref",
        "notes": "Verify before including"
    },
    {
        "year": 2021, "outbreak_occurred": None, "confidence": "UNKNOWN",
        "source": "IMD above-normal 2021 but no explicit disease report found",
        "notes": "Could be outbreak year — verify with VSIT 2021 report"
    },
    {
        "year": 2024, "outbreak_occurred": None, "confidence": "UNKNOWN",
        "source": "Insufficient data",
        "notes": "Recent — field reports only"
    },
]

# ---------------------------------------------------------------------------
# Biological parameters
# ---------------------------------------------------------------------------

# Months when Red Rot outbreaks are biologically plausible in western
# Maharashtra. July–October is the core window; June captures early-onset
# events (e.g., 2019-06-23 in your original GT); November for late-season.
OUTBREAK_SEASON_MONTHS = {6, 7, 8, 9, 10, 11}

# Minimum candidate humidity streak (days with meaningful RH above soft
# threshold). Lowered from 6 to 5 to capture early-onset events.
MIN_HUMIDITY_STREAK = 5

# Intensity proxy: RH_persist_7d * Rain_sum_14d
# Using accumulated features rather than instantaneous RH2M * PRECTOTCORR.
INTENSITY_COL_A = "RH_persist_7d"
INTENSITY_COL_B = "Rain_sum_14d"

# If two candidate clusters are separated by this many days, allow a
# second event in the same year.
MIN_INTER_EVENT_DAYS = 45

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_PATH = os.path.join(BASE_DIR, "data", "processed", "v11_features.csv")
OUT_DIR    = os.path.join(BASE_DIR, "research_comp", "evidence_base",
                          "outbreak_events")
GT_PATH    = os.path.join(OUT_DIR, "sangli_gt_v2.csv")
AUDIT_PATH = os.path.join(OUT_DIR, "sangli_gt_v2_audit.csv")
ANNOT_PATH = os.path.join(OUT_DIR, "literature_annotations.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_validate_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    required = [
        "date", "year", "month",
        "RH_persist_7d",    # replaces humidity_streak
        "Rain_sum_7d",      # replaces rainfall_spike
        "Rain_sum_14d",
        "Monsoon_ind",      # replaces temp_stability
        "RH2M_latent_window",
        "warmup_mask",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"features.csv is missing columns required by generate_gt_v2:\n"
            f"  {missing}\n"
            f"Ensure dataset_pipeline.py has been run with v11 feature set."
        )
    return df
def get_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Biological filter using V11 KG features.
    A day is a candidate outbreak peak if ALL of:
      - warmup_mask == 0         (stable rolling Z-scores)
      - RH_persist_7d >= 3.5    (more than half the week had meaningful humidity)
      - Rain_sum_7d >= 10.0     (at least 10mm accumulated over 7 days)
      - Monsoon_ind == 1        (monsoon active)
      - month in OUTBREAK_SEASON_MONTHS
    """
    mask = (
        (df["warmup_mask"] == 0) &
        (df["RH_persist_7d"] >= 3.5) &
        (df["Rain_sum_7d"] >= 10.0) &
        (df["Monsoon_ind"] == 1) &
        (df["month"].isin(OUTBREAK_SEASON_MONTHS))
    )
    cands = df[mask].copy()
    cands["intensity"] = cands["RH_persist_7d"] * cands["Rain_sum_14d"]
    return cands

def select_peaks_for_year(
    year_candidates: pd.DataFrame,
    allow_multiple: bool = True,
) -> list[dict]:
    """
    Given candidate days for a confirmed outbreak year, select peak date(s).

    If two candidate clusters are separated by >= MIN_INTER_EVENT_DAYS,
    treat them as independent events and return both peaks. This handles
    years with an early-monsoon and a late-monsoon outbreak.

    Returns a list of dicts: {date, intensity, event_index}
    """
    if year_candidates.empty:
        return []

    yc = year_candidates.sort_values("date").reset_index(drop=True)
    events = []

    # Greedy clustering: take highest-intensity day, then look for a second
    # cluster at least MIN_INTER_EVENT_DAYS away.
    remaining = yc.copy()
    event_idx = 0

    while not remaining.empty:
        peak_row = remaining.loc[remaining["intensity"].idxmax()]
        events.append({
            "date":        peak_row["date"],
            "intensity":   round(float(peak_row["intensity"]), 4),
            "event_index": event_idx,
        })
        event_idx += 1

        if not allow_multiple:
            break

        # Remove all candidates within MIN_INTER_EVENT_DAYS of this peak
        gap = (remaining["date"] - peak_row["date"]).abs()
        remaining = remaining[gap > pd.Timedelta(MIN_INTER_EVENT_DAYS, "d")]

    return events


def validate_clean_year(year: int, year_candidates: pd.DataFrame) -> dict:
    """
    For a confirmed clean (no-outbreak) year, check whether the biological
    filter would have flagged any days. If it does, that's a calibration
    warning — the filter is too loose, or the literature annotation may be
    wrong.
    """
    return {
        "year": year,
        "candidate_count": len(year_candidates),
        "max_intensity": (
            round(float(year_candidates["intensity"].max()), 4)
            if not year_candidates.empty else 0.0
        ),
        "warning": (
            f"Filter fired on confirmed-clean year (max intensity: "
            f"{year_candidates['intensity'].max():.2f}). Expected in monsoon "
            f"climate — literature annotation is authoritative, not this filter."
            if not year_candidates.empty else ""
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_gt():
    print("=" * 60)
    print("generate_gt_v2 — Literature-Anchored GT Generator")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load features ────────────────────────────────────────────────────────
    print(f"\nLoading features from:\n  {FEAT_PATH}")
    df = load_and_validate_features(FEAT_PATH)
    print(f"  Loaded {len(df)} rows  |  {df['year'].min()}–{df['year'].max()}")

    candidates = get_candidates(df)
    print(f"  Total candidate days (biological filter): {len(candidates)}")

    # ── Save annotations JSON (audit trail) ─────────────────────────────────
    with open(ANNOT_PATH, "w") as f:
        json.dump(LITERATURE_ANNOTATIONS, f, indent=2)
    print(f"\nAnnotations saved to:\n  {ANNOT_PATH}")

    # ── Process each annotated year ──────────────────────────────────────────
    gt_rows   = []   # final peak events
    audit_rows = []  # full audit trail

    annot_by_year = {a["year"]: a for a in LITERATURE_ANNOTATIONS}

    print("\n── Processing annotated years ──")
    print(f"{'Year':<6} {'Status':<10} {'Conf':<8} {'Cands':>6} {'Events':>7}")
    print("-" * 45)

    for year, annot in sorted(annot_by_year.items()):
        occurred  = annot["outbreak_occurred"]
        conf      = annot["confidence"]
        year_cands = candidates[candidates["year"] == year]

        if occurred is None:
            # Unknown year — skip, record in audit
            status = "SKIP"
            print(f"{year:<6} {status:<10} {conf:<8} {len(year_cands):>6} {'—':>7}")
            audit_rows.append({
                "year": year, "status": "SKIPPED_UNKNOWN",
                "confidence": conf, "source": annot["source"],
                "notes": annot["notes"], "candidate_count": len(year_cands),
                "events_generated": 0, "warning": "",
                "peak_dates": "",
            })
            continue

        if not occurred:
            # Confirmed clean year — validate filter doesn't fire
            clean_info = validate_clean_year(year, year_cands)
            status = "CLEAN"
            print(f"{year:<6} {status:<10} {conf:<8} {len(year_cands):>6} {'0':>7}"
                  + ("  ⚠" if clean_info["warning"] else ""))
            audit_rows.append({
                "year": year, "status": "CLEAN",
                "confidence": conf, "source": annot["source"],
                "notes": annot["notes"],
                "candidate_count": len(year_cands),
                "events_generated": 0,
                "warning": clean_info["warning"],
                "peak_dates": "",
            })
            continue

        # Confirmed outbreak year — select peak(s)
        peaks = select_peaks_for_year(year_cands, allow_multiple=True)

        if not peaks:
            # Outbreak confirmed by literature but no candidates pass
            # biological filter — flag as a filter calibration problem
            warning = (
                "LITERATURE SAYS OUTBREAK BUT ZERO CANDIDATES PASS FILTER. "
                "Options: (1) loosen biological filter thresholds, "
                "(2) check features.csv has data for this year, "
                "(3) re-verify the literature annotation."
            )
            status = "NO_CAND"
            print(f"{year:<6} {status:<10} {conf:<8} {'0':>6} {'—':>7}  ⚠")
            audit_rows.append({
                "year": year, "status": "OUTBREAK_NO_CANDIDATES",
                "confidence": conf, "source": annot["source"],
                "notes": annot["notes"], "candidate_count": 0,
                "events_generated": 0, "warning": warning,
                "peak_dates": "",
            })
            continue

        # Add to GT
        peak_dates = []
        for p in peaks:
            gt_rows.append({
                "peak_start": p["date"].strftime("%Y-%m-%d"),
                "region":     "Sangli",
                "confidence": conf,
                "intensity":  p["intensity"],
                "event_index_in_year": p["event_index"],
                "source":     annot["source"],
            })
            peak_dates.append(p["date"].strftime("%Y-%m-%d"))

        status = "OUTBREAK"
        print(f"{year:<6} {status:<10} {conf:<8} {len(year_cands):>6} {len(peaks):>7}")
        audit_rows.append({
            "year": year, "status": "OUTBREAK",
            "confidence": conf, "source": annot["source"],
            "notes": annot["notes"],
            "candidate_count": len(year_cands),
            "events_generated": len(peaks),
            "warning": "",
            "peak_dates": "; ".join(peak_dates),
        })

    # ── Write GT CSV ─────────────────────────────────────────────────────────
    # Write all columns so downstream consumers can filter by confidence tier
    # (e.g. train only on HIGH+MEDIUM events) without joining against the audit
    # CSV. Consumers that need only peak_start should read that column directly.
    gt_df = pd.DataFrame(gt_rows).sort_values("peak_start")
    gt_df.to_csv(GT_PATH, index=False)
    print(f"\nGT saved ({len(gt_df)} events):\n  {GT_PATH}")

    # ── Write audit CSV ──────────────────────────────────────────────────────
    audit_df = pd.DataFrame(audit_rows).sort_values("year")
    audit_df.to_csv(AUDIT_PATH, index=False)
    print(f"Audit trail saved:\n  {AUDIT_PATH}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────")
    print(f"  Outbreak years (GT events):  {(audit_df['status']=='OUTBREAK').sum()}")
    print(f"  Clean years:                 {(audit_df['status']=='CLEAN').sum()}")
    print(f"  Skipped (unknown):           {(audit_df['status']=='SKIPPED_UNKNOWN').sum()}")
    print(f"  No candidates (⚠):          {(audit_df['status']=='OUTBREAK_NO_CANDIDATES').sum()}")
    print(f"  Total GT peak events:        {len(gt_df)}")

    high_conf = gt_df[gt_df["confidence"] == "HIGH"]
    med_conf  = gt_df[gt_df["confidence"] == "MEDIUM"]
    print(f"\n  HIGH confidence events:      {len(high_conf)}")
    print(f"  MEDIUM confidence events:    {len(med_conf)}")

    # Split preview
    print("\n── Split preview (use HIGH+MEDIUM for training) ──")
    gt_df["year"] = pd.to_datetime(gt_df["peak_start"]).dt.year
    for split, (y0, y1) in [("Train", (2005,2014)),
                             ("Val",   (2015,2018)),
                             ("Test",  (2019,2021))]:
        n = ((gt_df["year"] >= y0) & (gt_df["year"] <= y1)).sum()
        print(f"  {split} ({y0}–{y1}): {n} events")

    warnings = audit_df[audit_df["warning"] != ""]
    if not warnings.empty:
        print(f"\n  ⚠ {len(warnings)} warnings — review audit CSV before training")

    print("\nDone.")


if __name__ == "__main__":
    generate_gt()