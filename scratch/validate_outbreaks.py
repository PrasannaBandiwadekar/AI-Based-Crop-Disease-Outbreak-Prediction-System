"""
validate_outbreaks.py
---------------------
Validates the v5 Advisory Dataset against real-world red rot outbreak events.

Input 1: v5/data/processed/advisory_dataset_v5.csv
Input 2: research_comp/evidence_base/outbreak_events/red_rot_outbreak_events.csv

Run:
  python scratch/validate_outbreaks.py
"""

import os
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADV_CSV = os.path.join(BASE, "v5", "data", "processed", "advisory_dataset_v5.csv")
GT_CSV  = os.path.join(BASE, "research_comp", "evidence_base", "outbreak_events", "red_rot_outbreak_events.csv")

# ── Load Data ─────────────────────────────────────────────────────────────────
adv_df = pd.read_csv(ADV_CSV, parse_dates=["date"])
gt_df = pd.read_csv(GT_CSV, parse_dates=["start_date", "end_date", "peak_start", "peak_end"])

print("=" * 60)
print("  Validation Report: Red Rot Outbreak Prediction (v5)")
print("=" * 60)
print("\n" + "-" * 60)
print(" 1. Event-wise Results")
print("-" * 60)

detected_count = 0
missed_count = 0
lead_times = []
early_warning_success_count = 0
total_events = len(gt_df)

for idx, row in gt_df.iterrows():
    region = row["region"]
    start_d = row["start_date"]
    end_d = row["end_date"]
    peak_start = row["peak_start"]
    peak_end = row["peak_end"]
    conf = row["confidence"]
    
    print(f"\nRegion: Bihar - {region}")
    print(f"Period: {start_d.strftime('%Y-%m')} to {end_d.strftime('%Y-%m')}")
    print(f"Peak: {peak_start.strftime('%Y-%m-%d')} to {peak_end.strftime('%Y-%m-%d')}")
    
    # Define windows
    # Check within outbreak window (start_date to end_date) and up to 7 days before start_date
    check_start = start_d - pd.Timedelta(days=7)
    check_end = end_d
    
    mask = (adv_df["date"] >= check_start) & (adv_df["date"] <= check_end)
    event_adv = adv_df[mask].copy()
    
    # ── STEP 2: Detection Check
    high_risk_mask = (event_adv["risk_level"] == "HIGH") | (event_adv["final_risk_score"] >= 0.75)
    high_risk_df = event_adv[high_risk_mask]
    
    detected = len(high_risk_df) > 0
    first_high_date = high_risk_df["date"].min() if detected else pd.NaT
    
    # ── STEP 3: Lead Time Calculation
    if detected:
        if first_high_date < peak_start:
            lead_time = (peak_start - first_high_date).days
        else:
            # Occurred during or after peak start
            lead_time = (peak_start - first_high_date).days # Will be <= 0
    else:
        lead_time = None
        
    # ── STEP 4: Early Warning Validation
    # Must occur BEFORE peak_start (check from check_start to peak_start - 1)
    ew_mask = (event_adv["early_warning"] == 1) & (event_adv["date"] < peak_start)
    early_warning = len(event_adv[ew_mask]) > 0
    
    # ── STEP 5: Sustained Risk Check
    # >= 2 consecutive days of HIGH risk
    if detected:
        # Find consecutive days
        event_adv["is_high"] = high_risk_mask.astype(int)
        # rolling sum of 2
        event_adv["high_roll"] = event_adv["is_high"].rolling(2).sum()
        sustained_high = (event_adv["high_roll"] >= 2).any()
    else:
        sustained_high = False
        
    # Output Event Result
    print(f"Detected: {'YES' if detected else 'NO'}")
    if detected:
        print(f"First HIGH risk: {first_high_date.strftime('%Y-%m-%d')}")
        print(f"Lead time: {lead_time} days")
    else:
        print(f"First HIGH risk: NULL")
        print(f"Lead time: NULL")
        
    print(f"Early warning: {'YES' if early_warning else 'NO'}")
    print(f"Sustained HIGH: {'YES' if sustained_high else 'NO'}")
    print(f"Confidence: {conf.upper()}")
    
    # Metrics Aggregation
    if detected:
        detected_count += 1
        if lead_time is not None:
            lead_times.append(lead_time)
    else:
        missed_count += 1
        
    if early_warning:
        early_warning_success_count += 1

print("\n" + "-" * 60)
print(" 2. Summary Metrics")
print("-" * 60)

detection_rate = (detected_count / total_events) * 100 if total_events > 0 else 0
miss_rate = (missed_count / total_events) * 100 if total_events > 0 else 0
avg_lead_time = np.mean(lead_times) if lead_times else 0
ew_success_rate = (early_warning_success_count / total_events) * 100 if total_events > 0 else 0

print(f"Detection Rate: {detection_rate:.1f}% ({detected_count}/{total_events})")
print(f"Miss Rate: {miss_rate:.1f}% ({missed_count}/{total_events})")
print(f"Average Lead Time: {avg_lead_time:.1f} days")
print(f"Early Warning Success Rate: {ew_success_rate:.1f}%")

# False Positive Risk Days
# HIGH outside outbreak windows
all_high_days = set(adv_df[(adv_df["risk_level"] == "HIGH") | (adv_df["final_risk_score"] >= 0.75)]["date"])

outbreak_days = set()
for _, row in gt_df.iterrows():
    check_start = row["start_date"] - pd.Timedelta(days=7)
    check_end = row["end_date"]
    drange = pd.date_range(check_start, check_end)
    outbreak_days.update(drange)

high_outside = [d for d in all_high_days if d not in outbreak_days]
fp_rate = (len(high_outside) / len(all_high_days)) * 100 if len(all_high_days) > 0 else 0

print(f"False Positive Risk Days: {fp_rate:.1f}% ({len(high_outside)} days outside active outbreak windows)")

print("\n" + "-" * 60)
print(" 3. Seasonal Sanity Check")
print("-" * 60)

adv_df["month"] = adv_df["date"].dt.month
adv_high_df = adv_df[(adv_df["risk_level"] == "HIGH") | (adv_df["final_risk_score"] >= 0.75)]

high_in_season = adv_high_df[adv_high_df["month"].isin([7, 8, 9, 10])]
high_out_season = adv_high_df[~adv_high_df["month"].isin([7, 8, 9, 10])]

total_high = len(adv_high_df)
pct_in_season = (len(high_in_season) / total_high) * 100 if total_high > 0 else 0
pct_out_season = (len(high_out_season) / total_high) * 100 if total_high > 0 else 0

print(f"% of HIGH risk days occurring in July-October: {pct_in_season:.1f}%")
print(f"% of HIGH risk days outside monsoon: {pct_out_season:.1f}%")

print("\n" + "=" * 60)
print("  End of Report")
print("=" * 60)
