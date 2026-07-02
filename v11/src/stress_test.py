"""
V11 KG-CTCN Synthetic Stress Testing & Region Shift Engine
--------------------------------------------------
Perturbs the raw weather dataset to test physical causality bounds.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference_engine import V11InferenceEngine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_WEATHER_PATH = os.path.join(os.path.dirname(BASE_DIR), "raw_data",
    "POWER_Point_Daily_20050101_20241231_016d85N_074d58E_LST.csv")
GT_PATH = os.path.join(os.path.dirname(BASE_DIR), "research_comp", "evidence_base", "outbreak_events", "sangli_synthetic_gt.csv")

# Set to True to print every day's risk_score during evaluation -- useful for
# sanity-checking that the model is producing sensible, non-degenerate scores
# rather than just trusting the final aggregate detection rate.
DEBUG = True

def load_base_weather():
    df = pd.read_csv(RAW_WEATHER_PATH, skiprows=14)
    df['date'] = pd.to_datetime(df['YEAR'] * 1000 + df['DOY'], format='%Y%j')
    return df.ffill().bfill()

# --- PERTURBATION GENERATORS ---

def perturb_rainfall_spikes(df):
    """Simulates extreme cloudbursts (+300% rain) on days that already had some rain."""
    mod_df = df.copy()
    np.random.seed(42)
    rain_days = mod_df['PRECTOTCORR'] > 5
    # Select 20% of rain days to become extreme spikes
    spike_mask = rain_days & (np.random.rand(len(mod_df)) < 0.2)
    mod_df.loc[spike_mask, 'PRECTOTCORR'] *= 4.0
    return mod_df

def perturb_humidity_plateau(df):
    """Simulates a broken sensor or abnormal stagnation where RH stays at 88% for 14 days randomly each year."""
    mod_df = df.copy()
    np.random.seed(42)
    for year in mod_df['date'].dt.year.unique():
        start_idx = np.random.randint(0, 350)
        year_mask = mod_df['date'].dt.year == year
        if year_mask.sum() > 0:
            idx = mod_df[year_mask].index[start_idx : start_idx + 14]
            mod_df.loc[idx, 'RH2M'] = 88.0
    return mod_df

def perturb_missing_sensors(df):
    """Randomly drops 10% of weather data and forward-fills it to simulate IoT failure."""
    mod_df = df.copy()
    np.random.seed(42)
    cols_to_drop = ['T2M', 'RH2M', 'PRECTOTCORR', 'T2M_MIN']
    mask = np.random.rand(len(mod_df)) < 0.1
    for col in cols_to_drop:
        mod_df.loc[mask, col] = np.nan
    return mod_df.ffill().bfill()

def perturb_delayed_monsoon(df):
    """
    Shifts all rainfall FORWARD by 15 days, so that the rain that historically
    fell on day D now appears on day D+15 -- mimicking a monsoon onset that
    arrives 15 days later than it actually did. The first 15 rows of the
    series have no historical predecessor to pull from, so they are filled
    with 0 (no rain yet -- monsoon hasn't started).

    NOTE: the original implementation here used `.shift(-15)`, which pulls
    FUTURE rainfall values backward into the present -- i.e. day D gets the
    rain that actually fell on day D+15. That simulates an EARLY monsoon,
    not a delayed one, and is the opposite perturbation from what the name
    and the report's narrative ("monsoon arrives late") describe. Use
    `.shift(15)` to genuinely push rainfall later in time.
    """
    mod_df = df.copy()
    mod_df['PRECTOTCORR'] = mod_df['PRECTOTCORR'].shift(15).fillna(0)
    return mod_df

# --- EVALUATION ENGINE ---

def evaluate_scenario(engine, perturbed_weather, name="Baseline", threshold=0.70):
    """
    Evaluates the model over the full ground truth using the perturbed weather.
    Uses the 2-day sustained alert rule from alert_engine conceptually.

    NOTE on `threshold`: this must match the production "High risk" cutoff
    used in inference_engine.py's run_inference() (risk_score >= 0.70 -> "High"),
    NOT an arbitrary value. The original default here was 0.9702, which does
    not correspond to anything in metadata.json, check_performance.py, or
    fp_audit.py -- it silently made every scenario return 0.0% detection,
    since risk_score is capped at 0.99 and rarely if ever sustains >0.97 for
    two consecutive days even on a true outbreak event. Use 0.70 to match
    the system's own High-risk classification, or pass in metadata.json's
    'optimal_threshold' explicitly if you want the F2-tuned operating point
    instead of the deployed business threshold (these are two different,
    intentionally distinct numbers -- see CHANGELOG discussion).

    NOTE: `engine.run_inference()` fetches weather via
    `self.weather_provider.get_weather_history(location, target_date, ...)`.
    Setting `engine.raw_weather_db` does nothing on its own -- that attribute
    is never read anywhere in inference_engine.py. To make the perturbed
    weather actually reach the model, we monkey-patch the provider's
    get_weather_history method for the duration of this scenario so it
    returns a slice of `perturbed_weather` instead of calling the live API.
    """
    location = "Sangli"  # must match whatever WeatherDataProvider.REGION_COORDS expects

    def _fake_get_weather_history(loc, target_date, window_days=400):
        end = pd.Timestamp(target_date)
        start = end - timedelta(days=window_days)
        sliced = perturbed_weather[
            (perturbed_weather['date'] > start) & (perturbed_weather['date'] <= end)
        ].copy()
        return sliced

    original_get_weather_history = engine.weather_provider.get_weather_history
    engine.weather_provider.get_weather_history = _fake_get_weather_history

    gt_df = pd.read_csv(GT_PATH)
    gt_df['peak_start'] = pd.to_datetime(gt_df['peak_start'])
    
    agro_inputs = {"variety_susceptibility": 2, "is_ratoon": 0, "crop_age_days": 150}
    
    detected = 0
    total = len(gt_df)
    lead_times = []
    
    # We also want to measure if the alert is triggered!
    # For speed, we only evaluate the 30-day window prior to each peak to check detection
    for _, row in gt_df.iterrows():
        peak = row['peak_start']
        # We need to simulate the days leading up to the peak (from peak - 10 to peak - 3)
        window_start = peak - timedelta(days=10)
        window_end = peak - timedelta(days=3)
        
        hit = False
        lead = 0
        consecutive_highs = 0
        
        for d in pd.date_range(window_start, window_end):
            try:
                inf = engine.run_inference(location, d, agro_inputs)
                score = inf['risk_score']
                if DEBUG:
                    print(f"    [{name}] {d.date()} risk_score={score:.4f} class={inf['risk_class']} "
                          f"kg_gate_open={inf['kg_gate_open']}")
                if score >= threshold:
                    consecutive_highs += 1
                else:
                    consecutive_highs = 0
                    
                # 2-day sustained alert rule
                if consecutive_highs >= 2:
                    hit = True
                    lead = (peak - d).days
                    break
            except ValueError:
                pass # not enough history
                
        if hit:
            detected += 1
            lead_times.append(lead)

    # Restore the real provider method so later scenarios / callers aren't affected
    engine.weather_provider.get_weather_history = original_get_weather_history

    edr = detected / max(1, total) * 100
    avg_lead = np.mean(lead_times) if lead_times else 0
    return edr, avg_lead

def run_stress_tests():
    print("============================================================")
    print(" V11 KG-CTCN SYNTHETIC STRESS TEST & REGION SHIFT VALIDATION")
    print("============================================================")
    
    engine = V11InferenceEngine()
    base_weather = load_base_weather()
    
    scenarios = [
        ("Baseline (Historical)", base_weather),
        ("Rainfall Spikes (+300%)", perturb_rainfall_spikes(base_weather)),
        ("Humidity Plateaus (14d)", perturb_humidity_plateau(base_weather)),
        ("Missing IoT Data (10%)", perturb_missing_sensors(base_weather)),
        ("Delayed Monsoon (Shift -15d)", perturb_delayed_monsoon(base_weather))
    ]
    
    print(f"{'Scenario Name':<30} | {'Detection Rate':<15} | {'Avg Lead Time (Days)'}")
    print("-" * 70)
    
    for name, df in scenarios:
        edr, lead = evaluate_scenario(engine, df, name=name)
        print(f"{name:<30} | {edr:>5.1f}% ({len(pd.read_csv(GT_PATH))}) | {lead:.1f}")
        
    print("============================================================")

if __name__ == "__main__":
    run_stress_tests()