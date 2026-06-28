import pandas as pd
import json
import os
from datetime import datetime

STATE_FILE = "system_state.json"
FEEDBACK_FILE = "feedback_db.csv"

def load_system_state():
    with open(STATE_FILE, 'r') as f:
        return json.load(f)

def save_system_state(state):
    state['last_updated'] = datetime.now().strftime("%Y-%m-%d")
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def calculate_reliability_metrics():
    """Aggregates feedback_db.csv to compute real-world accuracy."""
    if not os.path.exists(FEEDBACK_FILE):
        return None
        
    df = pd.read_csv(FEEDBACK_FILE)
    if len(df) < 5: # Minimal threshold for any display
        return None

    # Weighting Logic
    # YES/NO = 1.0, UNSURE = 0.5
    df['weight'] = df['feedback'].map({"YES": 1.0, "NO": 1.0, "UNSURE": 0.5})
    
    # 1. Precision (Alert Quality)
    # Correct Critical / Total Critical
    critical_alerts = df[df['alert_status'] == "CRITICAL"]
    if len(critical_alerts) > 0:
        correct_critical = (critical_alerts[critical_alerts['feedback'] == "YES"]['weight'].sum())
        total_critical = critical_alerts['weight'].sum()
        precision = (correct_critical / total_critical) if total_critical > 0 else 0
    else:
        precision = None

    # 2. Miss Rate (Recall Failure)
    # MISSED_OUTBREAKS / (Total YES + Total MISSED)
    total_yes = df[df['feedback'] == "YES"]['weight'].sum()
    missed_events = df[df['event_type'] == "MISSED_OUTBREAK"]['weight'].sum()
    total_outbreaks = total_yes + missed_events
    miss_rate = (missed_events / total_outbreaks) if total_outbreaks > 0 else 0

    # 3. Lead Time
    # Average of (symptom_date - alert_date)
    yes_with_dates = df[(df['feedback'] == "YES") & (df['symptom_date'].notna()) & (df['alert_date'].notna())]
    if len(yes_with_dates) > 0:
        yes_with_dates['symptom_date'] = pd.to_datetime(yes_with_dates['symptom_date'])
        yes_with_dates['alert_date'] = pd.to_datetime(yes_with_dates['alert_date'])
        lead_times = (yes_with_dates['symptom_date'] - yes_with_dates['alert_date']).dt.days
        avg_lead = lead_times[lead_times > 0].mean() # Filter out negative/invalid
    else:
        avg_lead = 0

    return {
        "precision": precision,
        "miss_rate": miss_rate,
        "avg_lead": avg_lead,
        "count": len(df)
    }

def get_calibration_suggestion():
    """Heuristic logic to suggest threshold shifts based on performance."""
    metrics = calculate_reliability_metrics()
    if not metrics or metrics['count'] < 10:
        return None
        
    state = load_system_state()
    # Check cooldown (3 days)
    last_cal = datetime.strptime(state['last_calibration_date'], "%Y-%m-%d")
    if (datetime.now() - last_cal).days < 3:
        return None
        
    suggestion = None
    
    # Logic A: High False Alarms (Precision < 60%)
    if metrics['precision'] is not None and metrics['precision'] < 0.60:
        new_moisture = state['thresholds']['moisture_gate_mm'] + 5
        suggestion = {
            "type": "MOISTURE_GATE",
            "current": state['thresholds']['moisture_gate_mm'],
            "proposed": new_moisture,
            "benefit": "Reduce false alerts",
            "tradeoff": "May delay early warnings slightly"
        }
        
    # Logic B: High Miss Rate (Miss Rate > 30%)
    elif metrics['miss_rate'] > 0.30:
        new_crit = max(0.1, state['thresholds']['critical'] - 0.05)
        suggestion = {
            "type": "PROBABILITY_THRESHOLD",
            "current": state['thresholds']['critical'],
            "proposed": new_crit,
            "benefit": "Detect more outbreaks",
            "tradeoff": "Will increase false alerts"
        }
        
    return suggestion

def apply_calibration(proposed_val, type):
    state = load_system_state()
    if type == "MOISTURE_GATE":
        state['thresholds']['moisture_gate_mm'] = proposed_val
    elif type == "PROBABILITY_THRESHOLD":
        state['thresholds']['critical'] = proposed_val
        
    state['last_calibration_date'] = datetime.now().strftime("%Y-%m-%d")
    save_system_state(state)
    return True
