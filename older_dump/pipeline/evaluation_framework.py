import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def evaluate_v5_model(model_name, y_true, y_probs, dates, threshold=0.5):
    """
    Canonical Evaluation Framework for v5.
    Ensures all models are measured by the same event-based metrics.
    """
    y_pred = (y_probs >= threshold).astype(int)
    
    # 1. Standard Metrics
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_probs)
    except:
        auc = 0.5
        
    # 2. Event-Based Evaluation
    # Load Ground Truth Events
    with open("v5/labels/event_log.json", "r") as f:
        actual_events = json.load(f)
        
    # Filter actual_events to only include those that start within the dates provided
    # This ensures we don't penalize the model for events outside the test set.
    test_start = dates.min()
    test_end = dates.max()
    
    relevant_events = [
        e for e in actual_events 
        if test_start <= pd.to_datetime(e['start_date']) <= test_end
    ]
    
    total_actual = len(relevant_events)
    detected_events = set()
    false_alarms = 0
    lead_times = []
    
    for t in range(len(y_pred)):
        if y_pred[t] == 1:
            # We issued an alert on day t.
            # Was there an event starting in [t+3, t+7]?
            success = False
            alert_date = dates[t]
            target_window_start = alert_date + pd.Timedelta(days=3)
            target_window_end = alert_date + pd.Timedelta(days=7)
            
            for event in relevant_events:
                event_start_date = pd.to_datetime(event['start_date'])
                
                if target_window_start <= event_start_date <= target_window_end:
                    detected_events.add(event['event_id'])
                    lead_time = (event_start_date - alert_date).days
                    lead_times.append(lead_time)
                    success = True
                    break
            
            if not success:
                false_alarms += 1
                
    tpe = len(detected_events)
    me = total_actual - tpe
    fae = false_alarms 
    
    event_detection_rate = tpe / total_actual if total_actual > 0 else 0
    avg_lead_time = np.mean(lead_times) if lead_times else 0
    
    results = {
        "model_name": model_name,
        "standard_metrics": {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(auc)
        },
        "event_metrics": {
            "test_period": [str(test_start.date()), str(test_end.date())],
            "total_actual_events": total_actual,
            "true_positive_events": tpe,
            "missed_events": me,
            "false_alarm_count": fae,
            "event_detection_rate": float(event_detection_rate),
            "avg_lead_time": float(avg_lead_time)
        }
    }
    
    return results

def save_evaluation_report(results, output_path):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation report saved to {output_path}")

# Simplified version for use in training scripts
if __name__ == "__main__":
    print("v5 Evaluation Framework initialized.")
