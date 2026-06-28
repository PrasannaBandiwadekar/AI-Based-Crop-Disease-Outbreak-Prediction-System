"""
V11 KG-CTCN Deployment Layer API
--------------------------------------------------
Farmer-facing interface. Handles missing inputs, maps defaults,
generates UUIDs, creates farmer-readable explainability strings,
and logs inference for offline retraining.
"""

import os
import sys
import uuid
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference_engine import V11InferenceEngine
from feedback_db import log_prediction
from alert_engine import evaluate_alert_state

# Hardcoded agronomic defaults by region for V11 (Fallback if farmer doesn't provide)
REGION_DEFAULTS = {
    "Sangli":   {"variety_susceptibility": 2, "is_ratoon": 1, "crop_age_days": 180},
    "Kolhapur": {"variety_susceptibility": 1, "is_ratoon": 0, "crop_age_days": 150},
    "Pune":     {"variety_susceptibility": 1, "is_ratoon": 0, "crop_age_days": 120},
    "DEFAULT":  {"variety_susceptibility": 1, "is_ratoon": 0, "crop_age_days": 150},
}

class DeploymentAPI:
    def __init__(self):
        print("Initializing V11 Inference Engine...")
        from feedback_db import init_db
        init_db()
        self.engine = V11InferenceEngine()
        
    def _get_agronomic_inputs(self, location, farmer_inputs):
        """Merges farmer inputs with region defaults."""
        defaults = REGION_DEFAULTS.get(location, REGION_DEFAULTS["DEFAULT"]).copy()
        if farmer_inputs:
            # Safely override with farmer inputs if provided
            for k in defaults.keys():
                if k in farmer_inputs and farmer_inputs[k] is not None:
                    defaults[k] = farmer_inputs[k]
        return defaults

    def _generate_explanation(self, inference_output):
        """Translates technical KG signals into farmer-readable language."""
        risk_class = inference_output["risk_class"]
        weather_names = inference_output["weather_feature_names"]
        raw_weather = inference_output["raw_weather_sequence"][-1] # The latest day in the 28-d sequence
        agro = inference_output["agro_inputs"]
        
        rh_persist_idx = weather_names.index("RH_persist_7d")
        rain_sum_idx = weather_names.index("Rain_sum_7d")
        t2m_lag_idx = weather_names.index("T2M_MIN_lag_15d")
        
        rh_persist = raw_weather[rh_persist_idx]
        rain_sum = raw_weather[rain_sum_idx]
        t2m_lag = raw_weather[t2m_lag_idx]
        v_susc = agro["variety_susceptibility"]
        
        explanations = []
        
        if rh_persist >= 4:
            explanations.append("High humidity (>85%) has been sustained for most of the past week, creating an ideal environment for fungal growth.")
        if rain_sum > 10:
            explanations.append(f"Recent rainfall ({rain_sum:.1f}mm) is sufficient to trigger the spread of spores across the field.")
        if 20 <= t2m_lag <= 28:
            explanations.append("Nighttime temperatures from two weeks ago were optimal for initial fungal incubation.")
        if v_susc >= 2:
            explanations.append("Your crop variety is highly susceptible to Red Rot in these conditions.")
        elif v_susc == 0:
            explanations.append("Your resistant crop variety is currently helping to lower the overall outbreak risk.")
            
        if not explanations:
            explanations.append("Weather conditions are currently normal with no major outbreak drivers detected.")
            
        # Advisory action
        advisory = "No immediate action required. Continue regular monitoring."
        if risk_class == "High":
            advisory = "CRITICAL: Inspect fields immediately for yellowing leaves and apply preventative fungicides if symptoms appear."
        elif risk_class == "Medium":
            advisory = "WARNING: Conditions are becoming favorable for Red Rot. Ensure proper field drainage and monitor crop health."
            
        return " ".join(explanations), advisory

    def predict(self, location, target_date=None, farmer_inputs=None):
        """
        Main public method for deployment inference.
        """
        if target_date is None:
            target_date = pd.to_datetime(datetime.now().date())
        else:
            target_date = pd.to_datetime(target_date)
            
        pred_id = str(uuid.uuid4())
        
        # 1. Feature Abstraction
        agro_inputs = self._get_agronomic_inputs(location, farmer_inputs)
        
        # 2. Inference (Black-Box)
        try:
            inf_out = self.engine.run_inference(location, target_date, agro_inputs)
        except ValueError as e:
            return {"error": str(e), "prediction_id": pred_id}
            
        # 3. Explainability
        explanation_text, advisory_text = self._generate_explanation(inf_out)
        
        # 4. Alert Orchestration (Event Clustering)
        alert_result = evaluate_alert_state(location, target_date, inf_out["risk_class"])
        alert_state   = alert_result["status"]
        alert_message = alert_result["message"]
        
        # 5. Feedback Logging (Offline Retraining & Audit)
        log_prediction(
            pred_id=pred_id,
            target_date=target_date,
            location=location,
            weather_sequence=inf_out["raw_weather_sequence"],
            agro_inputs=agro_inputs,
            risk_score=inf_out["risk_score"],
            risk_class=inf_out["risk_class"],
            confidence=inf_out["confidence_score"],
            alert_status=alert_state
        )
        
        # 6. Output Construction
        status_msg = alert_message
        if inf_out.get("is_signal_saturated"):
            status_msg = "[WARNING] Low Signal Integrity: Weather inputs are significantly outside normal training ranges. Use with caution."
            explanation_text = "[SIGNAL DEGRADED] " + explanation_text

        return {
            "prediction_id": pred_id,
            "risk_score": round(inf_out["risk_score"], 4),
            "risk_class": inf_out["risk_class"],
            "confidence_score": round(inf_out["confidence_score"], 4),
            "lead_time_window": "3–7 days",
            "explanation": explanation_text,
            "advisory_action": advisory_text,
            "alert_state": alert_state,
            "alert_message": status_msg,
            "is_signal_saturated": inf_out.get("is_signal_saturated", False),
            "feedback_prompt": "Did you observe a Red Rot outbreak 3 to 7 days after this date? Please submit Yes/No via the feedback API."
        }
