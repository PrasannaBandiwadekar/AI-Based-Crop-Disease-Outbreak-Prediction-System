import pandas as pd
import numpy as np
import joblib
import json

# ---------------------------
# 1. LOAD MODELS & DATA
# ---------------------------
m3 = joblib.load("red_rot_3d_calibrated.joblib")
m5 = joblib.load("red_rot_5d_calibrated.joblib")
m7 = joblib.load("red_rot_7d_calibrated.joblib")

# ---------------------------
# 2. LOAD STATE & THRESHOLDS
# ---------------------------
def load_dynamic_thresholds():
    try:
        with open("system_state.json", "r") as f:
            state = json.load(f)
            return (
                state['thresholds']['critical'],
                state['thresholds']['warning'],
                state['thresholds']['watch'],
                state['thresholds']['moisture_gate_mm']
            )
    except:
        return 0.30, 0.15, 0.05, 25 # Fallbacks

THRESHOLD_CRITICAL, THRESHOLD_WARNING, THRESHOLD_WATCH, MOISTURE_GATE_MM = load_dynamic_thresholds()

# ---------------------------
# 3. CAUSAL & INTERPRETATION LOGIC (Dual-Language)
# ---------------------------

# Explicit ML feature list (must match training order)
ML_FEATURE_COLS = [
    'WS10M', 'T2M', 'RH2M', 'T2M_MIN', 'T2M_MAX', 'PRECTOTCORR', 
    'temp_range', 'temp_mean_3d', 'temp_mean_7d', 'temp_mean_14d', 
    'rain_3d', 'rain_7d', 'rain_14d', 'is_rainy', 'dry_spell_7d', 
    'rh_3d', 'rh_7d', 'high_humidity', 'wind_3d', 
    'temp_lag_1', 'rain_lag_1', 'rh_lag_1', 
    'temp_lag_2', 'rain_lag_2', 'rh_lag_2', 
    'temp_lag_3', 'rain_lag_3', 'rh_lag_3', 
    'temp_lag_7', 'rain_lag_7', 'rh_lag_7', 
    'sin_day', 'cos_day', 'wet_streak', 'humid_streak', 
    'temp_optimal_red_rot', 'dry_to_wet_trigger', 'red_rot_risk_composite', 
    'fungal_risk', 'heat_stress', 'cold_stress'
]

def get_biological_interpretation(rain, rh, moisture_code):
    """Translates raw weather data into biological meanings (EN/MR)."""
    # English
    if rain > 25: rain_en, rain_mr = "Heavy accumulation", "पाण्याची मोठी साठवण"
    elif rain > 10: rain_en, rain_mr = "Moderate moisture", "मध्यम ओलावा"
    else: rain_en, rain_mr = "Low rainfall", "कमी पाऊस"
    
    if rh > 85: rh_en, rh_mr = "Sustained moisture", "जास्त आर्द्रता"
    elif rh > 75: rh_en, rh_mr = "Humid conditions", "दमट हवामान"
    else: rh_en, rh_mr = "Dry atmosphere", "कोरडे हवामान"
    
    moisture_map_en = {0: "Well-drained soil", 1: "Soil saturation rising", 2: "Standing water risk"}
    moisture_map_mr = {0: "निथळलेली जमीन", 1: "जमिनीतील ओलावा वाढत आहे", 2: "पाणी साचण्याचा धोका"}
    
    return {
        "en": {
            "rain": f"{rain:.1f}mm → {rain_en}",
            "rh": f"{rh:.1f}% → {rh_en}",
            "moisture": moisture_map_en.get(moisture_code, "Unknown")
        },
        "mr": {
            "rain": f"{rain:.1f}mm → {rain_mr}",
            "rh": f"{rh:.1f}% → {rh_mr}",
            "moisture": moisture_map_mr.get(moisture_code, "अज्ञात")
        }
    }

def get_comprehensive_advisory(df, sample_idx):
    """
    Refined logic for Live Advisory Generation.
    """
    # 1. Feature Prep
    # Data window for ML (needs 14 days)
    if sample_idx < 13: 
        return None 
        
    # SLICE AND REORDER: Key fix for ValueError Feature Mismatch
    # We take the 14-day window and strictly reorder columns to match ML_FEATURE_COLS
    window_df = df.iloc[sample_idx-13:sample_idx+1][ML_FEATURE_COLS]
    window = window_df.values.flatten().reshape(1, -1)
    
    # 2. ML Predictions
    p3 = m3.predict_proba(window)[0, 1]
    p5 = m5.predict_proba(window)[0, 1]
    p7 = m7.predict_proba(window)[0, 1]
    
    # 3. Metadata for Gating
    current = df.iloc[sample_idx]
    moisture_code = current['moisture_stress']
    is_suppressed = current['suppress_alert']
    
    # 4. Gating & Tier Logic
    raw_prob = p5
    if raw_prob >= THRESHOLD_CRITICAL and moisture_code == 2 and not is_suppressed:
        status, status_mr = "CRITICAL", "अत्यंत गंभीर"
        priority, priority_mr = "IMMEDIATE ACTION", "तातडीने कारवाई करा"
    elif raw_prob >= THRESHOLD_WARNING or moisture_code >= 1:
        status, status_mr = "WARNING", "इशारा"
        priority, priority_mr = "MONITOR CLOSELY", "बारकाईने लक्ष ठेवा"
    elif raw_prob >= THRESHOLD_WATCH:
        status, status_mr = "WATCH", "सावधान"
        priority, priority_mr = "PREVENTIVE", "प्रतिबंधात्मक उपाय"
    else:
        status, status_mr = "STABLE", "स्थिर"
        priority, priority_mr = "ROUTINE", "नियमित तपासणी"

    if is_suppressed and status != "STABLE":
        status, status_mr = "WATCH", "सावधान"
        priority, priority_mr = "SUPPRESSED (Early Growth)", "दाबले गेले (लहान पीक)"

    # 5. Causal Delta
    prev = df.iloc[max(0, sample_idx - 3)]
    causal_en = []
    if current['rain_3d'] > prev['rain_3d'] + 5:
        causal_en.append(f"Rain increased ({prev['rain_3d']:.1f}mm → {current['rain_3d']:.1f}mm)")
    if current['rh_3d'] > prev['rh_3d'] + 2:
        causal_en.append(f"Humidity rising ({prev['rh_3d']:.1f}% → {current['rh_3d']:.1f}%)")
    
    # 6. Advisory Templates
    adv_templates_en = {
        "CRITICAL": "🔴 CRITICAL: Risk in next 5 days. ACTION: Drain field immediately and apply fungicides.",
        "WARNING": "🟠 WARNING (Monitor Closely): Conditions favorable. ACTION: Inspect stalk for yellowing.",
        "WATCH": "🟡 WATCH: Low-level risk. ACTION: Continue weekly inspections.",
        "STABLE": "🟢 STABLE: No immediate threat."
    }
    adv_templates_mr = {
        "CRITICAL": "🔴 गंभीर: पुढील ५ दिवसात धोका. कृती: शेतातून जास्तीचे पाणी काढून टाका आणि बुरशीनाशकाची फवारणी करा.",
        "WARNING": "🟠 इशारा: लागण होण्यास पोषक वातावरण. कृती: पाने पिवळी पडत आहेत का ते तपासा.",
        "WATCH": "🟡 सावधान: कमी धोका. कृती: नियमित पाहणी सुरू ठेवा.",
        "STABLE": "🟢 स्थिर: कोणताही धोका नाही."
    }

    return {
        "en": {
            "status": status,
            "priority": priority,
            "interpretation": get_biological_interpretation(current['rain_3d'], current['rh_3d'], moisture_code)['en'],
            "causal": causal_en,
            "advisory": adv_templates_en.get(status, "")
        },
        "mr": {
            "status": status_mr,
            "priority": priority_mr,
            "interpretation": get_biological_interpretation(current['rain_3d'], current['rh_3d'], moisture_code)['mr'],
            "advisory": adv_templates_mr.get(status, "")
        },
        "probs": {"3d": p3, "5d": p5, "7d": p7},
        "metadata": {
            "date": str(current['date']),
            "moisture": int(moisture_code),
            "crop_age": int(current['crop_age_days']),
            "suppressed": bool(is_suppressed)
        }
    }

# ---------------------------
# 4. EXECUTION
# ---------------------------
if __name__ == "__main__":
    test_df = pd.read_csv("early_warning_dataset.csv")
    test_df['date'] = pd.to_datetime(test_df['date'])
    res = get_comprehensive_advisory(test_df, len(test_df)-1)
    import json
    print(json.dumps(res, indent=2))
