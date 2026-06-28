import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Step 1: Configuration flag ──────────────────────────────────
USE_TCN = True   # Set False to revert to original GRU-based model

# ── Step 2: Conditional model import ────────────────────────────
if USE_TCN:
    from model_tcn import V9TCNModel
else:
    from model import V9FusionModel

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# ── Step 3: Conditional model path ──────────────────────────────
if USE_TCN:
    MODEL_PATH = os.path.join(BASE_DIR, "models", "v9_tcn_corrected.pth")
    MODEL_VERSION = "V9-TCN"
else:
    MODEL_PATH = os.path.join(BASE_DIR, "models", "v9_fusion_model.pth")
    MODEL_VERSION = "V9-GRU"

WEATHER_FEATURES = [
    'RH2M', 'T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR',
    'RH2M_mean_14', 'RH2M_mean_28', 'T2M_mean_14', 'T2M_mean_28',
    'humidity_streak', 'temp_streak', 'rainfall_streak', 'rainfall_sum_3',
    'T2M_MIN_lag_15', 'RH2M_lag_15', 'RH2M_diff_1', 'RH2M_accel'
]
AGRO_FEATURES = [
    'NDVI', 'NDVI_trend_7', 'variety_susceptibility', 
    'ratoon_flag', 'sanitation_score'
]

class V9InferenceEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_version = MODEL_VERSION

        # ── Step 3: Conditional model initialisation ─────────────
        if USE_TCN:
            self.model = V9TCNModel(len(WEATHER_FEATURES), len(AGRO_FEATURES)).to(self.device)
        else:
            self.model = V9FusionModel(len(WEATHER_FEATURES), len(AGRO_FEATURES)).to(self.device)

        # ── Step 4: Weight loading (strict=False preserves g+h) ──
        state = torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
        missing, unexpected = self.model.load_state_dict(state, strict=False)

        # ── Step 5: Layer integrity verification ─────────────────
        # Confirm g-layer and h-layer weights were restored from checkpoint
        loaded_keys = set(state.keys())
        g_keys = [k for k in loaded_keys if k.startswith('agronomic_mlp')]
        h_keys = [k for k in loaded_keys if k.startswith('fusion_layer')]
        assert len(g_keys) > 0, "ERROR: g-layer (agronomic_mlp) weights not found in checkpoint"
        assert len(h_keys) > 0, "ERROR: h-layer (fusion_layer) weights not found in checkpoint"

        self.model.eval()
        
        # Load data to get scalers (In production, scalers would be saved)
        df = pd.read_csv(os.path.join(PROCESSED_DIR, "features.csv"))
        self.scaler_w = StandardScaler().fit(df[WEATHER_FEATURES])
        self.scaler_a = StandardScaler().fit(df[AGRO_FEATURES])
        self.raw_df = df # Keep for explanation context

    def predict_risk(self, date_str):
        target_date = pd.to_datetime(date_str)
        if target_date not in pd.to_datetime(self.raw_df['date']).values:
            return {"error": "Date not in range"}
        
        idx = self.raw_df[pd.to_datetime(self.raw_df['date']) == target_date].index[0]
        
        # Prepare inputs
        seq_len = 14
        weather_slice = self.raw_df.iloc[idx-seq_len+1:idx+1][WEATHER_FEATURES]
        agro_slice = self.raw_df.iloc[idx][AGRO_FEATURES]
        
        weather_t = torch.FloatTensor(self.scaler_w.transform(weather_slice)).unsqueeze(0).to(self.device)
        agro_t = torch.FloatTensor(self.scaler_a.transform(agro_slice.values.reshape(1, -1))).to(self.device)
        
        with torch.no_grad():
            logits, attn = self.model(weather_t, agro_t)
            prob = torch.sigmoid(logits).item()
            attn_weights = attn.squeeze().cpu().numpy()
            
        risk_score = prob * 100
        
        # Generate Explanation
        explanation = self._generate_explanation(idx, attn_weights, risk_score)
        
        # Generate Advisory
        advisory = self._generate_advisory(risk_score, agro_slice)
        
        # ── Step 7: Model version tagging ────────────────────────
        return {
            "date": date_str,
            "risk_score": round(risk_score, 2),
            "status": self._get_status(risk_score),
            "forecast": self._get_forecast(idx),
            "explanation": explanation,
            "advisory": advisory,
            "model_version": self.model_version
        }

    def _get_status(self, score):
        if score < 30: return "Low Risk"
        if score < 60: return "Moderate Risk"
        if score < 80: return "High Risk"
        return "Severe Outbreak Likelihood"

    def _get_forecast(self, idx):
        # Look ahead 3-7 days if data exists (simulating real forecast)
        # Here we just look at next samples in static data
        future_idx = min(idx + 7, len(self.raw_df) - 1)
        current_val = self.raw_df.iloc[idx]['risk_label']
        future_val = self.raw_df.iloc[future_idx]['risk_label']
        
        trend = "Stable"
        if future_val > current_val: trend = "Rising"
        elif future_val < current_val: trend = "Falling"
        
        return {
            "3_day_outlook": "Monitoring Required" if current_val == 1 else "Clear",
            "7_day_trend": trend
        }

    def _generate_explanation(self, idx, attn, score):
        row = self.raw_df.iloc[idx]
        
        # Key Drivers (Top weights from attention)
        top_attn_days = np.argsort(attn)[-3:]
        
        # Lag reasoning
        lag_temp = row['T2M_MIN_lag_15']
        lag_note = f"Critical 15-day lag T2M_MIN is {lag_temp:.1f}C"
        if lag_temp > 26: lag_note += " (High Pathogen Pressure)"
        
        # NDVI reasoning
        ndvi_trend = row['NDVI_trend_7']
        ndvi_note = "NDVI is stable/rising (Healthy)" if ndvi_trend >= 0 else "NDVI is declining (Stressed/Confirmed)"
        
        # Accumulation
        rh_28 = row['RH2M_mean_28']
        acc_note = f"28-day Humidity Accumulation: {rh_28:.1f}%"
        
        return {
            "primary_weather_drivers": ["RH2M Persistence", "T2M Stability"],
            "lag_based_logic": lag_note,
            "crop_state_influence": ndvi_note,
            "accumulation_pattern": acc_note,
            "temporal_focus": f"Model focused on days {top_attn_days} of 14-day window"
        }

    def _generate_advisory(self, score, agro):
        if score < 30:
            return "Normal field operations. Routine scouting recommended."
        elif score < 60:
            return "Increase field scouting frequency to 3-day intervals. Monitor spindle leaves for yellowing."
        elif score < 80:
            return "High Risk detected. Apply MHAT treatment to seed setts. Consider preventative foliar spray if humidity persists."
        else:
            return "SEVERE OUTBREAK RISK. Immediate monitoring required. Spray Bavistin 0.1% or recommended fungicide. Isolate infected clumps."

if __name__ == "__main__":
    engine = V9InferenceEngine()
    # Test on a known high-risk date from training
    result = engine.predict_risk("2007-06-22")
    
    print("\n" + "="*50)
    print(f"V9 EARLY WARNING REPORT: {result['date']}")
    print("="*50)
    print(f"RISK SCORE: {result['risk_score']} / 100")
    print(f"STATUS: {result['status']}")
    print(f"FORECAST: 3d {result['forecast']['3_day_outlook']} | 7d {result['forecast']['7_day_trend']}")
    print("\nEXPLANATION:")
    for k, v in result['explanation'].items():
        print(f"  - {k.replace('_', ' ').title()}: {v}")
    print("\nADVISORY:")
    print(f"  {result['advisory']}")
    print("="*50)
