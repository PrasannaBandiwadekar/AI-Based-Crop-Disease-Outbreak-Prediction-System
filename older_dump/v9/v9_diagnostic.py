import os
import torch
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import sys
sys.path.append(os.path.join(os.getcwd(), "v9", "src"))
from model import V9FusionModel

# Paths
BASE_DIR = os.getcwd()
FEATURES_PATH = os.path.join(BASE_DIR, "v9", "data", "processed", "features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "v9", "models", "v9_fusion_model.pth")
GT_PATH = os.path.join(BASE_DIR, "research_comp", "evidence_base", "outbreak_events", "sangli_synthetic_gt.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "v9", "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

WEATHER_FEATURES = [
    'RH2M', 'T2M', 'T2M_MAX', 'T2M_MIN', 
    'RH2M_mean_7', 'RH2M_mean_28', 'T2M_mean_28',
    'rainfall_sum_7', 'rainfall_sum_28',
    'T2M_MIN_lag_15', 'RH2M_lag_15',
    'RH2M_diff_1', 'RH2M_accel'
]
AGRO_FEATURES = [
    'NDVI', 'NDVI_trend_7', 'variety_susceptibility', 
    'ratoon_flag', 'sanitation_score'
]

class V9DiagnosticWrapper(V9FusionModel):
    def forward(self, weather_seq, agronomic_state):
        gru_out, _ = self.gru(weather_seq)
        attn_scores = self.attention_w(gru_out)
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=1)
        weather_context = torch.sum(attn_weights * gru_out, dim=1)
        agro_context = self.agronomic_mlp(agronomic_state)
        combined = torch.cat((weather_context, agro_context), dim=1)
        logits = self.fusion_layer(combined)
        return {
            "f_out": weather_context,
            "g_out": agro_context,
            "h_out": logits,
            "risk_score": torch.sigmoid(logits) * 100
        }

def run_diagnostics():
    print("V9 FAILURE DECOMPOSITION: Starting Analysis...")
    df = pd.read_csv(FEATURES_PATH)
    df['date'] = pd.to_datetime(df['date'])
    gt_df = pd.read_csv(GT_PATH)
    gt_df['peak_start'] = pd.to_datetime(gt_df['peak_start'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diag_model = V9DiagnosticWrapper(len(WEATHER_FEATURES), len(AGRO_FEATURES)).to(device)
    diag_model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    diag_model.eval()
    
    scaler_w = StandardScaler().fit(df[WEATHER_FEATURES])
    scaler_a = StandardScaler().fit(df[AGRO_FEATURES])
    
    print("Calibrating internal layer distributions...")
    f_norms, g_norms, h_vals = [], [], []
    seq_len = 14
    df = df.sort_values('date').reset_index(drop=True)
    
    with torch.no_grad():
        for i in range(seq_len - 1, len(df), 10):
            w_t = torch.FloatTensor(scaler_w.transform(df.iloc[i-seq_len+1:i+1][WEATHER_FEATURES])).unsqueeze(0).to(device)
            a_t = torch.FloatTensor(scaler_a.transform(df.iloc[i][AGRO_FEATURES].values.reshape(1, -1))).to(device)
            out = diag_model(w_t, a_t)
            f_norms.append(torch.norm(out['f_out']).item())
            g_norms.append(torch.norm(out['g_out']).item())
            
    f_q75, f_q25 = np.percentile(f_norms, 75), np.percentile(f_norms, 25)
    g_q75, g_q25 = np.percentile(g_norms, 75), np.percentile(g_norms, 25)
    h_high, h_low = 0.0, -0.4

    def get_level(val, q25, q75):
        if val >= q75: return "high"
        if val >= q25: return "medium"
        return "low"

    fn_analysis = []
    print("Analyzing missed outbreak events...")
    for _, event in gt_df.iterrows():
        estart = event['peak_start']
        dw_start, dw_end = estart - timedelta(days=7), estart + timedelta(days=3)
        best_score, best_diag = 0, None
        
        with torch.no_grad():
            curr = dw_start
            while curr <= dw_end:
                if curr in df['date'].values:
                    idx = df[df['date'] == curr].index[0]
                    if idx >= seq_len - 1:
                        w_t = torch.FloatTensor(scaler_w.transform(df.iloc[idx-seq_len+1:idx+1][WEATHER_FEATURES])).unsqueeze(0).to(device)
                        a_t = torch.FloatTensor(scaler_a.transform(df.iloc[idx][AGRO_FEATURES].values.reshape(1, -1))).to(device)
                        out = diag_model(w_t, a_t)
                        if out['risk_score'].item() > best_score:
                            best_score = out['risk_score'].item()
                            best_diag = out
                curr += timedelta(days=1)
        
        if best_score < 40.0:
            f_level = get_level(torch.norm(best_diag['f_out']).item(), f_q25, f_q75)
            g_level = get_level(torch.norm(best_diag['g_out']).item(), g_q25, g_q75)
            h_logit = best_diag['h_out'].item()
            h_level = "high" if h_logit > h_high else ("medium" if h_logit > h_low else "low")
            
            if f_level == "low": failure = "A"
            elif g_level == "low": failure = "B"
            elif h_logit < h_low: failure = "C"
            else: failure = "D"
            
            fn_analysis.append({
                "event_start_date": estart.strftime('%Y-%m-%d'),
                "f_level": f_level, "g_level": g_level, "h_level": h_level,
                "final_risk": round(best_score, 2), "failure": failure
            })

    # Save to file
    report_path = os.path.join(OUTPUT_DIR, "sangli_failure_decomposition.txt")
    with open(report_path, "w") as f:
        f.write("V9 SYSTEM FAILURE DECOMPOSITION — SANGLI GROUND TRUTH\n")
        f.write("="*60 + "\n\n")
        f.write("1. Failure Attribution Table\n")
        f.write(f"{'event_start_date':<18} | {'f_layer':<8} | {'g_layer':<8} | {'h_layer':<8} | {'final':<6} | {'failure'}\n")
        f.write("-" * 75 + "\n")
        counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        for row in fn_analysis:
            f.write(f"{row['event_start_date']:<18} | {row['f_level']:<8} | {row['g_level']:<8} | {row['h_level']:<8} | {row['final_risk']:<6.2f} | {row['failure']}\n")
            counts[row['failure']] += 1
            
        f.write("\n2. Layer-wise Failure Counts\n")
        layer_map = {"A": "f-layer", "B": "g-layer", "C": "h-layer", "D": "calibration"}
        f.write(f"{'Layer':<15} | {'Failure Count'}\n")
        f.write("-" * 30 + "\n")
        for k in ["A", "B", "C", "D"]:
            f.write(f"{layer_map[k]:<15} | {counts[k]}\n")
            
        dominant = max(counts, key=counts.get)
        f.write(f"\n3. Dominant Bottleneck Layer: {layer_map[dominant].upper()}\n")
        f.write(f"\n4. System Diagnosis Summary\n")
        f.write(f"Primary bottleneck: {layer_map[dominant]}. Failure to generate environmental pressure alerts despite crop vulnerability.\n")

    print(f"Diagnostics complete. Report stored in {report_path}")

if __name__ == "__main__":
    run_diagnostics()
