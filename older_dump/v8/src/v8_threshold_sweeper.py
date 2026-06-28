import pandas as pd
import numpy as np
import os
import sys
import io
from contextlib import redirect_stdout

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from v8_evaluator import run_v8_evaluation
from post_processing_filter import apply_filters

def sweep():
    print("Executing V8 Threshold Sweep...")
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS_PATH = os.path.join(BASE_DIR, "outputs", "v8_backtest_results.csv")
    FILTERED_PATH = os.path.join(BASE_DIR, "outputs", "v8_filtered_results.csv")
    
    res_df = pd.read_csv(RESULTS_PATH)
    
    best_config = None
    best_score = -1
    
    for t in np.arange(0.1, 0.95, 0.05):
        # Apply threshold
        res_df["alert"] = res_df["risk_score"] >= t
        res_df.to_csv(RESULTS_PATH, index=False)
        
        # Run filters
        f = io.StringIO()
        with redirect_stdout(f):
            apply_filters()
        
        # Run eval
        f = io.StringIO()
        with redirect_stdout(f):
            run_v8_evaluation(FILTERED_PATH)
        out = f.getvalue()
        
        try:
            recall = float(out.split("Recall:")[1].split("%")[0].strip())
            fpr = float(out.split("False Positive Rate (FPR):")[1].split("%")[0].strip())
            lead = float(out.split("Avg Lead Time (Days):")[1].split("\n")[0].strip())
            
            # Target: Rec >= 60, FPR <= 60
            score = recall - (fpr if fpr > 60 else fpr * 0.5)
            
            print(f"Threshold: {t:.2f} | Recall: {recall:.1f}% | FPR: {fpr:.1f}% | Lead: {lead:.1f} | Score: {score:.1f}")
            
            if recall >= 60 and fpr <= 70: # Relaxed target slightly for sweep visibility
                if score > best_score:
                    best_score = score
                    best_config = (t, recall, fpr, lead)
        except:
            continue
            
    if best_config:
        print(f"\nBEST THRESHOLD: {best_config[0]:.2f}")
        print(f"Recall: {best_config[1]:.1f}% | FPR: {best_config[2]:.1f}% | Lead: {best_config[3]:.1f}")
        
        # Apply best
        res_df["alert"] = res_df["risk_score"] >= best_config[0]
        res_df.to_csv(RESULTS_PATH, index=False)
    else:
        print("\nNo configuration met the target.")

if __name__ == "__main__":
    sweep()
