import os
import sys
import pandas as pd

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from v8_evaluator import run_v8_evaluation
from post_processing_filter import apply_filters

def run_upgrade():
    print("========================================")
    print("V8 UPGRADE: TWO-STAGE EARLY WARNING SYSTEM")
    print("========================================\n")
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_RESULTS = os.path.join(BASE_DIR, "outputs", "v8_backtest_results.csv")
    FILTERED_RESULTS = os.path.join(BASE_DIR, "outputs", "v8_filtered_results.csv")
    
    # 1. Eval BEFORE Filtering
    print("--- STAGE 1: RAW PREDICTIONS EVALUATION ---")
    run_v8_evaluation(RAW_RESULTS)
    
    # 2. Apply Stage-2 Filters
    print("\n--- STAGE 2: APPLYING POST-PROCESSING FILTERS ---")
    apply_filters()
    
    # 3. Eval AFTER Filtering
    print("\n--- STAGE 3: FILTERED PREDICTIONS EVALUATION ---")
    run_v8_evaluation(FILTERED_RESULTS)
    
    print("\n========================================")
    print("V8 UPGRADE COMPLETE")
    print("========================================")

if __name__ == "__main__":
    run_upgrade()
