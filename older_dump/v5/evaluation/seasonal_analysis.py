import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src import config

def run_seasonal_analysis():
    print("Running Seasonal Stability Analysis...")
    
    results_path = os.path.join(config.OUTPUTS_DIR, "backtest_results.csv")
    if not os.path.exists(results_path):
        raise FileNotFoundError("Run backtest_engine.py first.")
        
    df = pd.read_csv(results_path)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    
    # Categorize
    def get_season(m):
        if 1 <= m <= 5: return "Pre-Monsoon"
        if 6 <= m <= 10: return "Monsoon"
        return "Post-Monsoon"
    
    df["season"] = df["month"].apply(get_season)
    
    # Alert Density per season
    seasonal_stats = df.groupby("season").agg({
        "alert": ["count", "sum"]
    })
    seasonal_stats.columns = ["total_days", "alert_days"]
    seasonal_stats["alert_density (%)"] = (seasonal_stats["alert_days"] / seasonal_stats["total_days"]) * 100
    
    print("Seasonal Analysis Complete.")
    return seasonal_stats.to_dict(orient="index")

if __name__ == "__main__":
    print(run_seasonal_analysis())
