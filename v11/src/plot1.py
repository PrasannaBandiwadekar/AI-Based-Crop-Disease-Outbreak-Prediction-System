import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Connect to DB
conn = sqlite3.connect("v11/data/feedback_loop.sqlite")

# Load data
query = """
SELECT prediction_date, risk_score 
FROM predictions 
ORDER BY prediction_date
"""
df = pd.read_sql_query(query, conn)
conn.close()

# Convert date
df["prediction_date"] = pd.to_datetime(df["prediction_date"])

# -----------------------------
# 🔹 1. Smooth the signal
# -----------------------------
df["risk_smooth"] = df["risk_score"].rolling(window=5, min_periods=1).mean()

# -----------------------------
# 🔹 2. Identify high-risk events
# -----------------------------
threshold = 0.7
df["is_high"] = df["risk_score"] > threshold

# group consecutive segments
df["group"] = (df["is_high"] != df["is_high"].shift()).cumsum()

# -----------------------------
# 🔹 3. Plot
# -----------------------------
plt.figure(figsize=(14,5))

# Smoothed risk curve
plt.plot(df["prediction_date"], df["risk_smooth"], label="Risk (Smoothed)")

# Threshold
plt.axhline(y=threshold, linestyle='--', label="Threshold")

# -----------------------------
# 🔹 4. Highlight outbreak windows (>=3 days)
# -----------------------------
for _, g in df[df["is_high"]].groupby("group"):
    if len(g) >= 3:
        plt.axvspan(
            g["prediction_date"].iloc[0],
            g["prediction_date"].iloc[-1],
            alpha=0.2
        )

for _, g in df[df["is_high"]].groupby("group"):
    if len(g) >= 3:
        mid = g["prediction_date"].iloc[len(g)//2]
        plt.text(mid, 0.85, "Event", ha='center', fontsize=8)

# -----------------------------
# 🔹 5. Mark high-risk points
# -----------------------------
high = df[df["risk_score"] > threshold]
plt.scatter(high["prediction_date"], high["risk_score"], s=10, label="High Risk")

# Labels
plt.title("Event-Based Risk Dynamics for Red Rot Early Warning")
plt.xlabel("Date")
plt.ylabel("Risk Score")

plt.legend()
plt.grid()

# Save
plt.savefig("figure1_final.png", dpi=300)
plt.show()