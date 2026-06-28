import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# 1. LOAD DATA
# ---------------------------
print("🔍 Loading dataset...")
df = pd.read_csv("early_warning_dataset.csv")

TARGET = 'target_5d'
drop_cols = ['date', 'target_3d', 'target_5d', 'target_7d', 'target_score_5d']
initial_features = [col for col in df.columns if col not in drop_cols]

print(f"Total initial features: {len(initial_features)}")

# ---------------------------
# 2. REDUNDANCY ANALYSIS (Correlation)
# ---------------------------
print("\n🔗 Analyzing redundancy (Correlation > 0.95)...")
corr_matrix = df[initial_features].corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

print(f"Removing {len(to_drop)} redundant features: {to_drop}")
selected_features = [f for f in initial_features if f not in to_drop]

# ---------------------------
# 3. IMPORTANCE ANALYSIS (Random Forest)
# ---------------------------
print("\n📊 Analyzing predictive importance...")
X = df[selected_features]
y = df[TARGET]

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=selected_features).sort_values(ascending=False)

# Keep features that sum up to 95% of importance
cumulative_importance = importances.cumsum()
features_to_keep = cumulative_importance[cumulative_importance <= 0.96].index.tolist()

# Ensure we keep at least a few top ones if 95% is too restrictive or inclusive
if len(features_to_keep) < 5:
    features_to_keep = importances.head(10).index.tolist()

print(f"Top features representing 95% importance: {len(features_to_keep)}")
print(f"Dropped {len(selected_features) - len(features_to_keep)} low-impact features.")

# ---------------------------
# 4. SAVE OPTIMIZED DATASET
# ---------------------------
final_columns = features_to_keep + drop_cols
optimized_df = df[final_columns]

optimized_df.to_csv("optimized_early_warning_dataset.csv", index=False)

print("\n✅ Success! Optimized dataset saved as 'optimized_early_warning_dataset.csv'")
print(f"Final feature count: {len(features_to_keep)}")
print("-" * 30)
print("Final Selected Features:")
for i, feat in enumerate(features_to_keep, 1):
    print(f"{i}. {feat} ({importances[feat]:.4f})")
