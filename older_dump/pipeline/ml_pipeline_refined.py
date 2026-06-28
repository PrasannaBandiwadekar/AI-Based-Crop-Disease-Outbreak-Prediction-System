import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score

# Load dataset
df = pd.read_csv('early_warning_dataset.csv')

# Ensure sorted by date for rolling calculations
if 'date' in df.columns:
    df = df.sort_values('date').reset_index(drop=True)

# ---------------------------------------------------------------------
# 1. Add biologically‑meaningful weather features (rolling windows)
# ---------------------------------------------------------------------
# Wet days (precip > 1 mm) over the past 7 days
if 'PRECTOTCORR' in df.columns:
    df['wet_days_7'] = (df['PRECTOTCORR'] > 1).rolling(7).sum().fillna(0)

# High humidity days (>85%) over the past 5 days
if 'RH2M' in df.columns:
    df['high_humidity_days'] = (df['RH2M'] > 85).rolling(5).sum().fillna(0)

# Ideal temperature days (20‑30 °C) over the past 5 days
if 'T2M' in df.columns:
    df['ideal_temp_days'] = ((df['T2M'] > 20) & (df['T2M'] < 30)).rolling(5).sum().fillna(0)

# ---------------------------------------------------------------------
# 2. Define target and feature sets
# ---------------------------------------------------------------------
TARGET = 'target_5d'

# Columns that are not features (leakage / identifiers)
exclude_cols = ['date', 'target_3d', 'target_5d', 'target_7d', 'target_score_5d']
all_features = [c for c in df.columns if c not in exclude_cols + [TARGET]]

X_full = df[all_features]
y = df[TARGET]

# Time‑based split (80 % train, 20 % test)
split_idx = int(len(df) * 0.8)
X_train, X_test = X_full.iloc[:split_idx], X_full.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# ---------------------------------------------------------------------
# 3. Train full model (all engineered + new features)
# ---------------------------------------------------------------------
full_model = RandomForestClassifier(random_state=42)
full_model.fit(X_train, y_train)
full_pred = full_model.predict_proba(X_test)[:, 1]
full_auc = roc_auc_score(y_test, full_pred)
print('Full Feature ROC‑AUC (with new biologic features):', full_auc)

# ---------------------------------------------------------------------
# 4. Feature importance – identify top 15
# ---------------------------------------------------------------------
importances = pd.Series(full_model.feature_importances_, index=all_features)
top_features = importances.sort_values(ascending=False).head(15)
print('\nTop 15 features by importance:')
print(top_features)

# ---------------------------------------------------------------------
# 5. Remove suspicious seasonal signal (e.g., dayofyear)
# ---------------------------------------------------------------------
suspicious = ['dayofyear']  # you can extend this list if needed
features_no_season = [f for f in all_features if f not in suspicious]
X_no_season = df[features_no_season]
X_train_ns, X_test_ns = X_no_season.iloc[:split_idx], X_no_season.iloc[split_idx:]

model_ns = RandomForestClassifier(random_state=42)
model_ns.fit(X_train_ns, y_train)
pred_ns = model_ns.predict_proba(X_test_ns)[:, 1]
auc_ns = roc_auc_score(y_test, pred_ns)
print('\nAUC without seasonal leakage (dayofyear dropped):', auc_ns)

# ---------------------------------------------------------------------
# 6. Recall‑focused evaluation (threshold 0.4 as suggested)
# ---------------------------------------------------------------------
threshold = 0.4
y_pred_ns = (pred_ns >= threshold).astype(int)
recall = recall_score(y_test, y_pred_ns)
print('Recall at threshold 0.4 (no season):', recall)

# ---------------------------------------------------------------------
# 7. Optional: Save a cleaned dataset with only the strong, non‑seasonal features
# ---------------------------------------------------------------------
clean_features = [f for f in top_features.index if f not in suspicious]
df_clean = df[clean_features + [TARGET]]
df_clean.to_csv('feature_selected_no_season.csv', index=False)
print('\nSaved cleaned dataset (no seasonal feature) to feature_selected_no_season.csv')
