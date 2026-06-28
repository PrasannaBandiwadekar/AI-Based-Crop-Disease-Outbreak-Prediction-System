import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Load the early warning dataset (engineered features + targets)
df = pd.read_csv('early_warning_dataset.csv')

# Define target column
TARGET = 'target_5d'

# ---------------------------
# Step 1: Train a TRUE Baseline Model (core features only)
# ---------------------------
# Core features present in the dataset
baseline_features = ['T2M', 'PRECTOTCORR', 'RH2M']
# Ensure these columns exist
baseline_features = [col for col in baseline_features if col in df.columns]
X_baseline = df[baseline_features]
y = df[TARGET]

# Time-based split (80% train, 20% test)
split_idx = int(len(df) * 0.8)
Xb_train, Xb_test = X_baseline.iloc[:split_idx], X_baseline.iloc[split_idx:]
yb_train, yb_test = y.iloc[:split_idx], y.iloc[split_idx:]

baseline_model = RandomForestClassifier(random_state=42)
baseline_model.fit(Xb_train, yb_train)
baseline_pred = baseline_model.predict_proba(Xb_test)[:, 1]
baseline_auc = roc_auc_score(yb_test, baseline_pred)
print('Baseline ROC-AUC:', baseline_auc)

# ---------------------------
# Step 2: Evaluate Current Engineered Feature Set
# ---------------------------
# All feature columns except target/leakage columns
exclude_cols = ['date', 'target_3d', 'target_5d', 'target_7d', 'target_score_5d']
all_features = [col for col in df.columns if col not in exclude_cols]
X_full = df[all_features]
Xf_train, Xf_test = X_full.iloc[:split_idx], X_full.iloc[split_idx:]
yf_train, yf_test = y.iloc[:split_idx], y.iloc[split_idx:]

full_model = RandomForestClassifier(random_state=42)
full_model.fit(Xf_train, yf_train)
full_pred = full_model.predict_proba(Xf_test)[:, 1]
full_auc = roc_auc_score(yf_test, full_pred)
print('Full Feature ROC-AUC:', full_auc)

# ---------------------------
# Step 3: Compare Baseline vs Full
# ---------------------------
margin = 0.05
if full_auc > baseline_auc + margin:
    print(f'✅ Feature engineering adds value (ΔAUC = {full_auc - baseline_auc:.3f})')
else:
    print(f'⚠️ Feature set may be noisy (ΔAUC = {full_auc - baseline_auc:.3f})')

# ---------------------------
# Step 4: Feature Importance Check
# ---------------------------
importances = pd.Series(full_model.feature_importances_, index=all_features)
top_features = importances.sort_values(ascending=False).head(15)
print('\nTop 15 Features by importance:')
print(top_features)

# ---------------------------
# Step 5: Create CLEAN Feature Dataset
# ---------------------------
selected_features = top_features.index.tolist()
df_clean = df[selected_features + [TARGET]]
df_clean.to_csv('feature_selected_dataset.csv', index=False)
print('\nSaved cleaned dataset to feature_selected_dataset.csv')

# ---------------------------
# Step 6: Retrain Final Model on Cleaned Features
# ---------------------------
X_clean = df_clean[selected_features]
y_clean = df_clean[TARGET]
Xc_train, Xc_test = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
yc_train, yc_test = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]

final_model = RandomForestClassifier(random_state=42)
final_model.fit(Xc_train, yc_train)
final_pred = final_model.predict_proba(Xc_test)[:, 1]
final_auc = roc_auc_score(yc_test, final_pred)
print('\nFinal ROC-AUC on cleaned feature set:', final_auc)

# Optional: focus on recall (early outbreak detection)
# You could re‑train with class_weight='balanced' or adjust threshold later.
