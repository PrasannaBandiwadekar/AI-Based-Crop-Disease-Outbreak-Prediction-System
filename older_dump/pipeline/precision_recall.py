import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score

# Load data
df = pd.read_csv('early_warning_dataset.csv')
# Ensure sorted by date for any rolling calculations
if 'date' in df.columns:
    df = df.sort_values('date').reset_index(drop=True)

TARGET = 'target_5d'
# Columns to exclude (leakage / identifiers)
exclude_cols = ['date', 'target_3d', 'target_5d', 'target_7d', 'target_score_5d']
features = [c for c in df.columns if c not in exclude_cols + [TARGET]]

X = df[features]
y = df[TARGET]

# Time‑based split (80% train, 20% test)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Train RandomForest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]

# Metrics
roc = roc_auc_score(y_test, proba)
threshold = 0.4
pred = (proba >= threshold).astype(int)
rec = recall_score(y_test, pred)
prec = precision_score(y_test, pred)

print('ROC-AUC:', roc)
print('Recall @ threshold 0.4:', rec)
print('Precision @ threshold 0.4:', prec)
