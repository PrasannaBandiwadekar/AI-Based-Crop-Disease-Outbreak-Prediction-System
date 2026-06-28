import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score

# Load data
df = pd.read_csv('early_warning_dataset.csv')
if 'date' in df.columns:
    df = df.sort_values('date').reset_index(drop=True)

TARGET = 'target_5d'
exclude_cols = ['date', 'target_3d', 'target_5d', 'target_7d', 'target_score_5d']
features = [c for c in df.columns if c not in exclude_cols + [TARGET]]

X = df[features]
y = df[TARGET]

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]

print(f"{'Threshold':<12} | {'Recall':<10} | {'Precision':<10} | {'F1-Score':<10}")
print("-" * 50)

thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for t in thresholds:
    y_pred = (proba >= t).astype(int)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"{t:<12.1f} | {rec:<10.4f} | {prec:<10.4f} | {f1:<10.4f}")
