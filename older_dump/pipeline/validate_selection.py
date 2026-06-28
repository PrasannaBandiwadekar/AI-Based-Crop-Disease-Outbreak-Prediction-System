import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# ---------------------------
# 1. LOAD OPTIMIZED DATA
# ---------------------------
print("📊 Loading optimized dataset...")
df = pd.read_csv("optimized_early_warning_dataset.csv")

TARGET = 'target_5d'
drop_cols = ['date', 'target_3d', 'target_5d', 'target_7d', 'target_score_5d']
features = [col for col in df.columns if col not in drop_cols]

print(f"Features in use ({len(features)}): {features}")

X = df[features]
y = df[TARGET]

split_index = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# ---------------------------
# 2. TRAIN OPTIMIZED MODEL
# ---------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------
# 3. EVALUATE
# ---------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n=== Optimized Model Metrics ===")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

baseline_auc = 0.8968
diff = roc_auc_score(y_test, y_prob) - baseline_auc
print(f"Difference from Baseline: {diff:+.4f}")
