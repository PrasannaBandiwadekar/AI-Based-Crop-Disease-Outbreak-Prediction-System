import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# ---------------------------
# 1. LOAD DATA
# ---------------------------
df = pd.read_csv("early_warning_dataset.csv")

# ---------------------------
# 2. DEFINE FEATURES + TARGET
# ---------------------------

TARGET = 'target_5d'   # predict 5-day ahead outbreak

# Remove non-feature columns
drop_cols = ['date', 'target_3d', 'target_5d', 'target_7d', 'target_score_5d']
features = [col for col in df.columns if col not in drop_cols]

X = df[features]
y = df[TARGET]

# ---------------------------
# 3. TIME-BASED SPLIT
# ---------------------------

split_index = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# ---------------------------
# 4. TRAIN MODEL
# ---------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------
# 5. EVALUATE
# ---------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# ---------------------------
# 6. FEATURE IMPORTANCE
# ---------------------------

importances = pd.Series(model.feature_importances_, index=features)
print("\nTop Features:\n", importances.sort_values(ascending=False).head(10))