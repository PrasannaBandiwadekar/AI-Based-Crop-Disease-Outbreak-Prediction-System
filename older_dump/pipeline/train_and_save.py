import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

def train_and_save_model():
    print("Loading data...")
    df = pd.read_csv('early_warning_dataset.csv')
    
    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)

    TARGET = 'target_5d'
    exclude_cols = ['date', 'target_3d', 'target_5d', 'target_7d', 'target_score_5d']
    features = [c for c in df.columns if c not in exclude_cols + [TARGET]]

    X = df[features]
    y = df[TARGET]

    print("Training model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    # Save the model and feature list
    model_data = {
        'model': model,
        'features': features
    }
    joblib.dump(model_data, 'red_rot_model.pkl')
    print("✅ Model trained on full dataset and saved to red_rot_model.pkl")

if __name__ == '__main__':
    train_and_save_model()
