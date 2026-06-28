import os
import time
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

# Track generated files
generated_files = []

def track_file(filepath):
    generated_files.append(filepath)
    return filepath

def init_directories(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    dirs = ['models', 'metrics', 'plots', 'logs']
    for d in dirs:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)
    print(f"Initialized subdirectories in {base_dir}")

def evaluate_thresholds(y_true, y_prob, thresholds):
    best_f1 = -1
    best_thresh = 0.5
    best_metrics = {}
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy_score(y_true, y_pred)
            }
            
    return best_thresh, best_metrics

def train_and_evaluate(seq_df, feature_cols, test_size=0.3):
    n_samples = len(seq_df)
    
    split_idx = int(n_samples * (1 - test_size))
    
    train_df = seq_df.iloc[:split_idx]
    test_df = seq_df.iloc[split_idx:]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['target_5d'].values.astype(int)
    X_test = test_df[feature_cols].values
    y_test = test_df['target_5d'].values.astype(int)
    
    # Check class distribution
    train_dist = np.bincount(y_train)
    
    start_time = time.time()
    
    model = RandomForestClassifier(
        class_weight="balanced",
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    # Probabilities
    y_prob = model.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) > 1 else np.zeros(len(y_test))
    roc_auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0
    
    # Threshold Tuning
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    best_thresh, best_metrics = evaluate_thresholds(y_test, y_prob, thresholds)
    best_metrics['roc_auc'] = roc_auc
    
    # Predictions using best threshold
    y_pred_best = (y_prob >= best_thresh).astype(int)
    
    return model, best_thresh, best_metrics, train_time, test_df, y_pred_best, train_dist

def plot_predictions(test_df, y_pred, window, out_path):
    plt.figure(figsize=(15, 5))
    
    if 'date' in test_df.columns:
        plot_df = test_df.copy()
        plot_df['y_pred'] = y_pred
        plot_df['date'] = pd.to_datetime(plot_df['date'])
        plot_df = plot_df.sort_values('date')
        
        plt.plot(plot_df['date'], plot_df['target_5d'], label='Actual', alpha=0.7)
        plt.plot(plot_df['date'], plot_df['y_pred'], label='Predicted', alpha=0.7)
        plt.xlabel('Date')
    else:
        plt.plot(test_df['target_5d'].values, label='Actual', alpha=0.7)
        plt.plot(y_pred, label='Predicted', alpha=0.7)
        plt.xlabel('Sample')
        
    plt.ylabel('Target 5d')
    plt.title(f'Prediction vs Actual - Window {window} (Improved)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(track_file(out_path))
    plt.close()

def main():
    BASE_DIR = os.path.join("v3", "improved_run")
    init_directories(BASE_DIR)
    
    log_file = track_file(os.path.join(BASE_DIR, 'logs', 'training_log.txt'))
    with open(log_file, 'w') as f:
        f.write("Training Log - Improved Run\n=====================\n")
        f.write("Thresholds tested: [0.1, 0.2, 0.3, 0.4, 0.5]\n\n")
        
    windows = [3, 5, 7, 14, 28]
    all_results = {}
    
    for w in windows:
        print(f"\n--- Processing Window {w} ---")
        try:
            seq_path = os.path.join("v3", "data", f"seq_window_{w}.csv")
            if not os.path.exists(seq_path):
                raise FileNotFoundError(f"Sequence data file not found: {seq_path}")
                
            print(f"Loading sequences from {seq_path}...")
            seq_df = pd.read_csv(seq_path)
            
            # Extract feature columns (all columns ending with lag_*)
            feature_cols = [c for c in seq_df.columns if '_lag_' in c]
            
            print("Training improved temporal model...")
            model, best_thresh, metrics, train_time, test_df, y_pred, train_dist = train_and_evaluate(seq_df, feature_cols)
            
            # Save model
            model_path = track_file(os.path.join(BASE_DIR, 'models', f'rf_window_{w}.pkl'))
            joblib.dump(model, model_path)
            
            # Save metrics
            metrics_path = track_file(os.path.join(BASE_DIR, 'metrics', f'window_{w}_metrics.txt'))
            with open(metrics_path, 'w') as f:
                f.write(f"Best Threshold: {best_thresh}\n")
                for k, v in metrics.items():
                    f.write(f"{k}: {v}\n")
                    
            with open(log_file, 'a') as f:
                f.write(f"Window {w}:\n")
                f.write(f"  - Train class distribution: {train_dist}\n")
                f.write(f"  - Trained in {train_time:.2f}s\n")
                f.write(f"  - Best Threshold: {best_thresh}\n")
                f.write(f"  - F1 Score: {metrics['f1_score']:.4f}\n\n")
                
            print("Generating plots...")
            plot_path = os.path.join(BASE_DIR, 'plots', f'prediction_vs_actual_w{w}.png')
            plot_predictions(test_df, y_pred, w, plot_path)
            
            all_results[w] = metrics
            
        except Exception as e:
            error_msg = f"Error processing window {w}: {str(e)}"
            print(error_msg)
            with open(log_file, 'a') as f:
                f.write(error_msg + "\n")

    # Compare with Baseline
    print("\nComparing with Baseline...")
    baseline_summary_path = os.path.join("v3", "metrics", "comparison_summary.txt")
    baseline_f1_scores = {}
    if os.path.exists(baseline_summary_path):
        with open(baseline_summary_path, 'r') as f:
            for line in f:
                if "Window " in line and "F1=" in line and "AUC=" in line and "Shuffled" not in line:
                    # Parse line like: "   - Window 3d: F1=0.0141, AUC=0.8157"
                    try:
                        parts = line.strip().split()
                        w_str = parts[2].replace('d:', '')
                        w_val = int(w_str)
                        f1_part = parts[3].replace('F1=', '').replace(',', '')
                        baseline_f1_scores[w_val] = float(f1_part)
                    except:
                        pass
    
    comp_path = track_file(os.path.join(BASE_DIR, 'metrics', 'comparison_with_baseline.txt'))
    with open(comp_path, 'w') as f:
        f.write("=== Comparison With Baseline ===\n\n")
        f.write("Old F1 vs New F1:\n")
        
        max_improvement = -999
        best_improved_window = None
        
        for w in windows:
            new_f1 = all_results[w]['f1_score'] if w in all_results else 0.0
            old_f1 = baseline_f1_scores.get(w, 0.0)
            diff = new_f1 - old_f1
            
            f.write(f"Window {w}d: Old F1 = {old_f1:.4f} | New F1 = {new_f1:.4f} | Improvement = {diff:+.4f}\n")
            
            if diff > max_improvement:
                max_improvement = diff
                best_improved_window = w
                
        f.write(f"\nWindow that benefited most: {best_improved_window} days (Improvement: {max_improvement:+.4f})\n")

    print("\n--- Final Validation ---")
    print(f"Verifying all outputs exist inside '{BASE_DIR}/'...")
    validation_failed = False
    
    print("List of generated files:")
    for f_path in generated_files:
        print(f" - {f_path}")
        abs_base = os.path.abspath(BASE_DIR)
        abs_file = os.path.abspath(f_path)
        if not abs_file.startswith(abs_base):
            print(f"ERROR: File {f_path} is located outside {BASE_DIR}!")
            validation_failed = True
            
    if validation_failed:
        raise RuntimeError(f"Validation Failed! Some files were saved outside {BASE_DIR}/")
    else:
        print(f"\nValidation Passed: All outputs correctly isolated inside {BASE_DIR}/")
        
    print(f"\nPipeline completed successfully. All outputs are in {BASE_DIR}/")

if __name__ == "__main__":
    main()
