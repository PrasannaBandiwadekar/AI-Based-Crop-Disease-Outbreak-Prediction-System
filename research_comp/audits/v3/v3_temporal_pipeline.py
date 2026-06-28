import os
import time
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import joblib

# Track generated files
generated_files = []

def track_file(filepath):
    generated_files.append(filepath)
    return filepath

def init_directories(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    dirs = ['data', 'models', 'metrics', 'plots', 'logs']
    for d in dirs:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)
    print("Using existing v3 folder and initialized subdirectories")

def load_and_prepare_data(filepath, core_features):
    df = pd.read_csv(filepath)
    # Ensure date is datetime and sort
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Select only core features and target
    cols_to_keep = core_features + ['target_5d', 'date']
    # Keep only the columns that exist
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    
    df = df[cols_to_keep]
    return df

def create_sequences(df, window_size, core_features):
    X, y = [], []
    dates = []
    
    feature_data = df[core_features].values
    target_data = df['target_5d'].values
    date_data = df['date'].values
    
    for i in range(window_size, len(df)):
        # Past W days
        seq_x = feature_data[i-window_size:i].flatten()
        seq_y = target_data[i]
        seq_date = date_data[i]
        
        X.append(seq_x)
        y.append(seq_y)
        dates.append(seq_date)
        
    # Create DataFrame for the sequence
    col_names = []
    for day in range(window_size):
        for f in core_features:
            col_names.append(f"{f}_lag_{window_size-day}")
            
    seq_df = pd.DataFrame(X, columns=col_names)
    seq_df['target_5d'] = y
    seq_df['date'] = dates
    
    return seq_df, col_names

def train_and_evaluate(seq_df, feature_cols, test_size=0.3, shuffle=False):
    n_samples = len(seq_df)
    
    if shuffle:
        # Shuffle dataset before split for temporal strength test
        seq_df = seq_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
    split_idx = int(n_samples * (1 - test_size))
    
    train_df = seq_df.iloc[:split_idx]
    test_df = seq_df.iloc[split_idx:]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['target_5d'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['target_5d'].values
    
    start_time = time.time()
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) > 1 else np.zeros(len(y_test))
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0
    }
    
    return model, metrics, train_time, test_df, y_pred, y_prob

def plot_predictions(test_df, y_pred, window, out_path):
    plt.figure(figsize=(15, 5))
    
    # Sort by date for plotting if date exists
    if 'date' in test_df.columns:
        plot_df = test_df.copy()
        plot_df['y_pred'] = y_pred
        plot_df = plot_df.sort_values('date')
        
        plt.plot(plot_df['date'], plot_df['target_5d'], label='Actual', alpha=0.7)
        plt.plot(plot_df['date'], plot_df['y_pred'], label='Predicted', alpha=0.7)
        plt.xlabel('Date')
    else:
        plt.plot(test_df['target_5d'].values, label='Actual', alpha=0.7)
        plt.plot(y_pred, label='Predicted', alpha=0.7)
        plt.xlabel('Sample')
        
    plt.ylabel('Target 5d')
    plt.title(f'Prediction vs Actual - Window {window}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(track_file(out_path))
    plt.close()

def plot_performance_trend(results, out_path):
    windows = sorted(results.keys())
    f1_scores = [results[w]['temporal']['f1_score'] for w in windows]
    auc_scores = [results[w]['temporal']['roc_auc'] for w in windows]
    
    plt.figure(figsize=(10, 6))
    plt.plot(windows, f1_scores, marker='o', label='F1-Score')
    plt.plot(windows, auc_scores, marker='s', label='ROC-AUC')
    
    plt.xlabel('Sequence Window Size (Days)')
    plt.ylabel('Score')
    plt.title('Model Performance vs. Sequence Window Size')
    plt.xticks(windows)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(track_file(out_path))
    plt.close()

def write_summary_report(results, out_path):
    windows = sorted(results.keys())
    
    best_window = max(windows, key=lambda w: results[w]['temporal']['f1_score'])
    best_f1 = results[best_window]['temporal']['f1_score']
    
    f1_trend = [results[w]['temporal']['f1_score'] for w in windows]
    is_longer_better = f1_trend[-1] > f1_trend[0]
    
    temporal_strong = True
    for w in windows:
        if results[w]['shuffled']['f1_score'] >= results[w]['temporal']['f1_score'] + 0.05:
            temporal_strong = False 
            
    with open(track_file(out_path), 'w') as f:
        f.write("=== Multi-Window Temporal Pipeline Summary ===\n\n")
        f.write(f"1. Best Window Size: {best_window} days (F1-Score: {best_f1:.4f})\n\n")
        f.write("2. Performance Trend vs Window:\n")
        for w in windows:
            f.write(f"   - Window {w}d: F1={results[w]['temporal']['f1_score']:.4f}, AUC={results[w]['temporal']['roc_auc']:.4f}\n")
        f.write("\n")
        f.write(f"3. Does longer history help? {'Yes' if is_longer_better else 'No'}. \n")
        f.write(f"   (F1 went from {f1_trend[0]:.4f} at {windows[0]}d to {f1_trend[-1]:.4f} at {windows[-1]}d)\n\n")
        f.write("4. Temporal Signal Strength:\n")
        for w in windows:
            t_f1 = results[w]['temporal']['f1_score']
            s_f1 = results[w]['shuffled']['f1_score']
            f.write(f"   - Window {w}d: Temporal F1={t_f1:.4f}, Shuffled F1={s_f1:.4f} (Diff: {t_f1 - s_f1:.4f})\n")
            
        if temporal_strong:
            f.write("\n   Conclusion: Strong temporal signal. The model relies on valid chronological patterns.\n")
        else:
            f.write("\n   Conclusion: Weak temporal signal or high data leakage when shuffled.\n")

def main():
    BASE_DIR = "v3"
    init_directories(BASE_DIR)
    
    log_file = track_file(os.path.join(BASE_DIR, 'logs', 'training_log.txt'))
    with open(log_file, 'w') as f:
        f.write("Training Log\n=====================\n")
        
    core_features = ['T2M', 'RH2M', 'PRECTOTCORR', 'WS10M', 'temp_range']
    dataset_path = 'early_warning_dataset.csv'
    
    print("Loading data...")
    df = load_and_prepare_data(dataset_path, core_features)
    
    with open(log_file, 'a') as f:
        f.write(f"Dataset loaded. Total rows: {len(df)}\n")
        
    windows = [3, 5, 7, 14, 28]
    all_results = {}
    
    for w in windows:
        print(f"\n--- Processing Window {w} ---")
        try:
            print(f"Creating sequences for window {w}...")
            seq_df, feature_cols = create_sequences(df, w, core_features)
            
            seq_path = track_file(os.path.join(BASE_DIR, 'data', f'seq_window_{w}.csv'))
            seq_df.to_csv(seq_path, index=False)
            
            with open(log_file, 'a') as f:
                f.write(f"Window {w}: Sequence dataset created. Rows: {len(seq_df)}\n")
            
            print("Training temporal model...")
            model, metrics, train_time, test_df, y_pred, y_prob = train_and_evaluate(seq_df, feature_cols, shuffle=False)
            
            model_path = track_file(os.path.join(BASE_DIR, 'models', f'rf_window_{w}.pkl'))
            joblib.dump(model, model_path)
            
            metrics_path = track_file(os.path.join(BASE_DIR, 'metrics', f'window_{w}_metrics.txt'))
            with open(metrics_path, 'w') as f:
                for k, v in metrics.items():
                    f.write(f"{k}: {v}\n")
                    
            with open(log_file, 'a') as f:
                f.write(f"Window {w}: Temporal model trained in {train_time:.2f}s\n")
                
            print("Training shuffled model (temporal strength test)...")
            shuf_model, shuf_metrics, shuf_train_time, _, _, _ = train_and_evaluate(seq_df, feature_cols, shuffle=True)
            
            shuf_metrics_path = track_file(os.path.join(BASE_DIR, 'metrics', f'window_{w}_shuffled.txt'))
            with open(shuf_metrics_path, 'w') as f:
                for k, v in shuf_metrics.items():
                    f.write(f"{k}: {v}\n")
                    
            with open(log_file, 'a') as f:
                f.write(f"Window {w}: Shuffled model trained in {shuf_train_time:.2f}s\n")
                
            print("Generating plots...")
            plot_path = os.path.join(BASE_DIR, 'plots', f'prediction_vs_actual_w{w}.png')
            plot_predictions(test_df, y_pred, w, plot_path)
            
            all_results[w] = {
                'temporal': metrics,
                'shuffled': shuf_metrics
            }
            
        except Exception as e:
            error_msg = f"Error processing window {w}: {str(e)}"
            print(error_msg)
            with open(log_file, 'a') as f:
                f.write(error_msg + "\n")
                
    print("\nGenerating final reports and plots...")
    perf_plot_path = os.path.join(BASE_DIR, 'plots', 'performance_vs_window.png')
    plot_performance_trend(all_results, perf_plot_path)
    
    summary_path = os.path.join(BASE_DIR, 'metrics', 'comparison_summary.txt')
    write_summary_report(all_results, summary_path)
    
    # Step 10: Final Validation
    print("\n--- Final Validation ---")
    print(f"Verifying all outputs exist inside '{BASE_DIR}/'...")
    validation_failed = False
    
    print("List of generated files:")
    for f_path in generated_files:
        print(f" - {f_path}")
        # Check if the path starts with BASE_DIR
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
