#!/usr/bin/env python
"""
V6 Main Entry Point

Usage:
    python main.py train     — run full training pipeline
    python main.py labels    — inspect label generation only
    python main.py sequences — inspect sequence builder only
    python main.py evaluate  — run frozen evaluator on v6 outputs
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def cmd_train():
    from train import run_v6_training
    run_v6_training()


def cmd_labels():
    import pandas as pd
    from utils import load_raw_data
    from label_engine import load_outbreak_events, build_onset_labels, validate_label_alignment

    raw_df = load_raw_data()
    raw_df["date"] = pd.to_datetime(raw_df["date"])
    peaks = load_outbreak_events()
    labeled = build_onset_labels(raw_df, peaks)
    validate_label_alignment(labeled, peaks)

    from config import TARGET
    positives = labeled[labeled[TARGET] == 1]
    print(f"\nPositive label sample:\n{positives[['date', TARGET]].to_string()}")


def cmd_sequences():
    import pandas as pd
    from utils import load_raw_data
    from label_engine import load_outbreak_events, build_onset_labels
    from sequence_builder import build_daily_features, build_sequences, chronological_split

    raw_df = load_raw_data()
    raw_df["date"] = pd.to_datetime(raw_df["date"])
    peaks = load_outbreak_events()
    labeled = build_onset_labels(raw_df, peaks)
    features = build_daily_features(raw_df)
    X, y, dates = build_sequences(features, labeled)
    chronological_split(X, y, dates)
    print(f"\nX shape: {X.shape}, y shape: {y.shape}")


def cmd_evaluate():+
    """Run the frozen V5 evaluator against V6 backtest results."""
    import subprocess
    import shutil
    from config import OUTPUTS_DIR

    v5_eval_path = os.path.join(
        os.path.dirname(__file__), "..", "v5", "evaluation", "final_v5_evaluator.py"
    )
    v6_results = os.path.join(OUTPUTS_DIR, "v6_backtest_results.csv")
    v5_outputs = os.path.join(
        os.path.dirname(__file__), "..", "v5", "outputs", "backtest_results.csv"
    )

    if not os.path.exists(v6_results):
        print("ERROR: v6_backtest_results.csv not found. Run 'train' first.")
        return

    # Point frozen evaluator to V6 results by copying
    shutil.copy(v6_results, v5_outputs)
    print(f"Copied V6 results → {v5_outputs}")
    print("Running frozen evaluator...")

    subprocess.run([sys.executable, v5_eval_path], cwd=os.path.dirname(v5_eval_path))


if __name__ == "__main__":
    commands = {
        "train":     cmd_train,
        "labels":    cmd_labels,
        "sequences": cmd_sequences,
        "evaluate":  cmd_evaluate,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(f"Usage: python main.py [{' | '.join(commands.keys())}]")
        sys.exit(1)

    commands[sys.argv[1]]()
