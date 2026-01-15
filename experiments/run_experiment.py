"""
Experiment Runner

Runs all experiments with LDA, PCA, and Random Projection.
Saves results to CSV for analysis in notebooks.

Tracks runtime for each major step:
- Dimensionality reduction (fit + transform)
- Classifier training
- Evaluation
"""

import os
import sys
import csv
import argparse
import time
from datetime import datetime
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.extract_features import load_features
from reduction.lda import LDAReducer
from reduction.pca import PCAReducer
from reduction.random_projection import RandomProjectionReducer
from models.linear_classifier import LinearClassifier
from sklearn.metrics import accuracy_score


# Experiment configuration
COMPONENT_VALUES = [2, 5, 10, 20, 40, 80, 99]
SEEDS = [0, 1, 2, 3, 4]  # 5 seeds for statistical robustness
METHODS = ['lda', 'pca', 'rp']


def get_reducer(method, n_components, random_state=None):
    """
    Get the appropriate reducer instance.
    
    Args:
        method: 'lda', 'pca', or 'rp'
        n_components: Number of components
        random_state: Random seed (used for RP)
    
    Returns:
        Reducer instance
    """
    if method == 'lda':
        return LDAReducer(n_components=n_components)
    elif method == 'pca':
        return PCAReducer(n_components=n_components)
    elif method == 'rp':
        return RandomProjectionReducer(n_components=n_components, random_state=random_state)
    else:
        raise ValueError(f"Unknown method: {method}")


def run_single_experiment(X_train, y_train, X_test, y_test, method, n_components, seed):
    """
    Run a single experiment with detailed timing.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        method: Reduction method
        n_components: Number of components
        seed: Random seed
    
    Returns:
        dict with accuracy and timing breakdown
    """
    # Set random seed
    np.random.seed(seed)
    
    timings = {}
    
    # --- Dimensionality Reduction: Fit ---
    reducer = get_reducer(method, n_components, random_state=seed)
    
    t_start = time.perf_counter()
    reducer.fit(X_train, y_train)
    timings['reduction_fit_sec'] = time.perf_counter() - t_start
    
    # --- Dimensionality Reduction: Transform Train ---
    t_start = time.perf_counter()
    X_train_reduced = reducer.transform(X_train)
    timings['reduction_transform_train_sec'] = time.perf_counter() - t_start
    
    # --- Dimensionality Reduction: Transform Test ---
    t_start = time.perf_counter()
    X_test_reduced = reducer.transform(X_test)
    timings['reduction_transform_test_sec'] = time.perf_counter() - t_start
    
    # --- Classifier Training ---
    classifier = LinearClassifier(random_state=seed)
    
    t_start = time.perf_counter()
    classifier.fit(X_train_reduced, y_train)
    timings['classifier_train_sec'] = time.perf_counter() - t_start
    
    # --- Evaluation ---
    t_start = time.perf_counter()
    y_pred = classifier.predict(X_test_reduced)
    accuracy = accuracy_score(y_test, y_pred)
    timings['evaluation_sec'] = time.perf_counter() - t_start
    
    # Total runtime
    timings['total_runtime_sec'] = (
        timings['reduction_fit_sec'] +
        timings['reduction_transform_train_sec'] +
        timings['reduction_transform_test_sec'] +
        timings['classifier_train_sec'] +
        timings['evaluation_sec']
    )
    
    return {
        'accuracy': accuracy,
        **timings
    }


def run_all_experiments(feature_dir, results_file, verbose=True):
    """
    Run all experiments and save results with timing.
    
    Args:
        feature_dir: Directory with extracted features
        results_file: Path to save results CSV
        verbose: Print progress
    """
    # Load features
    if verbose:
        print("Loading features...")
    
    t_start = time.perf_counter()
    X_train, X_test, y_train, y_test = load_features(feature_dir)
    load_time = time.perf_counter() - t_start
    
    if verbose:
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  Classes: {len(np.unique(y_train))}")
        print(f"  Load time: {load_time:.2f}s")
    
    # Prepare results file
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    # CSV columns with timing breakdown
    csv_columns = [
        'method', 'components', 'seed', 'accuracy',
        'reduction_fit_sec', 'reduction_transform_train_sec', 
        'reduction_transform_test_sec', 'classifier_train_sec',
        'evaluation_sec', 'total_runtime_sec'
    ]
    
    # Write header
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_columns)
    
    # Run experiments
    total_experiments = len(METHODS) * len(COMPONENT_VALUES) * len(SEEDS)
    experiment_count = 0
    
    results = []
    
    for method in METHODS:
        for n_components in COMPONENT_VALUES:
            for seed in SEEDS:
                experiment_count += 1
                
                if verbose:
                    print(f"\n[{experiment_count}/{total_experiments}] "
                          f"Method: {method.upper()}, Components: {n_components}, Seed: {seed}")
                
                try:
                    result = run_single_experiment(
                        X_train, y_train, X_test, y_test,
                        method, n_components, seed
                    )
                    
                    if verbose:
                        print(f"  Accuracy: {result['accuracy']:.4f}")
                        print(f"  Runtime: {result['total_runtime_sec']:.3f}s "
                              f"(fit: {result['reduction_fit_sec']:.3f}s, "
                              f"train: {result['classifier_train_sec']:.3f}s)")
                    
                    # Build row for CSV
                    row = [
                        method, n_components, seed, result['accuracy'],
                        result['reduction_fit_sec'],
                        result['reduction_transform_train_sec'],
                        result['reduction_transform_test_sec'],
                        result['classifier_train_sec'],
                        result['evaluation_sec'],
                        result['total_runtime_sec']
                    ]
                    
                    # Save result immediately (don't lose data if interrupted)
                    with open(results_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(row)
                    
                    results.append({
                        'method': method,
                        'components': n_components,
                        'seed': seed,
                        **result
                    })
                    
                except Exception as e:
                    print(f"  ERROR: {e}")
                    # Still log failed experiments
                    error_row = [method, n_components, seed, 'ERROR'] + ['ERROR'] * 6
                    with open(results_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(error_row)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiments complete!")
        print(f"Results saved to: {results_file}")
        print(f"Total experiments: {len(results)}")
    
    return results


def print_summary(results_file):
    """
    Print summary statistics from results file including timing.
    """
    import pandas as pd
    
    df = pd.read_csv(results_file)
    
    # Filter out errors
    error_mask = df['accuracy'] == 'ERROR'
    if error_mask.any():
        print(f"\nWARNING: {error_mask.sum()} failed experiments excluded")
        df = df[~error_mask]
    
    # Convert types
    df['accuracy'] = df['accuracy'].astype(float)
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Accuracy summary
    accuracy_summary = df.groupby(['method', 'components'])['accuracy'].agg(['mean', 'std'])
    accuracy_summary.columns = ['mean_accuracy', 'std_accuracy']
    accuracy_summary = accuracy_summary.reset_index()
    
    for method in METHODS:
        print(f"\n{method.upper()} - Accuracy:")
        method_data = accuracy_summary[accuracy_summary['method'] == method]
        for _, row in method_data.iterrows():
            print(f"  {int(row['components']):3d} components: "
                  f"{row['mean_accuracy']:.4f} ± {row['std_accuracy']:.4f}")
    
    # Timing summary
    print("\n" + "-"*60)
    print("RUNTIME SUMMARY (seconds)")
    print("-"*60)
    
    timing_summary = df.groupby(['method', 'components']).agg({
        'reduction_fit_sec': 'mean',
        'classifier_train_sec': 'mean',
        'total_runtime_sec': 'mean'
    }).reset_index()
    
    for method in METHODS:
        print(f"\n{method.upper()} - Mean Runtime:")
        method_data = timing_summary[timing_summary['method'] == method]
        for _, row in method_data.iterrows():
            print(f"  {int(row['components']):3d} components: "
                  f"total={row['total_runtime_sec']:.3f}s "
                  f"(fit={row['reduction_fit_sec']:.3f}s, "
                  f"clf={row['classifier_train_sec']:.3f}s)")


def main():
    parser = argparse.ArgumentParser(description='Run LDA/PCA/RP experiments')
    parser.add_argument('--feature-dir', type=str, default='./features/saved',
                        help='Directory with extracted features')
    parser.add_argument('--results-file', type=str, default='./results/results.csv',
                        help='Path to save results CSV')
    parser.add_argument('--summary-only', action='store_true',
                        help='Only print summary of existing results')
    
    args = parser.parse_args()
    
    if args.summary_only:
        print_summary(args.results_file)
    else:
        # Check if features exist
        if not os.path.exists(os.path.join(args.feature_dir, 'X_train.npy')):
            print("ERROR: Features not found!")
            print(f"Please run feature extraction first:")
            print(f"  python features/extract_features.py")
            sys.exit(1)
        
        # Run experiments
        run_all_experiments(args.feature_dir, args.results_file)
        
        # Print summary
        print_summary(args.results_file)


if __name__ == "__main__":
    main()
