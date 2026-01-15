"""
Ablation Study: Feature Projection + SGDClassifier
===================================================
1. 512 → 256 feature projection before LDA (runtime vs accuracy tradeoff)
2. SGDClassifier with log_loss vs LogisticRegression

Author: Research Study
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from reduction.lda import LDAReducer
from reduction.pca import PCAReducer


def load_features(features_dir: Path) -> tuple:
    """Load pre-extracted features."""
    saved_dir = features_dir / "saved"
    X_train = np.load(saved_dir / "X_train.npy")
    y_train = np.load(saved_dir / "y_train.npy")
    X_test = np.load(saved_dir / "X_test.npy")
    y_test = np.load(saved_dir / "y_test.npy")
    return X_train, y_train, X_test, y_test


def project_features(X_train: np.ndarray, X_test: np.ndarray, 
                     target_dim: int, seed: int) -> tuple:
    """
    Project features to lower dimension using PCA.
    Returns projected features and timing info.
    """
    start = time.perf_counter()
    
    # Use PCA for deterministic projection (faster than random projection)
    projector = PCA(n_components=target_dim, random_state=seed)
    X_train_proj = projector.fit_transform(X_train)
    X_test_proj = projector.transform(X_test)
    
    proj_time = time.perf_counter() - start
    
    return X_train_proj, X_test_proj, proj_time


def run_lda_experiment(X_train, y_train, X_test, y_test, 
                       n_components: int, classifier_type: str,
                       seed: int) -> dict:
    """
    Run LDA + classifier experiment.
    
    Args:
        classifier_type: 'logistic' or 'sgd'
    """
    results = {}
    
    # LDA reduction
    start = time.perf_counter()
    lda = LDAReducer(n_components=n_components)
    lda.fit(X_train, y_train)
    results['lda_fit_sec'] = time.perf_counter() - start
    
    start = time.perf_counter()
    X_train_lda = lda.transform(X_train)
    results['lda_transform_train_sec'] = time.perf_counter() - start
    
    start = time.perf_counter()
    X_test_lda = lda.transform(X_test)
    results['lda_transform_test_sec'] = time.perf_counter() - start
    
    # Classifier
    if classifier_type == 'logistic':
        clf = LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            random_state=seed,
            n_jobs=-1
        )
    else:  # sgd
        clf = SGDClassifier(
            loss='log_loss',  # Logistic regression loss
            max_iter=1000,
            tol=1e-3,
            random_state=seed,
            n_jobs=-1,
            early_stopping=True,
            validation_fraction=0.1
        )
    
    start = time.perf_counter()
    clf.fit(X_train_lda, y_train)
    results['classifier_train_sec'] = time.perf_counter() - start
    
    start = time.perf_counter()
    accuracy = clf.score(X_test_lda, y_test)
    results['evaluation_sec'] = time.perf_counter() - start
    
    results['accuracy'] = accuracy
    
    return results


def run_ablation_study():
    """Run complete ablation study."""
    
    # Configuration
    seeds = [0, 1, 2, 3, 4]
    lda_components = [2, 5, 10, 20, 40, 80, 99]
    feature_dims = [512, 256]  # Original and projected
    classifier_types = ['logistic', 'sgd']
    
    features_dir = project_root / "features"
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("ABLATION STUDY: Feature Projection + SGDClassifier")
    print("=" * 70)
    
    # Load original 512-dim features
    print("\nLoading features...")
    X_train_512, y_train, X_test_512, y_test = load_features(features_dir)
    print(f"  Train: {X_train_512.shape}, Test: {X_test_512.shape}")
    
    all_results = []
    total_experiments = len(seeds) * len(lda_components) * len(feature_dims) * len(classifier_types)
    current_exp = 0
    
    for seed in seeds:
        np.random.seed(seed)
        
        # Pre-compute 256-dim projection for this seed
        X_train_256, X_test_256, proj_time = project_features(
            X_train_512, X_test_512, 256, seed
        )
        
        for feat_dim in feature_dims:
            # Select features based on dimension
            if feat_dim == 512:
                X_train = X_train_512
                X_test = X_test_512
                projection_time = 0.0
            else:
                X_train = X_train_256
                X_test = X_test_256
                projection_time = proj_time
            
            for n_components in lda_components:
                for clf_type in classifier_types:
                    current_exp += 1
                    
                    exp_start = time.perf_counter()
                    
                    # Run experiment
                    exp_results = run_lda_experiment(
                        X_train, y_train, X_test, y_test,
                        n_components=n_components,
                        classifier_type=clf_type,
                        seed=seed
                    )
                    
                    total_time = time.perf_counter() - exp_start
                    
                    # Record results
                    result = {
                        'feature_dim': feat_dim,
                        'lda_components': n_components,
                        'classifier': clf_type,
                        'seed': seed,
                        'accuracy': exp_results['accuracy'],
                        'projection_sec': projection_time,
                        'lda_fit_sec': exp_results['lda_fit_sec'],
                        'lda_transform_train_sec': exp_results['lda_transform_train_sec'],
                        'lda_transform_test_sec': exp_results['lda_transform_test_sec'],
                        'classifier_train_sec': exp_results['classifier_train_sec'],
                        'evaluation_sec': exp_results['evaluation_sec'],
                        'total_runtime_sec': total_time + projection_time
                    }
                    
                    all_results.append(result)
                    
                    print(f"[{current_exp:3d}/{total_experiments}] "
                          f"dim={feat_dim:3d} | comp={n_components:2d} | "
                          f"clf={clf_type:8s} | seed={seed} | "
                          f"acc={exp_results['accuracy']:.4f} | "
                          f"time={result['total_runtime_sec']:.2f}s")
    
    # Save results
    df = pd.DataFrame(all_results)
    output_path = results_dir / "ablation_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("ABLATION STUDY SUMMARY")
    print("=" * 70)
    
    # Summary by feature dimension
    print("\n1. Feature Projection Effect (512 vs 256):")
    print("-" * 50)
    for feat_dim in feature_dims:
        subset = df[df['feature_dim'] == feat_dim]
        mean_acc = subset['accuracy'].mean()
        std_acc = subset['accuracy'].std()
        mean_time = subset['total_runtime_sec'].mean()
        print(f"  {feat_dim}D: acc={mean_acc:.4f} ± {std_acc:.4f}, "
              f"avg_time={mean_time:.2f}s")
    
    # At max components (99)
    print("\n  At 99 LDA components (logistic):")
    for feat_dim in feature_dims:
        subset = df[(df['feature_dim'] == feat_dim) & 
                   (df['lda_components'] == 99) & 
                   (df['classifier'] == 'logistic')]
        mean_acc = subset['accuracy'].mean()
        mean_time = subset['total_runtime_sec'].mean()
        print(f"    {feat_dim}D → 99 LDA: acc={mean_acc:.4f}, time={mean_time:.2f}s")
    
    # Summary by classifier
    print("\n2. Classifier Comparison (LogisticRegression vs SGDClassifier):")
    print("-" * 50)
    for clf_type in classifier_types:
        subset = df[df['classifier'] == clf_type]
        mean_acc = subset['accuracy'].mean()
        std_acc = subset['accuracy'].std()
        mean_time = subset['classifier_train_sec'].mean()
        print(f"  {clf_type:10s}: acc={mean_acc:.4f} ± {std_acc:.4f}, "
              f"clf_train_time={mean_time:.2f}s")
    
    # Best configurations
    print("\n3. Best Configurations:")
    print("-" * 50)
    
    # Best overall
    best_idx = df['accuracy'].idxmax()
    best = df.loc[best_idx]
    print(f"  Best accuracy: {best['accuracy']:.4f}")
    print(f"    Config: dim={int(best['feature_dim'])}, "
          f"comp={int(best['lda_components'])}, "
          f"clf={best['classifier']}, seed={int(best['seed'])}")
    
    # Best 256D
    df_256 = df[df['feature_dim'] == 256]
    best_256_idx = df_256['accuracy'].idxmax()
    best_256 = df_256.loc[best_256_idx]
    print(f"\n  Best 256D accuracy: {best_256['accuracy']:.4f}")
    print(f"    Config: comp={int(best_256['lda_components'])}, "
          f"clf={best_256['classifier']}, seed={int(best_256['seed'])}")
    
    # Compare 512 vs 256 at same config
    print("\n4. Direct Comparison (512 vs 256 at 99 components, logistic):")
    print("-" * 50)
    for feat_dim in feature_dims:
        subset = df[(df['feature_dim'] == feat_dim) & 
                   (df['lda_components'] == 99) & 
                   (df['classifier'] == 'logistic')]
        print(f"  {feat_dim}D: {subset['accuracy'].values}")
    
    return df


if __name__ == "__main__":
    results = run_ablation_study()
