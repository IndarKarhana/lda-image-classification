"""
Statistical Significance Analysis for LDA vs Full Features.

This script runs multiple seeds to establish statistical significance of:
1. LDA improvement over full features
2. LDA improvement over PCA

Outputs:
- Mean ± std accuracy across seeds
- Paired t-test and Wilcoxon signed-rank test p-values
- Per-class consistency analysis

For publication: Demonstrates LDA advantage is not due to random chance.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Number of seeds for statistical significance
NUM_SEEDS = 5
SEEDS = [42, 123, 456, 789, 1024]

# Datasets to test
DATASETS = ['cifar100', 'tiny_imagenet']


def load_features(dataset: str, backbone: str = 'resnet18'):
    """Load pre-extracted features for a dataset."""
    if dataset == 'cifar100':
        from features.extract_features import get_or_extract_features
        return get_or_extract_features()
    else:  # tiny_imagenet
        from features.extract_tiny_imagenet import get_or_extract_features
        return get_or_extract_features(backbone)


def run_single_experiment(X_train, y_train, X_test, y_test, 
                          method: str, n_components: int, seed: int):
    """
    Run a single experiment with specified seed.
    
    Args:
        method: 'LDA', 'PCA', or 'Full'
        n_components: Number of components (ignored if method='Full')
        seed: Random seed for classifier
    
    Returns:
        accuracy, per_class_accuracy (array)
    """
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply dimensionality reduction
    if method == 'LDA':
        reducer = LinearDiscriminantAnalysis(n_components=n_components)
        X_train_reduced = reducer.fit_transform(X_train_scaled, y_train)
        X_test_reduced = reducer.transform(X_test_scaled)
    elif method == 'PCA':
        reducer = PCA(n_components=n_components, random_state=seed)
        X_train_reduced = reducer.fit_transform(X_train_scaled)
        X_test_reduced = reducer.transform(X_test_scaled)
    else:  # Full features
        X_train_reduced = X_train_scaled
        X_test_reduced = X_test_scaled
    
    # Train classifier with specified seed
    clf = LogisticRegression(
        max_iter=2000,
        solver='lbfgs',
        n_jobs=-1,
        random_state=seed
    )
    clf.fit(X_train_reduced, y_train)
    
    # Predict
    y_pred = clf.predict(X_test_reduced)
    
    # Overall accuracy
    accuracy = (y_pred == y_test).mean()
    
    # Per-class accuracy
    classes = np.unique(y_test)
    per_class_acc = np.zeros(len(classes))
    for i, c in enumerate(classes):
        mask = y_test == c
        per_class_acc[i] = (y_pred[mask] == y_test[mask]).mean()
    
    return accuracy, per_class_acc


def run_multi_seed_comparison(X_train, y_train, X_test, y_test, 
                               n_components: int, dataset_name: str):
    """
    Run LDA vs Full vs PCA comparison across multiple seeds.
    
    Returns:
        DataFrame with results for statistical tests
    """
    results = []
    
    for seed in SEEDS:
        print(f"  Seed {seed}...")
        
        # Full features
        acc_full, pc_full = run_single_experiment(
            X_train, y_train, X_test, y_test, 'Full', None, seed
        )
        
        # LDA
        acc_lda, pc_lda = run_single_experiment(
            X_train, y_train, X_test, y_test, 'LDA', n_components, seed
        )
        
        # PCA
        acc_pca, pc_pca = run_single_experiment(
            X_train, y_train, X_test, y_test, 'PCA', n_components, seed
        )
        
        results.append({
            'dataset': dataset_name,
            'seed': seed,
            'n_components': n_components,
            'acc_full': acc_full,
            'acc_lda': acc_lda,
            'acc_pca': acc_pca,
            'lda_vs_full': acc_lda - acc_full,
            'lda_vs_pca': acc_lda - acc_pca,
            'per_class_full': pc_full,
            'per_class_lda': pc_lda,
            'per_class_pca': pc_pca
        })
    
    return pd.DataFrame(results)


def compute_statistical_tests(results_df):
    """
    Compute statistical significance tests.
    
    Tests:
    1. Paired t-test (parametric, assumes normality)
    2. Wilcoxon signed-rank test (non-parametric)
    
    Returns:
        Dictionary with test statistics and p-values
    """
    acc_full = results_df['acc_full'].values
    acc_lda = results_df['acc_lda'].values
    acc_pca = results_df['acc_pca'].values
    
    tests = {}
    
    # LDA vs Full
    t_stat, t_pval = stats.ttest_rel(acc_lda, acc_full)
    w_stat, w_pval = stats.wilcoxon(acc_lda, acc_full, alternative='greater')
    
    tests['lda_vs_full'] = {
        'mean_diff': (acc_lda - acc_full).mean(),
        'std_diff': (acc_lda - acc_full).std(),
        't_statistic': t_stat,
        't_pvalue': t_pval,
        'wilcoxon_statistic': w_stat,
        'wilcoxon_pvalue': w_pval,
        'significant_at_005': t_pval < 0.05 and w_pval < 0.05
    }
    
    # LDA vs PCA
    t_stat, t_pval = stats.ttest_rel(acc_lda, acc_pca)
    w_stat, w_pval = stats.wilcoxon(acc_lda, acc_pca, alternative='greater')
    
    tests['lda_vs_pca'] = {
        'mean_diff': (acc_lda - acc_pca).mean(),
        'std_diff': (acc_lda - acc_pca).std(),
        't_statistic': t_stat,
        't_pvalue': t_pval,
        'wilcoxon_statistic': w_stat,
        'wilcoxon_pvalue': w_pval,
        'significant_at_005': t_pval < 0.05 and w_pval < 0.05
    }
    
    return tests


def compute_class_consistency(results_df):
    """
    Analyze consistency of LDA advantage across classes and seeds.
    
    Returns:
        - Fraction of classes where LDA wins in ALL seeds
        - Fraction of classes where LDA wins in MAJORITY of seeds
    """
    # Stack per-class results
    per_class_lda = np.vstack(results_df['per_class_lda'].values)  # (n_seeds, n_classes)
    per_class_full = np.vstack(results_df['per_class_full'].values)
    per_class_pca = np.vstack(results_df['per_class_pca'].values)
    
    n_seeds, n_classes = per_class_lda.shape
    
    # LDA vs Full: count wins per class
    lda_wins_full = (per_class_lda > per_class_full).sum(axis=0)  # (n_classes,)
    all_wins_full = (lda_wins_full == n_seeds).sum() / n_classes
    majority_wins_full = (lda_wins_full > n_seeds / 2).sum() / n_classes
    
    # LDA vs PCA: count wins per class
    lda_wins_pca = (per_class_lda > per_class_pca).sum(axis=0)
    all_wins_pca = (lda_wins_pca == n_seeds).sum() / n_classes
    majority_wins_pca = (lda_wins_pca > n_seeds / 2).sum() / n_classes
    
    return {
        'lda_vs_full': {
            'classes_lda_wins_all_seeds': all_wins_full,
            'classes_lda_wins_majority': majority_wins_full
        },
        'lda_vs_pca': {
            'classes_lda_wins_all_seeds': all_wins_pca,
            'classes_lda_wins_majority': majority_wins_pca
        }
    }


def run_cifar100_significance():
    """Run statistical significance analysis on CIFAR-100."""
    print("\n" + "=" * 70)
    print("CIFAR-100 STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 70)
    
    # Load features
    from features.extract_features import get_or_extract_features
    X_train, y_train, X_test, y_test, feature_dim = get_or_extract_features()
    
    # CIFAR-100 has 100 classes, max LDA components = 99
    n_components = 99
    
    print(f"\nFeature dimension: {feature_dim}")
    print(f"LDA components: {n_components}")
    print(f"Running {NUM_SEEDS} seeds: {SEEDS}")
    
    # Run comparison
    results_df = run_multi_seed_comparison(
        X_train, y_train, X_test, y_test, n_components, 'cifar100'
    )
    
    return results_df


def run_tiny_imagenet_significance():
    """Run statistical significance analysis on Tiny ImageNet."""
    print("\n" + "=" * 70)
    print("TINY IMAGENET STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 70)
    
    # Load features for all backbones
    from features.extract_tiny_imagenet import get_or_extract_features, BACKBONES
    
    all_results = []
    
    for backbone_name in BACKBONES.keys():
        print(f"\n{backbone_name}:")
        
        X_train, y_train, X_test, y_test, feature_dim = get_or_extract_features(backbone_name)
        
        # Tiny ImageNet has 200 classes, max LDA components = 199
        n_components = 199
        
        print(f"  Feature dimension: {feature_dim}")
        print(f"  LDA components: {n_components}")
        
        # Run comparison
        results_df = run_multi_seed_comparison(
            X_train, y_train, X_test, y_test, n_components, f'tiny_imagenet_{backbone_name}'
        )
        
        all_results.append(results_df)
    
    return pd.concat(all_results, ignore_index=True)


def main():
    """Run complete statistical significance analysis."""
    start_time = time.time()
    
    # Create results directory
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'statistical_significance'
    )
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = []
    all_tests = {}
    all_consistency = {}
    
    # CIFAR-100
    try:
        cifar_results = run_cifar100_significance()
        all_results.append(cifar_results)
        
        # Compute tests
        cifar_tests = compute_statistical_tests(cifar_results)
        all_tests['cifar100'] = cifar_tests
        
        # Compute consistency
        cifar_consistency = compute_class_consistency(cifar_results)
        all_consistency['cifar100'] = cifar_consistency
        
        print("\n" + "-" * 50)
        print("CIFAR-100 Results:")
        print(f"  Full features: {cifar_results['acc_full'].mean():.4f} ± {cifar_results['acc_full'].std():.4f}")
        print(f"  LDA:           {cifar_results['acc_lda'].mean():.4f} ± {cifar_results['acc_lda'].std():.4f}")
        print(f"  PCA:           {cifar_results['acc_pca'].mean():.4f} ± {cifar_results['acc_pca'].std():.4f}")
        print(f"\n  LDA vs Full:")
        print(f"    Mean improvement: {cifar_tests['lda_vs_full']['mean_diff']*100:+.2f}%")
        print(f"    t-test p-value: {cifar_tests['lda_vs_full']['t_pvalue']:.6f}")
        print(f"    Wilcoxon p-value: {cifar_tests['lda_vs_full']['wilcoxon_pvalue']:.6f}")
        print(f"    Significant (p<0.05): {cifar_tests['lda_vs_full']['significant_at_005']}")
        
    except Exception as e:
        print(f"CIFAR-100 analysis failed: {e}")
    
    # Tiny ImageNet
    try:
        tiny_results = run_tiny_imagenet_significance()
        all_results.append(tiny_results)
        
        # Compute tests per backbone
        for dataset_name in tiny_results['dataset'].unique():
            backbone_results = tiny_results[tiny_results['dataset'] == dataset_name]
            
            tests = compute_statistical_tests(backbone_results)
            all_tests[dataset_name] = tests
            
            consistency = compute_class_consistency(backbone_results)
            all_consistency[dataset_name] = consistency
            
            print(f"\n{dataset_name}:")
            print(f"  Full: {backbone_results['acc_full'].mean():.4f} ± {backbone_results['acc_full'].std():.4f}")
            print(f"  LDA:  {backbone_results['acc_lda'].mean():.4f} ± {backbone_results['acc_lda'].std():.4f}")
            print(f"  LDA vs Full p-value (Wilcoxon): {tests['lda_vs_full']['wilcoxon_pvalue']:.6f}")
            
    except Exception as e:
        print(f"Tiny ImageNet analysis failed: {e}")
    
    # Save all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save raw results (without per-class arrays)
        save_df = combined_df.drop(columns=['per_class_full', 'per_class_lda', 'per_class_pca'])
        save_df.to_csv(os.path.join(results_dir, 'multi_seed_results.csv'), index=False)
        
        # Save summary
        summary_rows = []
        for dataset in combined_df['dataset'].unique():
            df_subset = combined_df[combined_df['dataset'] == dataset]
            summary_rows.append({
                'dataset': dataset,
                'n_components': df_subset['n_components'].iloc[0],
                'acc_full_mean': df_subset['acc_full'].mean(),
                'acc_full_std': df_subset['acc_full'].std(),
                'acc_lda_mean': df_subset['acc_lda'].mean(),
                'acc_lda_std': df_subset['acc_lda'].std(),
                'acc_pca_mean': df_subset['acc_pca'].mean(),
                'acc_pca_std': df_subset['acc_pca'].std(),
                'lda_vs_full_mean': df_subset['lda_vs_full'].mean(),
                'lda_vs_pca_mean': df_subset['lda_vs_pca'].mean(),
                't_pvalue_lda_vs_full': all_tests.get(dataset, {}).get('lda_vs_full', {}).get('t_pvalue', np.nan),
                'wilcoxon_pvalue_lda_vs_full': all_tests.get(dataset, {}).get('lda_vs_full', {}).get('wilcoxon_pvalue', np.nan),
                't_pvalue_lda_vs_pca': all_tests.get(dataset, {}).get('lda_vs_pca', {}).get('t_pvalue', np.nan),
                'wilcoxon_pvalue_lda_vs_pca': all_tests.get(dataset, {}).get('lda_vs_pca', {}).get('wilcoxon_pvalue', np.nan),
            })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(results_dir, 'significance_summary.csv'), index=False)
        
        print("\n" + "=" * 70)
        print("RESULTS SAVED TO:")
        print(f"  {os.path.join(results_dir, 'multi_seed_results.csv')}")
        print(f"  {os.path.join(results_dir, 'significance_summary.csv')}")
    
    elapsed = time.time() - start_time
    print(f"\nTotal analysis time: {elapsed/60:.1f} minutes")


if __name__ == '__main__':
    main()
