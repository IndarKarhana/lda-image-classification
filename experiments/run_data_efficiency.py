"""
Training Data Efficiency Study.

Tests whether LDA advantage persists (or increases) with less training data.

Hypothesis: LDA may provide even LARGER advantage with limited training data because:
- Supervised dimensionality reduction better exploits available labels
- Full features may overfit more with fewer samples
- LDA regularization effect is stronger when data is scarce

Tests: 10%, 25%, 50%, 100% of training data

For publication: Shows practical utility for low-data regimes.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Training data fractions to test
DATA_FRACTIONS = [0.10, 0.25, 0.50, 1.00]

# Seeds for reproducibility
SEEDS = [42, 123, 456]


def stratified_subsample(X, y, fraction, seed):
    """
    Stratified subsampling to maintain class distribution.
    
    Args:
        X: Features (n_samples, n_features)
        y: Labels (n_samples,)
        fraction: Fraction of data to keep (0.0-1.0)
        seed: Random seed
    
    Returns:
        X_subset, y_subset
    """
    if fraction >= 1.0:
        return X, y
    
    # Use stratified split, keep only the "train" portion
    X_keep, _, y_keep, _ = train_test_split(
        X, y, 
        train_size=fraction,
        stratify=y,
        random_state=seed
    )
    
    return X_keep, y_keep


def run_experiment(X_train, y_train, X_test, y_test, 
                   method: str, n_components: int, seed: int):
    """
    Run a single experiment.
    
    Args:
        method: 'LDA', 'PCA', or 'Full'
        n_components: Number of components
        seed: Random seed for classifier
    
    Returns:
        accuracy
    """
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply dimensionality reduction
    if method == 'LDA':
        # Limit components based on available data
        n_classes = len(np.unique(y_train))
        max_components = min(n_components, n_classes - 1, X_train.shape[1] - 1)
        
        reducer = LinearDiscriminantAnalysis(n_components=max_components)
        try:
            X_train_reduced = reducer.fit_transform(X_train_scaled, y_train)
            X_test_reduced = reducer.transform(X_test_scaled)
        except Exception as e:
            print(f"    LDA failed: {e}")
            return np.nan
            
    elif method == 'PCA':
        reducer = PCA(n_components=n_components, random_state=seed)
        X_train_reduced = reducer.fit_transform(X_train_scaled)
        X_test_reduced = reducer.transform(X_test_scaled)
    else:  # Full features
        X_train_reduced = X_train_scaled
        X_test_reduced = X_test_scaled
    
    # Train classifier
    clf = LogisticRegression(
        max_iter=2000,
        solver='lbfgs',
        n_jobs=-1,
        random_state=seed
    )
    clf.fit(X_train_reduced, y_train)
    
    # Predict
    y_pred = clf.predict(X_test_reduced)
    accuracy = (y_pred == y_test).mean()
    
    return accuracy


def run_data_efficiency_study(X_train, y_train, X_test, y_test, 
                               n_components: int, dataset_name: str):
    """
    Run LDA vs Full vs PCA comparison across different training data amounts.
    
    Returns:
        DataFrame with results
    """
    results = []
    
    for fraction in DATA_FRACTIONS:
        n_samples = int(len(X_train) * fraction)
        print(f"\n  {fraction*100:.0f}% training data ({n_samples:,} samples):")
        
        for seed in SEEDS:
            # Subsample training data
            X_train_sub, y_train_sub = stratified_subsample(
                X_train, y_train, fraction, seed
            )
            
            # Full features
            acc_full = run_experiment(
                X_train_sub, y_train_sub, X_test, y_test, 
                'Full', None, seed
            )
            
            # LDA
            acc_lda = run_experiment(
                X_train_sub, y_train_sub, X_test, y_test,
                'LDA', n_components, seed
            )
            
            # PCA
            acc_pca = run_experiment(
                X_train_sub, y_train_sub, X_test, y_test,
                'PCA', n_components, seed
            )
            
            results.append({
                'dataset': dataset_name,
                'train_fraction': fraction,
                'n_train_samples': len(X_train_sub),
                'seed': seed,
                'n_components': n_components,
                'acc_full': acc_full,
                'acc_lda': acc_lda,
                'acc_pca': acc_pca,
                'lda_vs_full': acc_lda - acc_full if not np.isnan(acc_lda) else np.nan,
                'lda_vs_pca': acc_lda - acc_pca if not np.isnan(acc_lda) else np.nan
            })
            
        # Print progress
        frac_results = [r for r in results if r['train_fraction'] == fraction and r['dataset'] == dataset_name]
        mean_full = np.nanmean([r['acc_full'] for r in frac_results])
        mean_lda = np.nanmean([r['acc_lda'] for r in frac_results])
        mean_lda_gain = np.nanmean([r['lda_vs_full'] for r in frac_results])
        print(f"    Full: {mean_full:.4f}, LDA: {mean_lda:.4f}, LDA gain: {mean_lda_gain*100:+.2f}%")
    
    return pd.DataFrame(results)


def run_cifar100_efficiency():
    """Run data efficiency study on CIFAR-100."""
    print("\n" + "=" * 70)
    print("CIFAR-100 DATA EFFICIENCY STUDY")
    print("=" * 70)
    
    from features.extract_features import get_or_extract_features
    X_train, y_train, X_test, y_test, feature_dim = get_or_extract_features()
    
    n_components = 99  # Max for 100 classes
    
    print(f"Feature dimension: {feature_dim}")
    print(f"Total training samples: {len(X_train):,}")
    print(f"LDA components: {n_components}")
    
    return run_data_efficiency_study(
        X_train, y_train, X_test, y_test, n_components, 'cifar100'
    )


def run_tiny_imagenet_efficiency():
    """Run data efficiency study on Tiny ImageNet."""
    print("\n" + "=" * 70)
    print("TINY IMAGENET DATA EFFICIENCY STUDY")
    print("=" * 70)
    
    from features.extract_tiny_imagenet import get_or_extract_features, BACKBONES
    
    all_results = []
    
    # Test on subset of backbones to save time
    test_backbones = ['ResNet-18', 'MobileNetV3-Small']
    
    for backbone_name in test_backbones:
        if backbone_name not in BACKBONES:
            continue
            
        print(f"\n{backbone_name}:")
        
        X_train, y_train, X_test, y_test, feature_dim = get_or_extract_features(backbone_name)
        
        n_components = 199  # Max for 200 classes
        
        print(f"  Feature dimension: {feature_dim}")
        print(f"  Total training samples: {len(X_train):,}")
        
        results = run_data_efficiency_study(
            X_train, y_train, X_test, y_test, 
            n_components, f'tiny_imagenet_{backbone_name}'
        )
        all_results.append(results)
    
    return pd.concat(all_results, ignore_index=True)


def analyze_results(df):
    """Analyze and summarize data efficiency results."""
    print("\n" + "=" * 70)
    print("DATA EFFICIENCY ANALYSIS SUMMARY")
    print("=" * 70)
    
    # Aggregate by dataset and fraction
    summary = df.groupby(['dataset', 'train_fraction']).agg({
        'n_train_samples': 'first',
        'acc_full': ['mean', 'std'],
        'acc_lda': ['mean', 'std'],
        'acc_pca': ['mean', 'std'],
        'lda_vs_full': ['mean', 'std'],
        'lda_vs_pca': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    
    for dataset in df['dataset'].unique():
        print(f"\n{dataset}:")
        dataset_summary = summary[summary['dataset'] == dataset]
        
        print(f"  {'Fraction':<10} {'Samples':<10} {'Full':<15} {'LDA':<15} {'LDA Gain':<12}")
        print("  " + "-" * 60)
        
        for _, row in dataset_summary.iterrows():
            frac = f"{row['train_fraction']*100:.0f}%"
            samples = f"{int(row['n_train_samples_first']):,}"
            full = f"{row['acc_full_mean']:.4f}±{row['acc_full_std']:.4f}"
            lda = f"{row['acc_lda_mean']:.4f}±{row['acc_lda_std']:.4f}"
            gain = f"{row['lda_vs_full_mean']*100:+.2f}%"
            print(f"  {frac:<10} {samples:<10} {full:<15} {lda:<15} {gain:<12}")
    
    # Key finding: Does LDA advantage increase with less data?
    print("\n" + "-" * 70)
    print("KEY QUESTION: Does LDA advantage increase with less training data?")
    print("-" * 70)
    
    for dataset in df['dataset'].unique():
        dataset_summary = summary[summary['dataset'] == dataset].sort_values('train_fraction')
        gains = dataset_summary['lda_vs_full_mean'].values
        fractions = dataset_summary['train_fraction'].values
        
        # Compare gain at 10% vs 100%
        if len(gains) >= 2:
            gain_10 = gains[0]  # First fraction (10%)
            gain_100 = gains[-1]  # Last fraction (100%)
            
            print(f"\n{dataset}:")
            print(f"  LDA gain at 10% data: {gain_10*100:+.2f}%")
            print(f"  LDA gain at 100% data: {gain_100*100:+.2f}%")
            
            if gain_10 > gain_100:
                print(f"  → LDA advantage INCREASES with less data (+{(gain_10-gain_100)*100:.2f}%)")
            else:
                print(f"  → LDA advantage does not increase with less data")
    
    return summary


def main():
    """Run complete data efficiency study."""
    start_time = time.time()
    
    # Create results directory
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'data_efficiency'
    )
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = []
    
    # CIFAR-100
    try:
        cifar_results = run_cifar100_efficiency()
        all_results.append(cifar_results)
    except Exception as e:
        print(f"CIFAR-100 study failed: {e}")
    
    # Tiny ImageNet
    try:
        tiny_results = run_tiny_imagenet_efficiency()
        all_results.append(tiny_results)
    except Exception as e:
        print(f"Tiny ImageNet study failed: {e}")
    
    # Combine and save results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(
            os.path.join(results_dir, 'data_efficiency_results.csv'),
            index=False
        )
        
        # Analyze and save summary
        summary = analyze_results(combined_df)
        summary.to_csv(
            os.path.join(results_dir, 'data_efficiency_summary.csv'),
            index=False
        )
        
        print("\n" + "=" * 70)
        print("RESULTS SAVED TO:")
        print(f"  {os.path.join(results_dir, 'data_efficiency_results.csv')}")
        print(f"  {os.path.join(results_dir, 'data_efficiency_summary.csv')}")
    
    elapsed = time.time() - start_time
    print(f"\nTotal analysis time: {elapsed/60:.1f} minutes")


if __name__ == '__main__':
    main()
