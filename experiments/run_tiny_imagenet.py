"""
Tiny ImageNet Backbone Comparison Experiment.

Tests whether the surprising LDA finding from CIFAR-100 generalizes:
- LDA improving accuracy over full features (not just maintaining it)
- LDA advantage varying by backbone architecture

DETAILED TIMING BREAKDOWN included to answer:
"Where exactly does the speedup come from?"

Timing categories:
1. Feature extraction (ONE-TIME, same for all methods - cached)
2. Standardization (small, same for all)
3. Dimensionality reduction fit time
4. Dimensionality reduction transform time (train + test)
5. Classifier training time
6. Classifier inference time

This allows fair claims about speedup.
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
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.extract_tiny_imagenet import get_or_extract_features, BACKBONES


def run_experiment_with_timing(X_train, y_train, X_test, y_test, 
                                method: str, n_components: int = None):
    """
    Run a single experiment with DETAILED timing breakdown.
    
    Args:
        method: 'LDA', 'PCA', or 'None' (full features)
        n_components: Number of components (ignored if method='None')
    
    Returns:
        dict with accuracy and all timing components
    """
    timing = {}
    total_start = time.time()
    
    # Standardize features (same for all methods)
    std_start = time.time()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    timing['standardization_sec'] = time.time() - std_start
    
    # Apply dimensionality reduction
    if method == 'LDA':
        # Fit LDA
        fit_start = time.time()
        reducer = LinearDiscriminantAnalysis(n_components=n_components)
        reducer.fit(X_train_scaled, y_train)
        timing['reduction_fit_sec'] = time.time() - fit_start
        
        # Transform train
        transform_start = time.time()
        X_train_reduced = reducer.transform(X_train_scaled)
        timing['transform_train_sec'] = time.time() - transform_start
        
        # Transform test
        transform_start = time.time()
        X_test_reduced = reducer.transform(X_test_scaled)
        timing['transform_test_sec'] = time.time() - transform_start
        
    elif method == 'PCA':
        # Fit PCA
        fit_start = time.time()
        reducer = PCA(n_components=n_components)
        reducer.fit(X_train_scaled)
        timing['reduction_fit_sec'] = time.time() - fit_start
        
        # Transform train
        transform_start = time.time()
        X_train_reduced = reducer.transform(X_train_scaled)
        timing['transform_train_sec'] = time.time() - transform_start
        
        # Transform test
        transform_start = time.time()
        X_test_reduced = reducer.transform(X_test_scaled)
        timing['transform_test_sec'] = time.time() - transform_start
        
    else:  # None - use full features
        timing['reduction_fit_sec'] = 0.0
        timing['transform_train_sec'] = 0.0
        timing['transform_test_sec'] = 0.0
        X_train_reduced = X_train_scaled
        X_test_reduced = X_test_scaled
    
    # Train classifier
    clf_train_start = time.time()
    clf = LogisticRegression(
        max_iter=2000,
        solver='lbfgs',
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train_reduced, y_train)
    timing['classifier_train_sec'] = time.time() - clf_train_start
    
    # Inference on test set
    inference_start = time.time()
    y_pred = clf.predict(X_test_reduced)
    timing['classifier_inference_sec'] = time.time() - inference_start
    
    # Calculate accuracy
    accuracy = (y_pred == y_test).mean()
    
    timing['total_pipeline_sec'] = time.time() - total_start
    timing['accuracy'] = accuracy
    
    return timing


def run_backbone_comparison():
    """Run LDA vs PCA vs Full comparison across all backbones with detailed timing."""
    
    # Tiny ImageNet has 200 classes, so max LDA components = 199
    component_values = [10, 20, 40, 80, 99, 150, 199]
    
    results = []
    
    for backbone_name in BACKBONES.keys():
        print(f"\n{'='*60}")
        print(f"Testing {backbone_name}")
        print('='*60)
        
        # Get features (timing for extraction is tracked separately)
        extraction_start = time.time()
        X_train, y_train, X_test, y_test, feature_dim = get_or_extract_features(
            backbone_name
        )
        extraction_time = time.time() - extraction_start
        
        print(f"Feature dimension: {feature_dim}")
        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"Feature extraction/load time: {extraction_time:.2f}s")
        
        # Test full features (no reduction)
        print("\nTesting full features (no reduction)...")
        timing = run_experiment_with_timing(X_train, y_train, X_test, y_test, 'None')
        results.append({
            'backbone': backbone_name,
            'feature_dim': feature_dim,
            'method': 'None (Full)',
            'components': feature_dim,
            'accuracy': timing['accuracy'],
            'standardization_sec': timing['standardization_sec'],
            'reduction_fit_sec': timing['reduction_fit_sec'],
            'transform_train_sec': timing['transform_train_sec'],
            'transform_test_sec': timing['transform_test_sec'],
            'classifier_train_sec': timing['classifier_train_sec'],
            'classifier_inference_sec': timing['classifier_inference_sec'],
            'total_pipeline_sec': timing['total_pipeline_sec']
        })
        print(f"  Full features: {timing['accuracy']:.4f}")
        print(f"    Classifier train: {timing['classifier_train_sec']:.2f}s")
        print(f"    Inference: {timing['classifier_inference_sec']*1000:.1f}ms")
        
        # Test LDA and PCA at different component counts
        for n_comp in component_values:
            # Skip if more components than features allow
            if n_comp >= feature_dim:
                continue
                
            print(f"\nTesting {n_comp} components...")
            
            # LDA
            timing_lda = run_experiment_with_timing(
                X_train, y_train, X_test, y_test, 'LDA', n_comp
            )
            results.append({
                'backbone': backbone_name,
                'feature_dim': feature_dim,
                'method': 'LDA',
                'components': n_comp,
                'accuracy': timing_lda['accuracy'],
                'standardization_sec': timing_lda['standardization_sec'],
                'reduction_fit_sec': timing_lda['reduction_fit_sec'],
                'transform_train_sec': timing_lda['transform_train_sec'],
                'transform_test_sec': timing_lda['transform_test_sec'],
                'classifier_train_sec': timing_lda['classifier_train_sec'],
                'classifier_inference_sec': timing_lda['classifier_inference_sec'],
                'total_pipeline_sec': timing_lda['total_pipeline_sec']
            })
            
            # PCA
            timing_pca = run_experiment_with_timing(
                X_train, y_train, X_test, y_test, 'PCA', n_comp
            )
            results.append({
                'backbone': backbone_name,
                'feature_dim': feature_dim,
                'method': 'PCA',
                'components': n_comp,
                'accuracy': timing_pca['accuracy'],
                'standardization_sec': timing_pca['standardization_sec'],
                'reduction_fit_sec': timing_pca['reduction_fit_sec'],
                'transform_train_sec': timing_pca['transform_train_sec'],
                'transform_test_sec': timing_pca['transform_test_sec'],
                'classifier_train_sec': timing_pca['classifier_train_sec'],
                'classifier_inference_sec': timing_pca['classifier_inference_sec'],
                'total_pipeline_sec': timing_pca['total_pipeline_sec']
            })
            
            print(f"  LDA: {timing_lda['accuracy']:.4f} (fit: {timing_lda['reduction_fit_sec']:.2f}s, train: {timing_lda['classifier_train_sec']:.2f}s)")
            print(f"  PCA: {timing_pca['accuracy']:.4f} (fit: {timing_pca['reduction_fit_sec']:.2f}s, train: {timing_pca['classifier_train_sec']:.2f}s)")
            print(f"  LDA advantage: {(timing_lda['accuracy'] - timing_pca['accuracy'])*100:+.2f}%")
    
    return pd.DataFrame(results)


def analyze_results(df: pd.DataFrame):
    """Analyze and summarize results with timing breakdown."""
    print("\n" + "="*70)
    print("TINY IMAGENET RESULTS SUMMARY")
    print("="*70)
    
    # Per-backbone analysis
    for backbone in df['backbone'].unique():
        backbone_df = df[df['backbone'] == backbone]
        full_row = backbone_df[backbone_df['method'] == 'None (Full)'].iloc[0]
        full_acc = full_row['accuracy']
        
        lda_df = backbone_df[backbone_df['method'] == 'LDA']
        pca_df = backbone_df[backbone_df['method'] == 'PCA']
        
        best_lda = lda_df.loc[lda_df['accuracy'].idxmax()]
        best_pca = pca_df.loc[pca_df['accuracy'].idxmax()]
        
        print(f"\n{backbone}:")
        print(f"  Full features:  {full_acc:.4f}")
        print(f"  Best LDA:       {best_lda['accuracy']:.4f} ({int(best_lda['components'])} components)")
        print(f"  Best PCA:       {best_pca['accuracy']:.4f} ({int(best_pca['components'])} components)")
        print(f"  LDA vs Full:    {(best_lda['accuracy'] - full_acc)*100:+.2f}%")
        print(f"  LDA vs PCA:     {(best_lda['accuracy'] - best_pca['accuracy'])*100:+.2f}%")
    
    # Key finding: Does LDA improve over full features?
    print("\n" + "-"*70)
    print("KEY FINDING: Does LDA improve over full features?")
    print("-"*70)
    
    for backbone in df['backbone'].unique():
        backbone_df = df[df['backbone'] == backbone]
        full_acc = backbone_df[backbone_df['method'] == 'None (Full)']['accuracy'].values[0]
        best_lda = backbone_df[backbone_df['method'] == 'LDA']['accuracy'].max()
        
        improvement = (best_lda - full_acc) * 100
        if improvement > 0:
            symbol = "✓ YES"
        else:
            symbol = "✗ NO"
        
        print(f"  {backbone}: {symbol} ({improvement:+.2f}%)")
    
    # TIMING BREAKDOWN ANALYSIS
    print("\n" + "="*70)
    print("TIMING BREAKDOWN: Where Does the Speedup Come From?")
    print("="*70)
    print("\n(Feature extraction time is SAME for all methods - not shown)")
    print("(This is the 'after feature extraction' pipeline timing)\n")
    
    for backbone in df['backbone'].unique():
        backbone_df = df[df['backbone'] == backbone]
        full_row = backbone_df[backbone_df['method'] == 'None (Full)'].iloc[0]
        
        # Get LDA-199 (max components for 200 classes)
        lda_max = backbone_df[(backbone_df['method'] == 'LDA') & 
                              (backbone_df['components'] == 199)]
        if len(lda_max) == 0:
            lda_max = backbone_df[backbone_df['method'] == 'LDA'].iloc[-1:]
        lda_row = lda_max.iloc[0]
        
        print(f"\n{backbone} (Full vs LDA-{int(lda_row['components'])}):")
        print(f"  {'Step':<25} {'Full':<12} {'LDA':<12} {'Speedup':<10}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
        
        # Standardization
        print(f"  {'Standardization':<25} {full_row['standardization_sec']:.3f}s       {lda_row['standardization_sec']:.3f}s       {'1.0×':<10}")
        
        # Reduction
        print(f"  {'Reduction fit':<25} {'N/A':<12} {lda_row['reduction_fit_sec']:.3f}s       {'-':<10}")
        print(f"  {'Transform (train)':<25} {'N/A':<12} {lda_row['transform_train_sec']:.3f}s       {'-':<10}")
        print(f"  {'Transform (test)':<25} {'N/A':<12} {lda_row['transform_test_sec']:.3f}s       {'-':<10}")
        
        # Classifier training
        clf_speedup = full_row['classifier_train_sec'] / lda_row['classifier_train_sec'] if lda_row['classifier_train_sec'] > 0 else float('inf')
        print(f"  {'Classifier train':<25} {full_row['classifier_train_sec']:.3f}s       {lda_row['classifier_train_sec']:.3f}s       {clf_speedup:.1f}×")
        
        # Inference
        inf_speedup = full_row['classifier_inference_sec'] / lda_row['classifier_inference_sec'] if lda_row['classifier_inference_sec'] > 0 else float('inf')
        print(f"  {'Inference':<25} {full_row['classifier_inference_sec']*1000:.1f}ms       {lda_row['classifier_inference_sec']*1000:.1f}ms       {inf_speedup:.1f}×")
        
        # Total
        total_speedup = full_row['total_pipeline_sec'] / lda_row['total_pipeline_sec'] if lda_row['total_pipeline_sec'] > 0 else float('inf')
        print(f"  {'TOTAL PIPELINE':<25} {full_row['total_pipeline_sec']:.3f}s       {lda_row['total_pipeline_sec']:.3f}s       {total_speedup:.1f}×")
    
    # Summary table for paper
    print("\n" + "="*70)
    print("SUMMARY TABLE (for paper)")
    print("="*70)
    print(f"\n{'Backbone':<20} {'Full Acc':<12} {'LDA Acc':<12} {'Δ Accuracy':<12} {'Clf Train ↓':<12}")
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    
    for backbone in df['backbone'].unique():
        backbone_df = df[df['backbone'] == backbone]
        full_row = backbone_df[backbone_df['method'] == 'None (Full)'].iloc[0]
        best_lda = backbone_df[backbone_df['method'] == 'LDA'].loc[
            backbone_df[backbone_df['method'] == 'LDA']['accuracy'].idxmax()
        ]
        
        delta = (best_lda['accuracy'] - full_row['accuracy']) * 100
        speedup = full_row['classifier_train_sec'] / best_lda['classifier_train_sec']
        
        print(f"{backbone:<20} {full_row['accuracy']*100:.2f}%       {best_lda['accuracy']*100:.2f}%       {delta:+.2f}%        {speedup:.1f}×")


def run_classwise_analysis(backbone_name: str = 'ResNet-18'):
    """
    Analyze per-class accuracy to understand where LDA helps/hurts.
    """
    print(f"\n{'='*60}")
    print(f"Class-wise Analysis for {backbone_name}")
    print('='*60)
    
    X_train, y_train, X_test, y_test, feature_dim = get_or_extract_features(backbone_name)
    
    # Get max LDA components for 200 classes
    n_components = 199
    
    results = []
    
    for method in ['None', 'LDA', 'PCA']:
        # Fit reducer and classifier
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if method == 'LDA':
            reducer = LinearDiscriminantAnalysis(n_components=n_components)
            X_train_red = reducer.fit_transform(X_train_scaled, y_train)
            X_test_red = reducer.transform(X_test_scaled)
        elif method == 'PCA':
            reducer = PCA(n_components=n_components)
            X_train_red = reducer.fit_transform(X_train_scaled)
            X_test_red = reducer.transform(X_test_scaled)
        else:
            X_train_red = X_train_scaled
            X_test_red = X_test_scaled
        
        clf = LogisticRegression(max_iter=2000, solver='lbfgs', n_jobs=-1, random_state=42)
        clf.fit(X_train_red, y_train)
        
        # Get predictions
        y_pred = clf.predict(X_test_red)
        
        # Per-class accuracy
        for class_id in range(200):
            mask = y_test == class_id
            class_acc = (y_pred[mask] == y_test[mask]).mean()
            results.append({
                'class_id': class_id,
                'method': method if method != 'None' else 'Full',
                'accuracy': class_acc,
                'n_samples': mask.sum()
            })
    
    df = pd.DataFrame(results)
    
    # Pivot to compare methods
    pivot = df.pivot(index='class_id', columns='method', values='accuracy')
    pivot['LDA_vs_Full'] = pivot['LDA'] - pivot['Full']
    pivot['LDA_vs_PCA'] = pivot['LDA'] - pivot['PCA']
    
    # Summary stats
    lda_helps = (pivot['LDA_vs_Full'] > 0).sum()
    lda_hurts = (pivot['LDA_vs_Full'] < 0).sum()
    lda_same = (pivot['LDA_vs_Full'] == 0).sum()
    
    print(f"\nLDA vs Full Features:")
    print(f"  Classes where LDA helps: {lda_helps}/200")
    print(f"  Classes where LDA hurts: {lda_hurts}/200")
    print(f"  Classes unchanged: {lda_same}/200")
    print(f"  Average improvement: {pivot['LDA_vs_Full'].mean()*100:+.2f}%")
    
    # Top improvements and declines
    print(f"\nTop 10 classes where LDA helps most:")
    top_improved = pivot.nlargest(10, 'LDA_vs_Full')[['Full', 'LDA', 'LDA_vs_Full']]
    print(top_improved.to_string())
    
    print(f"\nTop 10 classes where LDA hurts most:")
    top_declined = pivot.nsmallest(10, 'LDA_vs_Full')[['Full', 'LDA', 'LDA_vs_Full']]
    print(top_declined.to_string())
    
    return df, pivot


if __name__ == "__main__":
    # Create results directory
    os.makedirs('results/tiny_imagenet', exist_ok=True)
    
    # Run backbone comparison
    print("Running Tiny ImageNet backbone comparison...")
    print("This tests if CIFAR-100 LDA findings generalize to larger dataset.\n")
    
    df = run_backbone_comparison()
    
    # Save results
    results_file = 'results/tiny_imagenet/backbone_comparison.csv'
    df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")
    
    # Analyze results
    analyze_results(df)
    
    # Run class-wise analysis on ResNet-18
    classwise_df, pivot_df = run_classwise_analysis('ResNet-18')
    classwise_file = 'results/tiny_imagenet/classwise_analysis.csv'
    classwise_df.to_csv(classwise_file, index=False)
    print(f"\nClass-wise analysis saved to {classwise_file}")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
