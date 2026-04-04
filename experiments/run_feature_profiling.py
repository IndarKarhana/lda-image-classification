"""
Feature Profiling Experiment
==============================
Runs the Feature Space Profiler on all backbone × dataset combinations
and saves comprehensive profiling results.

This produces the data for:
  - Table: Feature space characteristics per backbone
  - Figure: Eigenvalue spectrum comparison
  - Figure: Bhattacharyya distance heatmaps
  - Insight: Which backbones benefit most from LDA

Author: Research Study
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.extract_features_multi import (
    get_or_extract_cifar100,
    get_or_extract_tiny_imagenet,
    BACKBONES,
)
from reduction.feature_profiler import profile_feature_space

# ─── Configuration ───

DATASETS = {
    "cifar100": {
        "extract_fn": get_or_extract_cifar100,
        "n_classes": 100,
    },
    "tiny_imagenet": {
        "extract_fn": get_or_extract_tiny_imagenet,
        "n_classes": 200,
    },
}

RESULTS_DIR = "results/profiling"


def run_profiling():
    """Run feature space profiling on all backbone × dataset combinations."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []

    for backbone_name in BACKBONES:
        for dataset_name, ds_config in DATASETS.items():
            print(f"\n{'#'*70}")
            print(f"# Profiling: {backbone_name} × {dataset_name}")
            print(f"{'#'*70}")

            # Extract features
            t0 = time.perf_counter()
            X_train, y_train, X_test, y_test, dim = ds_config["extract_fn"](backbone_name)
            extraction_time = time.perf_counter() - t0
            print(f"  Feature extraction: {extraction_time:.1f}s")

            # Profile feature space
            t0 = time.perf_counter()
            profile = profile_feature_space(
                X_train, y_train,
                backbone=backbone_name,
                dataset=dataset_name,
                verbose=True,
            )
            profiling_time = time.perf_counter() - t0

            # Collect results
            result = {
                "backbone": backbone_name,
                "dataset": dataset_name,
                "feature_dim": dim,
                "n_classes": ds_config["n_classes"],
                "n_samples": X_train.shape[0],
                "dim_class_ratio": dim / ds_config["n_classes"],
                "sw_effective_rank": profile.sw_effective_rank,
                "sb_effective_rank": profile.sb_effective_rank,
                "sw_condition_number": profile.sw_condition_number,
                "sw_spectral_decay_rate": profile.sw_spectral_decay_rate,
                "dir_at_50": profile.dir_at_50,
                "dir_at_90": profile.dir_at_90,
                "dir_at_95": profile.dir_at_95,
                "optimal_components_95": profile.optimal_components_95,
                "covariance_heterogeneity": profile.covariance_heterogeneity,
                "mean_bhattacharyya": profile.mean_bhattacharyya,
                "min_bhattacharyya": profile.min_bhattacharyya,
                "fraction_non_gaussian": profile.fraction_non_gaussian,
                "feature_complexity_score": profile.feature_complexity_score,
                "lda_benefit_prediction": profile.lda_benefit_prediction,
                "extraction_time": extraction_time,
                "profiling_time": profiling_time,
            }
            all_results.append(result)

            # Save eigenvalue data for plotting
            eig_dir = os.path.join(RESULTS_DIR, "eigenvalues")
            os.makedirs(eig_dir, exist_ok=True)
            np.savez(
                os.path.join(eig_dir, f"{backbone_name}_{dataset_name}_eigenvalues.npz"),
                sw_eigenvalues=profile.sw_eigenvalues,
                sb_eigenvalues=profile.sb_eigenvalues,
                discriminant_eigenvalues=profile.discriminant_eigenvalues,
            )

            # Save Bhattacharyya distance matrix
            bhat_dir = os.path.join(RESULTS_DIR, "bhattacharyya")
            os.makedirs(bhat_dir, exist_ok=True)
            np.savez(
                os.path.join(bhat_dir, f"{backbone_name}_{dataset_name}_bhattacharyya.npz"),
                distance_matrix=profile.bhattacharyya_matrix,
                confused_pairs=np.array(profile.confused_pairs) if profile.confused_pairs else np.array([]),
            )

            # Save per-run CSV (crash-safe)
            df_run = pd.DataFrame([result])
            csv_path = os.path.join(RESULTS_DIR, "profiling_results.csv")
            df_run.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
            print(f"  Saved to {csv_path}")

    # Save complete results
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(os.path.join(RESULTS_DIR, "profiling_complete.csv"), index=False)

    # Pretty print summary
    print(f"\n\n{'='*80}")
    print("PROFILING SUMMARY")
    print(f"{'='*80}")
    for r in all_results:
        print(f"\n  {r['backbone']:15s} × {r['dataset']:15s}")
        print(f"    Dim: {r['feature_dim']:5d}  Classes: {r['n_classes']:3d}  Ratio: {r['dim_class_ratio']:.2f}")
        print(f"    FCS: {r['feature_complexity_score']:.4f}  Prediction: {r['lda_benefit_prediction']}")
        print(f"    Optimal d: {r['optimal_components_95']}  DIR@95: {r['dir_at_95']:.4f}")
        print(f"    Cov heterogeneity: {r['covariance_heterogeneity']:.4f}")
        print(f"    Sw condition: {r['sw_condition_number']:.2e}")
        print(f"    Non-Gaussian: {r['fraction_non_gaussian']:.1%}")

    print(f"\n✅ All profiling results saved to {RESULTS_DIR}/")
    return df_all


if __name__ == "__main__":
    run_profiling()
