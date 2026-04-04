"""
DG-LDA Comprehensive Experiment
==================================
Compares DG-LDA (full, regularized, CW-only) against baselines
(vanilla LDA, PCA, Full features) across all backbones and datasets.

Experiment matrix:
  - 4 backbones × 2 datasets × 6 methods × 3 seeds = 144 experiments
  - Methods: dglda_full, dglda_regularized, dglda_cw, vanilla_lda, pca, full
  - Every run records: accuracy, fit_time, transform_time, eval_time, n_components

Author: Research Study
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.extract_features_multi import (
    get_or_extract_cifar100,
    get_or_extract_tiny_imagenet,
    BACKBONES,
)
from reduction.dg_lda import DGLDA
from reduction.regularized_lda import RegularizedLDA
from reduction.cw_lda import ConfusionWeightedLDA

# ─── Configuration ───

SEEDS = [42, 123, 456]

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

RESULTS_DIR = "results/dglda"


def evaluate_accuracy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
) -> tuple:
    """Train LogisticRegression and evaluate accuracy. Returns (accuracy, train_time, eval_time)."""
    clf = LogisticRegression(
        solver="lbfgs", max_iter=1000, n_jobs=-1, random_state=seed, C=1.0
    )
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    acc = clf.score(X_test, y_test)
    eval_time = time.perf_counter() - t0

    return acc, train_time, eval_time


def run_single_experiment(
    method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    backbone: str,
    dataset: str,
    seed: int,
) -> dict:
    """
    Run a single experiment for one method.

    Returns dict with all metrics.
    """
    np.random.seed(seed)
    N, D = X_train.shape

    # Standardize features
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    result = {
        "method": method,
        "backbone": backbone,
        "dataset": dataset,
        "seed": seed,
        "feature_dim": D,
        "n_classes": n_classes,
    }

    if method == "full":
        # No reduction — full feature baseline
        result["n_components"] = D
        result["reduction_time"] = 0.0
        result["transform_time"] = 0.0
        acc, clf_time, eval_time = evaluate_accuracy(X_tr, y_train, X_te, y_test, seed)

    elif method == "vanilla_lda":
        # Standard sklearn LDA
        n_comp = n_classes - 1
        t0 = time.perf_counter()
        lda = LinearDiscriminantAnalysis(n_components=n_comp)
        lda.fit(X_tr, y_train)
        reduction_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        X_tr_r = lda.transform(X_tr)
        X_te_r = lda.transform(X_te)
        transform_time = time.perf_counter() - t0

        result["n_components"] = n_comp
        result["reduction_time"] = reduction_time
        result["transform_time"] = transform_time
        acc, clf_time, eval_time = evaluate_accuracy(X_tr_r, y_train, X_te_r, y_test, seed)

    elif method == "pca":
        # PCA baseline (same n_components as vanilla LDA for fair comparison)
        n_comp = n_classes - 1
        t0 = time.perf_counter()
        pca = PCA(n_components=n_comp, random_state=seed)
        pca.fit(X_tr)
        reduction_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        X_tr_r = pca.transform(X_tr)
        X_te_r = pca.transform(X_te)
        transform_time = time.perf_counter() - t0

        result["n_components"] = n_comp
        result["reduction_time"] = reduction_time
        result["transform_time"] = transform_time
        acc, clf_time, eval_time = evaluate_accuracy(X_tr_r, y_train, X_te_r, y_test, seed)

    elif method == "dglda_full":
        # Full DG-LDA: profiling + adaptive components + CW + regularization
        t0 = time.perf_counter()
        dglda = DGLDA(mode="full", verbose=False)
        X_tr_r = dglda.fit_transform(X_tr, y_train, backbone=backbone, dataset=dataset)
        reduction_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        X_te_r = dglda.transform(X_te)
        transform_time = time.perf_counter() - t0

        result["n_components"] = dglda.result_.selected_components
        result["reduction_time"] = reduction_time
        result["transform_time"] = transform_time
        result["shrinkage_alpha"] = dglda.result_.shrinkage_alpha
        result["selection_strategy"] = dglda.result_.selection_strategy
        result["fcs"] = dglda.result_.profile.feature_complexity_score
        acc, clf_time, eval_time = evaluate_accuracy(X_tr_r, y_train, X_te_r, y_test, seed)

    elif method == "dglda_regularized":
        # Regularized LDA only (no CW, no adaptive components)
        n_comp = n_classes - 1
        t0 = time.perf_counter()
        dglda = DGLDA(mode="regularized", n_components=n_comp, verbose=False)
        X_tr_r = dglda.fit_transform(X_tr, y_train, backbone=backbone, dataset=dataset)
        reduction_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        X_te_r = dglda.transform(X_te)
        transform_time = time.perf_counter() - t0

        result["n_components"] = n_comp
        result["reduction_time"] = reduction_time
        result["transform_time"] = transform_time
        result["shrinkage_alpha"] = dglda.result_.shrinkage_alpha
        acc, clf_time, eval_time = evaluate_accuracy(X_tr_r, y_train, X_te_r, y_test, seed)

    elif method == "dglda_cw":
        # CW-LDA only (confusion-weighted, no regularization adaptation)
        n_comp = n_classes - 1
        t0 = time.perf_counter()
        dglda = DGLDA(mode="cw_only", n_components=n_comp, verbose=False)
        X_tr_r = dglda.fit_transform(X_tr, y_train, backbone=backbone, dataset=dataset)
        reduction_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        X_te_r = dglda.transform(X_te)
        transform_time = time.perf_counter() - t0

        result["n_components"] = n_comp
        result["reduction_time"] = reduction_time
        result["transform_time"] = transform_time
        acc, clf_time, eval_time = evaluate_accuracy(X_tr_r, y_train, X_te_r, y_test, seed)

    else:
        raise ValueError(f"Unknown method: {method}")

    result["accuracy"] = acc
    result["classifier_train_time"] = clf_time
    result["eval_time"] = eval_time
    result["total_time"] = result.get("reduction_time", 0) + result.get("transform_time", 0) + clf_time + eval_time

    return result


def run_all_experiments():
    """Run the complete DG-LDA experiment suite."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "dglda_results.csv")

    methods = ["full", "vanilla_lda", "pca", "dglda_full", "dglda_regularized", "dglda_cw"]
    total = len(BACKBONES) * len(DATASETS) * len(methods) * len(SEEDS)
    count = 0

    print(f"\n{'='*70}")
    print(f"DG-LDA EXPERIMENT SUITE")
    print(f"  Backbones: {list(BACKBONES.keys())}")
    print(f"  Datasets: {list(DATASETS.keys())}")
    print(f"  Methods: {methods}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total experiments: {total}")
    print(f"{'='*70}")

    all_results = []

    for backbone_name in BACKBONES:
        for dataset_name, ds_config in DATASETS.items():
            print(f"\n{'─'*50}")
            print(f"  {backbone_name} × {dataset_name}")
            print(f"{'─'*50}")

            # Extract features once per backbone×dataset
            X_train, y_train, X_test, y_test, dim = ds_config["extract_fn"](backbone_name)

            for method in methods:
                for seed in SEEDS:
                    count += 1
                    print(f"  [{count}/{total}] {method} (seed={seed})...", end=" ", flush=True)

                    t0 = time.perf_counter()
                    result = run_single_experiment(
                        method=method,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        n_classes=ds_config["n_classes"],
                        backbone=backbone_name,
                        dataset=dataset_name,
                        seed=seed,
                    )
                    wall_time = time.perf_counter() - t0

                    print(f"acc={result['accuracy']:.4f}  ({wall_time:.1f}s)")

                    all_results.append(result)

                    # Save incrementally (crash-safe)
                    df_row = pd.DataFrame([result])
                    df_row.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)

    # Save complete results
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(os.path.join(RESULTS_DIR, "dglda_complete.csv"), index=False)

    # Print summary
    print(f"\n\n{'='*80}")
    print("RESULTS SUMMARY (mean accuracy across seeds)")
    print(f"{'='*80}")

    summary = df_all.groupby(["backbone", "dataset", "method"])["accuracy"].agg(["mean", "std"]).reset_index()
    summary.columns = ["backbone", "dataset", "method", "mean_acc", "std_acc"]

    for dataset_name in DATASETS:
        print(f"\n  Dataset: {dataset_name}")
        print(f"  {'Backbone':<15} {'Method':<20} {'Accuracy':>10} {'± Std':>8}")
        print(f"  {'-'*55}")
        ds_data = summary[summary["dataset"] == dataset_name].sort_values(
            ["backbone", "mean_acc"], ascending=[True, False]
        )
        for _, row in ds_data.iterrows():
            print(f"  {row['backbone']:<15} {row['method']:<20} {row['mean_acc']:>10.4f} {row['std_acc']:>7.4f}")

    # Compute DG-LDA improvements over vanilla LDA
    print(f"\n\n{'='*80}")
    print("DG-LDA IMPROVEMENT OVER VANILLA LDA")
    print(f"{'='*80}")

    for dataset_name in DATASETS:
        print(f"\n  Dataset: {dataset_name}")
        for backbone_name in BACKBONES:
            mask = (summary["backbone"] == backbone_name) & (summary["dataset"] == dataset_name)
            sub = summary[mask].set_index("method")

            if "vanilla_lda" in sub.index and "dglda_full" in sub.index:
                vanilla = sub.loc["vanilla_lda", "mean_acc"]
                dglda = sub.loc["dglda_full", "mean_acc"]
                improvement = (dglda - vanilla) * 100
                print(f"    {backbone_name:<15}: vanilla={vanilla:.4f}  DG-LDA={dglda:.4f}  Δ={improvement:+.2f}pp")

    print(f"\n✅ All results saved to {RESULTS_DIR}/")
    return df_all


if __name__ == "__main__":
    run_all_experiments()
