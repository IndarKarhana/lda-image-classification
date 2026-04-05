"""
MLP Classifier Benchmark
========================
Tests whether LDA's advantage holds with a non-linear classifier (2-layer MLP),
not just logistic regression. Uses the same features and reduction methods.

Protocol:
  - Same frozen features as main experiment (cached .npz)
  - Same 10 reduction methods
  - 2 classifiers: LogisticRegression (existing) vs MLP (new)
  - 5 seeds for statistical comparison
  - Reports accuracy and wall-clock time for each

Usage:
  python experiments/run_mlp_benchmark.py                          # resnet18 + resnet50 × cifar100
  python experiments/run_mlp_benchmark.py --backbone all           # all 4 backbones
  python experiments/run_mlp_benchmark.py --dataset tiny_imagenet  # Tiny ImageNet
  python experiments/run_mlp_benchmark.py --backbone all --dataset cifar100 --dataset tiny_imagenet

Author: Research Study
"""

import os
import sys
import time
import warnings
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.extract_features_multi import (
    get_or_extract_cifar100,
    get_or_extract_tiny_imagenet,
    BACKBONES,
)

SEEDS = [42, 123, 456, 789, 1024]


# ═══════════════════════════════════════════════════════════════════════
# Classifiers
# ═══════════════════════════════════════════════════════════════════════

def make_classifiers(seed: int) -> Dict[str, Any]:
    """Return dict of classifier_name → sklearn classifier."""
    return {
        "LogReg": LogisticRegression(
            solver="lbfgs", max_iter=5000, C=1.0, random_state=seed
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=seed,
            batch_size=256,
        ),
    }


def evaluate_classifier(clf, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
    """Train classifier, return (accuracy%, time_seconds)."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    t0 = time.perf_counter()
    clf.fit(X_tr, y_train)
    acc = clf.score(X_te, y_test) * 100
    return acc, time.perf_counter() - t0


# ═══════════════════════════════════════════════════════════════════════
# Reduction Methods (simplified set — the key ones)
# ═══════════════════════════════════════════════════════════════════════

def reduce_full(X_train, y_train, X_test, n_components):
    return X_train.copy(), X_test.copy(), X_train.shape[1], 0.0

def reduce_pca(X_train, y_train, X_test, n_components):
    t0 = time.perf_counter()
    pca = PCA(n_components=n_components, random_state=42)
    X_tr = pca.fit_transform(X_train)
    X_te = pca.transform(X_test)
    return X_tr, X_te, n_components, time.perf_counter() - t0

def reduce_lda(X_train, y_train, X_test, n_components):
    t0 = time.perf_counter()
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_tr = lda.fit_transform(X_train, y_train)
    X_te = lda.transform(X_test)
    return X_tr, X_te, n_components, time.perf_counter() - t0

def reduce_rlda(X_train, y_train, X_test, n_components):
    t0 = time.perf_counter()
    lda = LinearDiscriminantAnalysis(
        n_components=n_components, solver="eigen", shrinkage="auto"
    )
    X_tr = lda.fit_transform(X_train, y_train)
    X_te = lda.transform(X_test)
    return X_tr, X_te, n_components, time.perf_counter() - t0


METHODS = {
    "Full": reduce_full,
    "PCA": reduce_pca,
    "LDA": reduce_lda,
    "R-LDA": reduce_rlda,
}


# ═══════════════════════════════════════════════════════════════════════
# Main Runner
# ═══════════════════════════════════════════════════════════════════════

def run_mlp_benchmark(backbone: str, dataset: str) -> pd.DataFrame:
    """Run MLP vs LogReg benchmark for one backbone × dataset."""
    feature_dim = BACKBONES[backbone]["feature_dim"]

    if dataset == "cifar100":
        n_classes = 100
        X_train, y_train, X_test, y_test, _ = get_or_extract_cifar100(backbone)
        dataset_label = "CIFAR-100"
    elif dataset == "tiny_imagenet":
        n_classes = 200
        X_train, y_train, X_test, y_test, _ = get_or_extract_tiny_imagenet(backbone)
        dataset_label = "Tiny ImageNet"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    n_components = n_classes - 1

    # Pre-standardize features (same as main benchmark)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"\n{'═' * 78}")
    print(f"  MLP BENCHMARK — {backbone.upper()} × {dataset_label}")
    print(f"  Features: {feature_dim}D → {n_components}D | {len(SEEDS)} seeds")
    print(f"{'═' * 78}")

    results = []

    for method_name, reduce_fn in METHODS.items():
        # Reduce once (deterministic)
        X_tr_red, X_te_red, dim, t_reduce = reduce_fn(
            X_train_s, y_train, X_test_s, n_components
        )

        for clf_name in ["LogReg", "MLP"]:
            accs = []
            times = []
            for seed in SEEDS:
                classifiers = make_classifiers(seed)
                clf = classifiers[clf_name]
                acc, t_clf = evaluate_classifier(
                    clf, X_tr_red, y_train, X_te_red, y_test
                )
                accs.append(acc)
                times.append(t_clf)

            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            mean_time = np.mean(times)

            results.append({
                "backbone": backbone,
                "dataset": dataset,
                "method": method_name,
                "classifier": clf_name,
                "accuracy_mean": round(mean_acc, 2),
                "accuracy_std": round(std_acc, 2),
                "dim": dim,
                "time_reduce": round(t_reduce, 2),
                "time_classify": round(mean_time, 2),
                "time_total": round(t_reduce + mean_time, 2),
                "n_seeds": len(SEEDS),
            })

            icon = "✅" if mean_acc > 0 else "❌"
            print(f"  {icon} {method_name:<8s} + {clf_name:<8s}  "
                  f"{mean_acc:6.2f}±{std_acc:.2f}%  "
                  f"{dim:5d}D  {t_reduce + mean_time:7.1f}s")

    df = pd.DataFrame(results)

    # Summary: LDA gain over Full for each classifier
    print(f"\n  {'─' * 60}")
    print(f"  SUMMARY — Does LDA still help with MLP?")
    for clf_name in ["LogReg", "MLP"]:
        clf_df = df[df["classifier"] == clf_name]
        full_acc = clf_df[clf_df["method"] == "Full"]["accuracy_mean"].values[0]
        lda_acc = clf_df[clf_df["method"] == "LDA"]["accuracy_mean"].values[0]
        pca_acc = clf_df[clf_df["method"] == "PCA"]["accuracy_mean"].values[0]
        print(f"    {clf_name}: Full={full_acc:.2f}%  PCA={pca_acc:.2f}%  "
              f"LDA={lda_acc:.2f}%  (LDA-Full={lda_acc-full_acc:+.2f}%)")

    return df


def main():
    parser = argparse.ArgumentParser(description="MLP vs LogReg classifier benchmark")
    parser.add_argument("--backbone", type=str, nargs="+", default=None,
                        help="Backbone name(s) or 'all'. Default: resnet18 resnet50")
    parser.add_argument("--dataset", type=str, nargs="+",
                        default=["cifar100"],
                        choices=["cifar100", "tiny_imagenet"])
    args = parser.parse_args()

    if args.backbone is None:
        backbones = ["resnet18", "resnet50"]
    elif "all" in args.backbone:
        backbones = list(BACKBONES.keys())
    else:
        backbones = args.backbone

    all_dfs = []
    for bb in backbones:
        for ds in args.dataset:
            df = run_mlp_benchmark(bb, ds)
            all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    os.makedirs("results/mlp_benchmark", exist_ok=True)
    csv_path = "results/mlp_benchmark/mlp_vs_logreg.csv"
    combined.to_csv(csv_path, index=False)
    print(f"\n  📄 Results saved to: {csv_path}")

    # Cross-config summary
    print(f"\n{'═' * 78}")
    print("  CROSS-CONFIG: LDA gain over Full by classifier")
    print(f"{'═' * 78}")
    for clf_name in ["LogReg", "MLP"]:
        gains = []
        clf_df = combined[combined["classifier"] == clf_name]
        for _, grp in clf_df.groupby(["backbone", "dataset"]):
            full = grp[grp["method"] == "Full"]["accuracy_mean"].values
            lda = grp[grp["method"] == "LDA"]["accuracy_mean"].values
            if len(full) > 0 and len(lda) > 0:
                gains.append(lda[0] - full[0])
        if gains:
            print(f"  {clf_name}: LDA-Full gains = {[f'{g:+.2f}' for g in gains]}")
            print(f"         Mean gain: {np.mean(gains):+.2f}%, "
                  f"Min: {min(gains):+.2f}%, Max: {max(gains):+.2f}%")
            print(f"         LDA helps in {sum(1 for g in gains if g > 0)}/{len(gains)} configs")


if __name__ == "__main__":
    main()
