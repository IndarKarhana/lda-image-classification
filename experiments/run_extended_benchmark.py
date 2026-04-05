"""
Extended Benchmark — 10 Methods × 2 Classifiers × 6 Backbones × 3 Datasets
=============================================================================

Master experiment script for the extended study adding:
  1. MLP classifier (alongside LogReg)
  2. ViT-B/16 and DINOv2 ViT-S/14 backbones
  3. CUB-200-2011 fine-grained classification dataset

Runs all 10 reduction methods from run_academic_benchmark.py with both
LogisticRegression and MLPClassifier, across all backbone × dataset combos.

Multi-seed (5 seeds) for statistical comparison.

Usage:
  # Quick test — one backbone × one dataset
  python experiments/run_extended_benchmark.py --backbone resnet50 --dataset cifar100

  # Full run — all backbones × all datasets (for GCP)
  python experiments/run_extended_benchmark.py --backbone all --dataset all

  # Only new additions (ViT + DINOv2 on all datasets)
  python experiments/run_extended_benchmark.py --backbone vit_b16 dinov2_vits14 --dataset all

  # CUB-200 only
  python experiments/run_extended_benchmark.py --backbone all --dataset cub200

  # MLP comparison on existing cached features
  python experiments/run_extended_benchmark.py --backbone resnet18 resnet50 --dataset cifar100 --classifier mlp

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
from sklearn.covariance import ledoit_wolf
from scipy.sparse import coo_matrix
from scipy.linalg import eigh as scipy_eigh
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.extract_features_multi import (
    get_or_extract_cifar100,
    get_or_extract_tiny_imagenet,
    get_or_extract_cub200,
    BACKBONES,
)

SEEDS = [42, 123, 456, 789, 1024]

DATASET_CONFIG = {
    "cifar100":       {"n_classes": 100, "label": "CIFAR-100",      "load_fn": get_or_extract_cifar100},
    "tiny_imagenet":  {"n_classes": 200, "label": "Tiny ImageNet",  "load_fn": get_or_extract_tiny_imagenet},
    "cub200":         {"n_classes": 200, "label": "CUB-200-2011",   "load_fn": get_or_extract_cub200},
}


# ═══════════════════════════════════════════════════════════════════════
# Classifiers
# ═══════════════════════════════════════════════════════════════════════

def make_logreg(seed: int) -> LogisticRegression:
    return LogisticRegression(solver="lbfgs", max_iter=5000, C=1.0, random_state=seed)


def make_mlp(seed: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=seed,
        batch_size=256,
    )


CLASSIFIERS = {
    "LogReg": make_logreg,
    "MLP":    make_mlp,
}


# ═══════════════════════════════════════════════════════════════════════
# Shared Utilities (same as run_academic_benchmark.py)
# ═══════════════════════════════════════════════════════════════════════

def evaluate(clf, X_train: np.ndarray, y_train: np.ndarray,
             X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
    """Post-projection standardize → train classifier → return (accuracy%, time_s)."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    t0 = time.perf_counter()
    clf.fit(X_tr, y_train)
    acc = clf.score(X_te, y_test) * 100
    return acc, time.perf_counter() - t0


def solve_whitened_eigen(Sb, Sw_inv_sqrt, n_components):
    Sb_white = Sw_inv_sqrt.T @ Sb @ Sw_inv_sqrt
    eigenvalues, eigvecs_white = np.linalg.eigh(Sb_white)
    eigenvectors = Sw_inv_sqrt @ eigvecs_white
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, idx[:n_components]], eigenvalues[idx[:n_components]]


# ═══════════════════════════════════════════════════════════════════════
# Reduction Methods (all 10 — copied from run_academic_benchmark.py)
# ═══════════════════════════════════════════════════════════════════════

def method_full(X_train, y_train, X_test, n_components, **kw):
    return X_train.copy(), X_test.copy(), X_train.shape[1], 0.0


def method_pca(X_train, y_train, X_test, n_components, **kw):
    t0 = time.perf_counter()
    pca = PCA(n_components=n_components, random_state=42)
    X_tr = pca.fit_transform(X_train)
    X_te = pca.transform(X_test)
    return X_tr, X_te, n_components, time.perf_counter() - t0


def method_lda(X_train, y_train, X_test, n_components, **kw):
    t0 = time.perf_counter()
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_tr = lda.fit_transform(X_train, y_train)
    X_te = lda.transform(X_test)
    return X_tr, X_te, n_components, time.perf_counter() - t0


def method_pca_lda(X_train, y_train, X_test, n_components, **kw):
    t0 = time.perf_counter()
    N = X_train.shape[0]
    C = len(np.unique(y_train))
    pca_dim = min(N - C, max(4 * n_components, 500), X_train.shape[1])
    pca = PCA(n_components=pca_dim, random_state=42)
    X_tr_pca = pca.fit_transform(X_train)
    X_te_pca = pca.transform(X_test)
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_tr = lda.fit_transform(X_tr_pca, y_train)
    X_te = lda.transform(X_te_pca)
    return X_tr, X_te, n_components, time.perf_counter() - t0


def method_rlda(X_train, y_train, X_test, n_components, **kw):
    t0 = time.perf_counter()
    lda = LinearDiscriminantAnalysis(
        n_components=n_components, solver="eigen", shrinkage="auto"
    )
    X_tr = lda.fit_transform(X_train, y_train)
    X_te = lda.transform(X_test)
    return X_tr, X_te, n_components, time.perf_counter() - t0


def method_lfda(X_train, y_train, X_test, n_components, **kw):
    """LFDA (Sugiyama 2007) — Local Fisher Discriminant Analysis."""
    t0 = time.perf_counter()
    k = kw.get("k", 7)
    pca_preprocess = kw.get("pca_preprocess", 300)
    N, D = X_train.shape
    classes = np.unique(y_train)
    C = len(classes)

    pca_pre = None
    if D > pca_preprocess:
        pca_pre = PCA(n_components=pca_preprocess, random_state=42)
        X_work = pca_pre.fit_transform(X_train)
        X_test_work = pca_pre.transform(X_test)
    else:
        X_work = X_train.copy()
        X_test_work = X_test.copy()

    D_work = X_work.shape[1]
    all_rows, all_cols, all_vals = [], [], []

    for c in classes:
        mask = y_train == c
        idx = np.where(mask)[0]
        X_c = X_work[idx]
        n_c = len(idx)
        k_actual = min(k, n_c - 1)
        if k_actual < 1:
            continue
        from scipy.spatial import cKDTree
        tree = cKDTree(X_c)
        dists, nbrs = tree.query(X_c, k=k_actual + 1)
        sigma = np.median(dists[:, -1]) + 1e-10
        affinities = np.exp(-dists[:, 1:]**2 / (2 * sigma**2)) / n_c
        rows_local = np.repeat(np.arange(n_c), k_actual)
        cols_local = nbrs[:, 1:].flatten()
        rows_global = idx[rows_local]
        cols_global = idx[cols_local]
        vals_flat = affinities.flatten()
        all_rows.extend([rows_global, cols_global])
        all_cols.extend([cols_global, rows_global])
        all_vals.extend([vals_flat, vals_flat])

    all_rows = np.concatenate(all_rows)
    all_cols = np.concatenate(all_cols)
    all_vals = np.concatenate(all_vals)
    W_local = coo_matrix((all_vals, (all_rows, all_cols)), shape=(N, N)).tocsr()

    D_diag = np.array(W_local.sum(axis=1)).flatten()
    XtDX = (X_work * D_diag[:, np.newaxis]).T @ X_work
    WX = np.asarray(W_local.dot(X_work))
    XtWX = X_work.T @ WX
    Sw_local = (XtDX - XtWX) / N

    mean_work = X_work.mean(axis=0)
    X_c = X_work - mean_work
    St = X_c.T @ X_c / N
    Sb_local = St - Sw_local

    reg = 1e-4 * np.trace(Sw_local) / D_work
    Sw_local_reg = Sw_local + reg * np.eye(D_work)

    n_comp = min(n_components, C - 1, D_work - 1)
    eigenvalues, eigenvectors = scipy_eigh(
        Sb_local, Sw_local_reg,
        subset_by_index=[D_work - n_comp, D_work - 1]
    )
    idx_sort = np.argsort(eigenvalues)[::-1]
    W_proj = eigenvectors[:, idx_sort]

    X_tr_out = (X_work - mean_work) @ W_proj
    X_te_out = (X_test_work - mean_work) @ W_proj
    return X_tr_out, X_te_out, n_comp, time.perf_counter() - t0


def method_nca(X_train, y_train, X_test, n_components, **kw):
    """NCA (Goldberger et al. 2004) — Neighborhood Component Analysis."""
    from sklearn.neighbors import NeighborhoodComponentsAnalysis
    t0 = time.perf_counter()
    pca_preprocess = kw.get("pca_preprocess", 200)
    max_fit_samples = kw.get("max_fit_samples", 10000)
    max_iter = kw.get("max_iter", 100)
    N, D = X_train.shape

    pca_pre = None
    if D > pca_preprocess:
        pca_pre = PCA(n_components=pca_preprocess, random_state=42)
        X_work = pca_pre.fit_transform(X_train)
        X_test_work = pca_pre.transform(X_test)
    else:
        X_work = X_train.copy()
        X_test_work = X_test.copy()

    if N > max_fit_samples:
        rng = np.random.RandomState(42)
        classes = np.unique(y_train)
        samples_per_class = max_fit_samples // len(classes)
        fit_idx = []
        for c in classes:
            c_idx = np.where(y_train == c)[0]
            n_take = min(samples_per_class, len(c_idx))
            fit_idx.extend(rng.choice(c_idx, size=n_take, replace=False))
        fit_idx = np.array(fit_idx)
        X_fit = X_work[fit_idx]
        y_fit = y_train[fit_idx]
    else:
        X_fit = X_work
        y_fit = y_train

    n_comp = min(n_components, X_work.shape[1] - 1)
    nca = NeighborhoodComponentsAnalysis(
        n_components=n_comp, max_iter=max_iter, random_state=42, verbose=0,
    )
    nca.fit(X_fit, y_fit)
    X_tr_out = nca.transform(X_work)
    X_te_out = nca.transform(X_test_work)
    return X_tr_out, X_te_out, n_comp, time.perf_counter() - t0


def method_rda(X_train, y_train, X_test, n_components, **kw):
    """RDA (Residual Discriminant Augmentation) — OURS."""
    n_residual = kw.get("n_residual", 20)
    t0 = time.perf_counter()
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    W = lda.scalings_[:, :n_components]
    Q, _ = np.linalg.qr(W)
    X_train_res = X_train - (X_train @ Q) @ Q.T
    X_test_res = X_test - (X_test @ Q) @ Q.T
    n_res = min(n_residual, X_train.shape[1] - n_components - 1)
    if n_res < 1:
        n_res = 1
    pca_res = PCA(n_components=n_res, random_state=42)
    X_train_res_pca = pca_res.fit_transform(X_train_res)
    X_test_res_pca = pca_res.transform(X_test_res)
    X_train_out = np.hstack([X_train_lda, X_train_res_pca])
    X_test_out = np.hstack([X_test_lda, X_test_res_pca])
    return X_train_out, X_test_out, X_train_out.shape[1], time.perf_counter() - t0


def method_dsb(X_train, y_train, X_test, n_components, **kw):
    """DSB (Discriminant Subspace Boosting) — OURS."""
    n_rounds = kw.get("n_rounds", 2)
    t0 = time.perf_counter()
    N, D = X_train.shape
    classes = np.unique(y_train)
    C = len(classes)
    n_comp = min(n_components, C - 1)
    sample_weights = np.ones(N, dtype=np.float64) / N
    all_train_features = []
    all_test_features = []

    for round_idx in range(n_rounds):
        global_mean_w = np.average(X_train, weights=sample_weights, axis=0)
        Sw = np.zeros((D, D), dtype=np.float64)
        Sb = np.zeros((D, D), dtype=np.float64)
        for i, c in enumerate(classes):
            mask = y_train == c
            w_c = sample_weights[mask]
            w_c_sum = w_c.sum()
            if w_c_sum < 1e-10:
                continue
            mc = np.average(X_train[mask], weights=w_c, axis=0)
            diff = X_train[mask] - mc
            Sw += (diff * w_c[:, np.newaxis]).T @ diff
            dm = (mc - global_mean_w).reshape(-1, 1)
            Sb += w_c_sum * (dm @ dm.T)
        alpha = 0.05
        target = np.trace(Sw) / D * np.eye(D) if np.trace(Sw) > 0 else np.eye(D)
        Sw_reg = (1 - alpha) * Sw + alpha * target
        eigvals_sw, eigvecs_sw = np.linalg.eigh(Sw_reg)
        eigvals_sw = np.maximum(eigvals_sw, 1e-10)
        Sw_inv_sqrt = eigvecs_sw * (1.0 / np.sqrt(eigvals_sw))[np.newaxis, :]
        W, _ = solve_whitened_eigen(Sb, Sw_inv_sqrt, n_comp)
        X_train_proj = (X_train - global_mean_w) @ W
        X_test_proj = (X_test - global_mean_w) @ W
        all_train_features.append(X_train_proj)
        all_test_features.append(X_test_proj)
        if round_idx < n_rounds - 1:
            clf_boost = LogisticRegression(solver="lbfgs", max_iter=500, random_state=42)
            clf_boost.fit(X_train_proj, y_train)
            pred = clf_boost.predict(X_train_proj)
            wrong = (pred != y_train).astype(float)
            error_rate = np.average(wrong, weights=sample_weights)
            if 0 < error_rate < 0.5:
                beta = error_rate / (1 - error_rate)
                sample_weights *= np.exp(wrong * np.log(1 / beta + 1e-10))
                sample_weights /= sample_weights.sum()

    X_train_concat = np.hstack(all_train_features)
    X_test_concat = np.hstack(all_test_features)
    if X_train_concat.shape[1] > n_comp:
        pca = PCA(n_components=n_comp, random_state=42)
        X_train_out = pca.fit_transform(X_train_concat)
        X_test_out = pca.transform(X_test_concat)
        out_dim = n_comp
    else:
        X_train_out = X_train_concat
        X_test_out = X_test_concat
        out_dim = X_train_concat.shape[1]
    return X_train_out, X_test_out, out_dim, time.perf_counter() - t0


def method_rda_smd(X_train, y_train, X_test, n_components, **kw):
    """RDA+SMD (Residual + Spectral Margin Discriminants) — OURS."""
    n_residual = kw.get("n_residual", 20)
    margin_fraction = kw.get("margin_fraction", 0.3)
    t0 = time.perf_counter()
    classes = np.unique(y_train)
    C = len(classes)
    N, D = X_train.shape
    n_comp = min(n_components, C - 1)
    global_mean = X_train.mean(axis=0)

    class_means = np.zeros((C, D), dtype=np.float64)
    class_counts = np.zeros(C)
    X_centered = np.zeros_like(X_train, dtype=np.float64)
    for i, c in enumerate(classes):
        mask = y_train == c
        class_means[i] = X_train[mask].mean(axis=0)
        class_counts[i] = mask.sum()
        X_centered[mask] = X_train[mask] - class_means[i]

    Sw = X_centered.T @ X_centered / N
    subsample_size = min(10000, N)
    _, alpha = ledoit_wolf(X_centered[:subsample_size])
    target_mat = np.trace(Sw) / D * np.eye(D)
    Sw_reg = (1 - alpha) * Sw + alpha * target_mat
    eigvals, eigvecs = np.linalg.eigh(Sw_reg)
    eigvals = np.maximum(eigvals, 1e-10)
    Sw_inv_sqrt = eigvecs * (1.0 / np.sqrt(eigvals))[np.newaxis, :]
    diffs = class_means - global_mean
    Sb = (diffs * class_counts[:, np.newaxis]).T @ diffs / N

    W1, _ = solve_whitened_eigen(Sb, Sw_inv_sqrt, n_comp)
    class_means_proj = (class_means - global_mean) @ W1
    n_total_pairs = C * (C - 1) // 2
    n_hard = max(1, int(n_total_pairs * margin_fraction))
    pair_dists = []
    for i in range(C):
        for j in range(i + 1, C):
            dist = np.linalg.norm(class_means_proj[i] - class_means_proj[j])
            pair_dists.append((i, j, dist))
    pair_dists.sort(key=lambda x: x[2])

    Sb_margin = np.zeros((D, D), dtype=np.float64)
    for i, j, dist in pair_dists[:n_hard]:
        w = 1.0 / (dist + 1e-6)
        dm = (class_means[i] - class_means[j]).reshape(-1, 1)
        Sb_margin += w * (dm @ dm.T)
    for i, j, dist in pair_dists[n_hard:]:
        w = 0.1 / (dist + 1e-6)
        dm = (class_means[i] - class_means[j]).reshape(-1, 1)
        Sb_margin += w * (dm @ dm.T)
    Sb_margin /= N

    W_smd, _ = solve_whitened_eigen(Sb_margin, Sw_inv_sqrt, n_comp)
    X_train_smd = (X_train - global_mean) @ W_smd
    X_test_smd = (X_test - global_mean) @ W_smd

    lda = LinearDiscriminantAnalysis(n_components=n_comp)
    lda.fit(X_train, y_train)
    W_lda = lda.scalings_[:, :n_comp]
    Q, _ = np.linalg.qr(W_lda)
    X_train_res = X_train - (X_train @ Q) @ Q.T
    X_test_res = X_test - (X_test @ Q) @ Q.T
    n_res = min(n_residual, X_train.shape[1] - n_comp - 1)
    if n_res < 1:
        n_res = 1
    pca_res = PCA(n_components=n_res, random_state=42)
    X_train_res_pca = pca_res.fit_transform(X_train_res)
    X_test_res_pca = pca_res.transform(X_test_res)
    X_train_out = np.hstack([X_train_smd, X_train_res_pca])
    X_test_out = np.hstack([X_test_smd, X_test_res_pca])
    return X_train_out, X_test_out, X_train_out.shape[1], time.perf_counter() - t0


# ═══════════════════════════════════════════════════════════════════════
# Method Registry
# ═══════════════════════════════════════════════════════════════════════

def get_method_configs(n_components: int, feature_dim: int) -> List[Dict]:
    """Build ordered list of 10 methods."""
    n_residual = 20 if feature_dim <= 576 else (25 if feature_dim <= 1280 else 30)
    return [
        {"name": "Full",               "fn": method_full,    "category": "control",   "kwargs": {}},
        {"name": f"PCA",               "fn": method_pca,     "category": "control",   "kwargs": {}},
        {"name": f"LDA",               "fn": method_lda,     "category": "classical", "kwargs": {}},
        {"name": f"PCA+LDA",           "fn": method_pca_lda, "category": "classical", "kwargs": {}},
        {"name": f"R-LDA",             "fn": method_rlda,    "category": "classical", "kwargs": {}},
        {"name": f"LFDA",              "fn": method_lfda,    "category": "academic",
         "kwargs": {"k": 7, "pca_preprocess": 300}},
        {"name": f"NCA",               "fn": method_nca,     "category": "academic",
         "kwargs": {"pca_preprocess": 200, "max_fit_samples": 10000, "max_iter": 100}},
        {"name": f"RDA",               "fn": method_rda,     "category": "ours",
         "kwargs": {"n_residual": n_residual}},
        {"name": f"DSB",               "fn": method_dsb,     "category": "ours",
         "kwargs": {"n_rounds": 2}},
        {"name": f"RDA+SMD",           "fn": method_rda_smd, "category": "ours",
         "kwargs": {"n_residual": n_residual, "margin_fraction": 0.3}},
    ]


# ═══════════════════════════════════════════════════════════════════════
# Main Runner
# ═══════════════════════════════════════════════════════════════════════

def run_config(backbone: str, dataset: str, clf_names: List[str],
               seeds: List[int]) -> pd.DataFrame:
    """Run full benchmark for one backbone × dataset × classifier(s) × seeds."""
    feature_dim = BACKBONES[backbone]["feature_dim"]
    ds_cfg = DATASET_CONFIG[dataset]
    n_classes = ds_cfg["n_classes"]
    n_components = n_classes - 1

    print(f"\n{'═' * 80}")
    print(f"  EXTENDED BENCHMARK — {backbone.upper()} × {ds_cfg['label']}")
    print(f"  Features: {feature_dim}D → {n_components}D | Classifiers: {clf_names} | {len(seeds)} seeds")
    print(f"{'═' * 80}")

    # Load features
    X_train, y_train, X_test, y_test, _ = ds_cfg["load_fn"](backbone)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    print(f"  train={X_train_s.shape}, test={X_test_s.shape}")

    methods = get_method_configs(n_components, feature_dim)
    results = []

    for cfg in methods:
        name = cfg["name"]
        fn = cfg["fn"]
        category = cfg["category"]
        kwargs = cfg["kwargs"]

        # Reduce once (deterministic)
        print(f"\n  Reducing: {name}...", end="", flush=True)
        try:
            X_tr_red, X_te_red, dim, t_reduce = fn(
                X_train_s, y_train, X_test_s, n_components, **kwargs
            )
            print(f" → {dim}D in {t_reduce:.1f}s")
        except Exception as e:
            print(f" ❌ FAILED: {e}")
            for clf_name in clf_names:
                results.append({
                    "backbone": backbone, "dataset": dataset,
                    "method": name, "category": category,
                    "classifier": clf_name,
                    "accuracy_mean": 0.0, "accuracy_std": 0.0,
                    "dim": 0, "time_reduce": 0, "time_classify": 0,
                    "time_total": 0, "n_seeds": 0, "status": "FAILED",
                })
            continue

        # Run each classifier × seeds
        for clf_name in clf_names:
            clf_factory = CLASSIFIERS[clf_name]
            accs = []
            times = []
            for seed in seeds:
                clf = clf_factory(seed)
                acc, t_clf = evaluate(clf, X_tr_red, y_train, X_te_red, y_test)
                accs.append(acc)
                times.append(t_clf)

            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            mean_time = np.mean(times)

            results.append({
                "backbone": backbone,
                "dataset": dataset,
                "method": name,
                "category": category,
                "classifier": clf_name,
                "accuracy_mean": round(mean_acc, 2),
                "accuracy_std": round(std_acc, 2),
                "dim": dim,
                "feature_dim": feature_dim,
                "n_classes": n_classes,
                "n_components": n_components,
                "time_reduce": round(t_reduce, 2),
                "time_classify": round(mean_time, 2),
                "time_total": round(t_reduce + mean_time, 2),
                "n_seeds": len(seeds),
                "status": "OK",
            })

            print(f"    {clf_name:<8s}: {mean_acc:6.2f}±{std_acc:.2f}%  "
                  f"{dim:5d}D  {t_reduce + mean_time:7.1f}s")

    df = pd.DataFrame(results)

    # Summary
    print(f"\n  {'─' * 70}")
    print(f"  SUMMARY — {backbone.upper()} × {ds_cfg['label']}")
    for clf_name in clf_names:
        clf_df = df[(df["classifier"] == clf_name) & (df["status"] == "OK")]
        if len(clf_df) == 0:
            continue
        full_acc = clf_df[clf_df["method"] == "Full"]["accuracy_mean"].values
        lda_acc = clf_df[clf_df["method"] == "LDA"]["accuracy_mean"].values
        best = clf_df.loc[clf_df["accuracy_mean"].idxmax()]
        print(f"    {clf_name}: Full={full_acc[0]:.2f}%  LDA={lda_acc[0]:.2f}%  "
              f"Best={best['accuracy_mean']:.2f}% ({best['method']})")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Extended benchmark: 10 methods × 2 classifiers × 6 backbones × 3 datasets"
    )
    parser.add_argument("--backbone", type=str, nargs="+", default=None,
                        help="Backbone name(s) or 'all'. Default: resnet18 resnet50")
    parser.add_argument("--dataset", type=str, nargs="+", default=None,
                        help="Dataset(s) or 'all'. Default: cifar100")
    parser.add_argument("--classifier", type=str, nargs="+", default=None,
                        help="Classifier(s) or 'all'. Default: both LogReg and MLP")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of seeds (1-5). Default: 5")
    parser.add_argument("--output-dir", type=str, default="results/extended_benchmark",
                        help="Output directory")
    args = parser.parse_args()

    # Resolve backbones
    if args.backbone is None:
        backbones = ["resnet18", "resnet50"]
    elif "all" in args.backbone:
        backbones = list(BACKBONES.keys())
    else:
        backbones = args.backbone

    # Resolve datasets
    if args.dataset is None:
        datasets = ["cifar100"]
    elif "all" in args.dataset:
        datasets = list(DATASET_CONFIG.keys())
    else:
        datasets = args.dataset

    # Resolve classifiers
    if args.classifier is None:
        clf_names = list(CLASSIFIERS.keys())
    elif "all" in args.classifier:
        clf_names = list(CLASSIFIERS.keys())
    else:
        clf_names = args.classifier

    seeds = SEEDS[:args.seeds]

    print("═" * 80)
    print("  EXTENDED BENCHMARK CONFIGURATION")
    print(f"  Backbones: {backbones}")
    print(f"  Datasets:  {datasets}")
    print(f"  Classifiers: {clf_names}")
    print(f"  Seeds: {seeds}")
    print(f"  Total configs: {len(backbones)} × {len(datasets)} × {len(clf_names)} = "
          f"{len(backbones) * len(datasets) * len(clf_names)}")
    print("═" * 80)

    all_dfs = []
    os.makedirs(args.output_dir, exist_ok=True)

    for bb in backbones:
        for ds in datasets:
            df = run_config(bb, ds, clf_names, seeds)
            all_dfs.append(df)

            # Save incrementally (crash-safe)
            csv_path = os.path.join(args.output_dir, f"{bb}_{ds}.csv")
            df.to_csv(csv_path, index=False)
            print(f"  📄 Saved: {csv_path}")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined_path = os.path.join(args.output_dir, "all_results.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\n  📄 All results saved to: {combined_path}")

    # ── Grand summary ──
    print(f"\n{'═' * 80}")
    print("  GRAND SUMMARY: LDA gain over Full by backbone × dataset × classifier")
    print(f"{'═' * 80}")

    ok_df = combined[combined["status"] == "OK"]
    for clf_name in clf_names:
        print(f"\n  Classifier: {clf_name}")
        clf_df = ok_df[ok_df["classifier"] == clf_name]
        for _, grp in clf_df.groupby(["backbone", "dataset"]):
            bb = grp["backbone"].iloc[0]
            ds = grp["dataset"].iloc[0]
            full = grp[grp["method"] == "Full"]["accuracy_mean"].values
            lda = grp[grp["method"] == "LDA"]["accuracy_mean"].values
            best_row = grp.loc[grp["accuracy_mean"].idxmax()]
            if len(full) > 0 and len(lda) > 0:
                print(f"    {bb:>15s} × {ds:<15s}  Full={full[0]:6.2f}  "
                      f"LDA={lda[0]:6.2f} ({lda[0]-full[0]:+.2f})  "
                      f"Best={best_row['accuracy_mean']:6.2f} ({best_row['method']})")

    print(f"\n{'═' * 80}")
    print("  ✅ EXTENDED BENCHMARK COMPLETE")
    print(f"{'═' * 80}")


if __name__ == "__main__":
    main()
