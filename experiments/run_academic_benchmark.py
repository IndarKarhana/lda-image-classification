"""
Academic Benchmark — Our Novel Methods vs Established LDA Variants
===================================================================

Comprehensive comparison of our novel dimensionality reduction methods
against the most cited supervised DR methods from the literature.

METHODS COMPARED:
═══════════════════════════════════════════════════════════════════════
  Controls:
    - Full features (no reduction)
    - PCA (unsupervised baseline, Hotelling 1933)

  Classical Supervised DR:
    - Vanilla LDA (Fisher 1936 / Rao 1948)
    - PCA+LDA (Belhumeur et al. 1997 — "Fisherfaces")
    - R-LDA (Friedman 1989 — regularized discriminant analysis / Ledoit-Wolf)

  Modern Supervised DR:
    - LFDA (Sugiyama 2007 — Local Fisher Discriminant Analysis)
    - NCA (Goldberger et al. 2004 — Neighborhood Component Analysis)

  Our Novel Methods:
    - RDA (Residual Discriminant Augmentation)
    - DSB (Discriminant Subspace Boosting)
    - RDA+SMD (RDA + Spectral Margin Discriminants)

BACKBONES:
    resnet18 (512D), resnet50 (2048D), mobilenetv3 (576D), efficientnet (1280D)

DATASETS:
    cifar100 (100 classes, C-1=99), tiny_imagenet (200 classes, C-1=199)

Usage:
  python experiments/run_academic_benchmark.py                                # resnet18+50 × cifar100
  python experiments/run_academic_benchmark.py --backbone resnet18             # single backbone
  python experiments/run_academic_benchmark.py --backbone all                  # all 4 backbones
  python experiments/run_academic_benchmark.py --dataset tiny_imagenet         # Tiny ImageNet
  python experiments/run_academic_benchmark.py --backbone resnet50 --dataset cifar100

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
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import ledoit_wolf
from scipy.sparse import coo_matrix
from scipy.linalg import eigh as scipy_eigh

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Project imports ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.extract_features_multi import (
    get_or_extract_cifar100,
    get_or_extract_tiny_imagenet,
    BACKBONES,
)

SEED = 42


# ═══════════════════════════════════════════════════════════════════════
# Shared Utilities
# ═══════════════════════════════════════════════════════════════════════

def evaluate(X_train: np.ndarray, y_train: np.ndarray,
             X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
    """Standardize projected features, train LogisticRegression, return (accuracy%, classify_seconds).

    Post-projection standardization ensures fair comparison:
    - Concatenation methods (RDA) mix LDA + PCA features at different scales
    - Without this, lbfgs convergence varies wildly (31s vs 670s) and
      accuracy differences partly reflect conditioning, not method quality
    - Applied uniformly to ALL methods for fairness
    """
    t0 = time.perf_counter()
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    clf = LogisticRegression(solver="lbfgs", max_iter=5000, random_state=SEED)
    clf.fit(X_train_std, y_train)
    acc = clf.score(X_test_std, y_test) * 100
    return acc, time.perf_counter() - t0


def solve_whitened_eigen(Sb: np.ndarray, Sw_inv_sqrt: np.ndarray,
                         n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    """Solve Sb in whitened Sw space → (projection W, eigenvalues)."""
    Sb_white = Sw_inv_sqrt.T @ Sb @ Sw_inv_sqrt
    eigenvalues, eigvecs_white = np.linalg.eigh(Sb_white)
    eigenvectors = Sw_inv_sqrt @ eigvecs_white
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, idx[:n_components]], eigenvalues[idx[:n_components]]


def precompute_stats(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Precompute scatter matrices, Ledoit-Wolf shrinkage, whitening."""
    classes = np.unique(y)
    C = len(classes)
    N, D = X.shape
    global_mean = X.mean(axis=0)

    class_means = np.zeros((C, D), dtype=np.float64)
    class_counts = np.zeros(C)
    X_centered = np.zeros_like(X, dtype=np.float64)

    for i, c in enumerate(classes):
        mask = y == c
        class_means[i] = X[mask].mean(axis=0)
        class_counts[i] = mask.sum()
        X_centered[mask] = X[mask] - class_means[i]

    Sw = X_centered.T @ X_centered / N
    diffs = class_means - global_mean
    Sb = (diffs * class_counts[:, np.newaxis]).T @ diffs / N

    # Ledoit-Wolf shrinkage
    subsample_size = min(10000, N)
    _, alpha = ledoit_wolf(X_centered[:subsample_size])
    target = np.trace(Sw) / D * np.eye(D)
    Sw_reg = (1 - alpha) * Sw + alpha * target

    # Whitening: Sw_reg^{-1/2}
    eigvals, eigvecs = np.linalg.eigh(Sw_reg)
    eigvals = np.maximum(eigvals, 1e-10)
    inv_sqrt = 1.0 / np.sqrt(eigvals)
    Sw_inv_sqrt = eigvecs * inv_sqrt[np.newaxis, :]

    return {
        "Sw": Sw, "Sb": Sb, "Sw_reg": Sw_reg, "Sw_inv_sqrt": Sw_inv_sqrt,
        "class_means": class_means, "global_mean": global_mean,
        "class_counts": class_counts, "lw_alpha": alpha,
        "classes": classes, "C": C, "N": N, "D": D,
    }


# ═══════════════════════════════════════════════════════════════════════
#  CONTROL METHODS
# ═══════════════════════════════════════════════════════════════════════

def method_full(X_train, y_train, X_test, n_components, **kw):
    """Full features (no reduction) — upper/lower bound control."""
    return X_train.copy(), X_test.copy(), X_train.shape[1], 0.0


def method_pca(X_train, y_train, X_test, n_components, **kw):
    """PCA (Hotelling 1933) — unsupervised baseline."""
    t0 = time.perf_counter()
    pca = PCA(n_components=n_components, random_state=SEED)
    X_tr = pca.fit_transform(X_train)
    X_te = pca.transform(X_test)
    return X_tr, X_te, n_components, time.perf_counter() - t0


# ═══════════════════════════════════════════════════════════════════════
#  CLASSICAL SUPERVISED DR
# ═══════════════════════════════════════════════════════════════════════

def method_lda(X_train, y_train, X_test, n_components, **kw):
    """Vanilla LDA (Fisher 1936) — standard supervised baseline."""
    t0 = time.perf_counter()
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_tr = lda.fit_transform(X_train, y_train)
    X_te = lda.transform(X_test)
    return X_tr, X_te, n_components, time.perf_counter() - t0


def method_pca_lda(X_train, y_train, X_test, n_components, **kw):
    """
    PCA+LDA (Belhumeur et al. 1997 — "Fisherfaces").
    PCA to N-C dims (making Sw nonsingular), then LDA to C-1 dims.
    Classic two-stage approach from face recognition literature.
    """
    t0 = time.perf_counter()
    N = X_train.shape[0]
    C = len(np.unique(y_train))

    # PCA to min(N-C, 4*n_components) — enough to keep Sw full-rank
    pca_dim = min(N - C, max(4 * n_components, 500), X_train.shape[1])
    pca = PCA(n_components=pca_dim, random_state=SEED)
    X_tr_pca = pca.fit_transform(X_train)
    X_te_pca = pca.transform(X_test)

    # LDA on PCA-reduced features
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_tr = lda.fit_transform(X_tr_pca, y_train)
    X_te = lda.transform(X_te_pca)

    return X_tr, X_te, n_components, time.perf_counter() - t0


def method_rlda(X_train, y_train, X_test, n_components, **kw):
    """
    Regularized LDA (Friedman 1989 — Regularized Discriminant Analysis).
    Uses sklearn's Ledoit-Wolf automatic shrinkage estimator for Sw.
    Equivalent to R-LDA with optimal λ via Ledoit-Wolf.
    """
    t0 = time.perf_counter()
    lda = LinearDiscriminantAnalysis(
        n_components=n_components,
        solver="eigen",
        shrinkage="auto",  # Ledoit-Wolf optimal shrinkage
    )
    X_tr = lda.fit_transform(X_train, y_train)
    X_te = lda.transform(X_test)
    return X_tr, X_te, n_components, time.perf_counter() - t0


# ═══════════════════════════════════════════════════════════════════════
#  MODERN SUPERVISED DR — LFDA (Sugiyama 2007)
# ═══════════════════════════════════════════════════════════════════════

def method_lfda(X_train, y_train, X_test, n_components, **kw):
    """
    Local Fisher Discriminant Analysis (Sugiyama 2007).

    Unlike standard LDA which uses global class statistics, LFDA weights
    scatter matrices by local neighborhood structure. This preserves
    multimodal class distributions that LDA would average away.

    Algorithm:
      1. PCA pre-reduction (for speed on high-dim features)
      2. k-NN within each class → sparse affinity matrix
      3. Local within-class scatter via Laplacian: Sw_local = X^T L_w X
      4. Between-class scatter: Sb_local = St - Sw_local
      5. Generalized eigenvalue problem → projection

    Reference: Sugiyama, "Dimensionality Reduction of Multimodal Labeled
               Data by Local Fisher Discriminant Analysis", JMLR 2007.
    """
    t0 = time.perf_counter()
    k = kw.get("k", 7)
    pca_preprocess = kw.get("pca_preprocess", 300)

    N, D = X_train.shape
    classes = np.unique(y_train)
    C = len(classes)

    # ── PCA pre-reduction for high-dim features ──
    pca_pre = None
    if D > pca_preprocess:
        pca_pre = PCA(n_components=pca_preprocess, random_state=SEED)
        X_work = pca_pre.fit_transform(X_train)
        X_test_work = pca_pre.transform(X_test)
    else:
        X_work = X_train.copy()
        X_test_work = X_test.copy()

    D_work = X_work.shape[1]

    # ── Build sparse kNN affinity within each class ──
    all_rows, all_cols, all_vals = [], [], []

    for c in classes:
        mask = y_train == c
        idx = np.where(mask)[0]
        X_c = X_work[idx]
        n_c = len(idx)
        k_actual = min(k, n_c - 1)
        if k_actual < 1:
            continue

        # Use scipy.spatial.cKDTree to avoid sklearn threadpoolctl crash
        # on macOS/conda (AttributeError in threadpoolctl.get_version)
        from scipy.spatial import cKDTree
        tree = cKDTree(X_c)
        dists, nbrs = tree.query(X_c, k=k_actual + 1)

        # Local scaling (median of k-th neighbor distance)
        sigma = np.median(dists[:, -1]) + 1e-10

        # Heat kernel affinities, weighted by 1/n_c (within-class normalization)
        affinities = np.exp(-dists[:, 1:]**2 / (2 * sigma**2)) / n_c

        # Build coordinate lists for sparse matrix
        rows_local = np.repeat(np.arange(n_c), k_actual)
        cols_local = nbrs[:, 1:].flatten()
        rows_global = idx[rows_local]
        cols_global = idx[cols_local]
        vals_flat = affinities.flatten()

        # Symmetric: add both (i,j) and (j,i)
        all_rows.extend([rows_global, cols_global])
        all_cols.extend([cols_global, rows_global])
        all_vals.extend([vals_flat, vals_flat])

    all_rows = np.concatenate(all_rows)
    all_cols = np.concatenate(all_cols)
    all_vals = np.concatenate(all_vals)

    W_local = coo_matrix((all_vals, (all_rows, all_cols)), shape=(N, N)).tocsr()

    # ── Local within-class scatter via Laplacian ──
    # Sw_local = X^T (D_w - W_w) X = X^T D_w X - X^T W_w X
    D_diag = np.array(W_local.sum(axis=1)).flatten()

    # X^T D_w X  (diagonal weighting)
    XtDX = (X_work * D_diag[:, np.newaxis]).T @ X_work
    # X^T W_w X  (sparse matmul)
    WX = np.asarray(W_local.dot(X_work))
    XtWX = X_work.T @ WX

    Sw_local = (XtDX - XtWX) / N

    # ── Total scatter ──
    mean_work = X_work.mean(axis=0)
    X_c = X_work - mean_work
    St = X_c.T @ X_c / N

    # ── Between-class scatter = Total - Within (local) ──
    Sb_local = St - Sw_local

    # ── Regularize Sw_local ──
    reg = 1e-4 * np.trace(Sw_local) / D_work
    Sw_local_reg = Sw_local + reg * np.eye(D_work)

    # ── Solve generalized eigenvalue problem ──
    n_comp = min(n_components, C - 1, D_work - 1)
    eigenvalues, eigenvectors = scipy_eigh(
        Sb_local, Sw_local_reg,
        subset_by_index=[D_work - n_comp, D_work - 1]
    )
    # Sort descending
    idx_sort = np.argsort(eigenvalues)[::-1]
    W_proj = eigenvectors[:, idx_sort]

    # ── Project data ──
    X_tr_out = (X_work - mean_work) @ W_proj
    X_te_out = (X_test_work - mean_work) @ W_proj

    t_total = time.perf_counter() - t0
    return X_tr_out, X_te_out, n_comp, t_total


# ═══════════════════════════════════════════════════════════════════════
#  MODERN SUPERVISED DR — NCA (Goldberger et al. 2004)
# ═══════════════════════════════════════════════════════════════════════

def method_nca(X_train, y_train, X_test, n_components, **kw):
    """
    Neighborhood Component Analysis (Goldberger et al. 2004).

    Learns a linear transformation that maximizes leave-one-out kNN
    accuracy in the projected space. Uses gradient descent on a
    stochastic softmax criterion.

    For large datasets, we:
      1. PCA pre-reduce to ~200D (NCA is O(N²·d) per iteration)
      2. Subsample training data for fitting (NCA doesn't scale well)
      3. Apply learned transform to full dataset

    Reference: Goldberger et al., "Neighbourhood Components Analysis",
               NeurIPS 2004.
    """
    from sklearn.neighbors import NeighborhoodComponentsAnalysis

    t0 = time.perf_counter()
    pca_preprocess = kw.get("pca_preprocess", 200)
    max_fit_samples = kw.get("max_fit_samples", 10000)
    max_iter = kw.get("max_iter", 100)

    N, D = X_train.shape

    # ── PCA pre-reduction ──
    pca_pre = None
    if D > pca_preprocess:
        pca_pre = PCA(n_components=pca_preprocess, random_state=SEED)
        X_work = pca_pre.fit_transform(X_train)
        X_test_work = pca_pre.transform(X_test)
    else:
        X_work = X_train.copy()
        X_test_work = X_test.copy()

    # ── Stratified subsample for NCA fitting ──
    if N > max_fit_samples:
        rng = np.random.RandomState(SEED)
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

    # ── Fit NCA ──
    n_comp = min(n_components, X_work.shape[1] - 1)
    nca = NeighborhoodComponentsAnalysis(
        n_components=n_comp,
        max_iter=max_iter,
        random_state=SEED,
        verbose=0,
    )
    nca.fit(X_fit, y_fit)

    # ── Transform full data ──
    X_tr_out = nca.transform(X_work)
    X_te_out = nca.transform(X_test_work)

    t_total = time.perf_counter() - t0
    return X_tr_out, X_te_out, n_comp, t_total


# ═══════════════════════════════════════════════════════════════════════
#  OUR NOVEL METHOD 1: RDA (Residual Discriminant Augmentation)
# ═══════════════════════════════════════════════════════════════════════

def method_rda(X_train, y_train, X_test, n_components, **kw):
    """
    Residual Discriminant Augmentation (OURS).

    LDA captures ALL between-class variance in C-1 dims but discards
    within-class variance. RDA recovers the most informative within-class
    directions from LDA's ORTHOGONAL COMPLEMENT.

    Output: C-1 LDA dims + n_residual PCA-of-residual dims.
    """
    n_residual = kw.get("n_residual", 20)
    t0 = time.perf_counter()

    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)

    # Orthonormalize LDA directions
    W = lda.scalings_[:, :n_components]
    Q, _ = np.linalg.qr(W)

    # Project out LDA subspace → residual in orthogonal complement
    X_train_res = X_train - (X_train @ Q) @ Q.T
    X_test_res = X_test - (X_test @ Q) @ Q.T

    # PCA on residual
    n_res = min(n_residual, X_train.shape[1] - n_components - 1)
    if n_res < 1:
        n_res = 1
    pca_res = PCA(n_components=n_res, random_state=SEED)
    X_train_res_pca = pca_res.fit_transform(X_train_res)
    X_test_res_pca = pca_res.transform(X_test_res)

    X_train_out = np.hstack([X_train_lda, X_train_res_pca])
    X_test_out = np.hstack([X_test_lda, X_test_res_pca])

    t_reduce = time.perf_counter() - t0
    return X_train_out, X_test_out, X_train_out.shape[1], t_reduce


# ═══════════════════════════════════════════════════════════════════════
#  OUR NOVEL METHOD 2: DSB (Discriminant Subspace Boosting)
# ═══════════════════════════════════════════════════════════════════════

def method_dsb(X_train, y_train, X_test, n_components, **kw):
    """
    Discriminant Subspace Boosting (OURS).

    Applies boosting logic to the PROJECTION, not the classifier.
    Each round: LDA → classify → upweight misclassified samples in
    scatter computation → re-solve LDA. Concatenate multi-round features.
    """
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

        # Regularize
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

        # Update weights for next round
        if round_idx < n_rounds - 1:
            clf_boost = LogisticRegression(solver="lbfgs", max_iter=500, random_state=SEED)
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

    # Compress back to n_comp dims if boosting expanded
    if X_train_concat.shape[1] > n_comp:
        pca = PCA(n_components=n_comp, random_state=SEED)
        X_train_out = pca.fit_transform(X_train_concat)
        X_test_out = pca.transform(X_test_concat)
        out_dim = n_comp
    else:
        X_train_out = X_train_concat
        X_test_out = X_test_concat
        out_dim = X_train_concat.shape[1]

    t_reduce = time.perf_counter() - t0
    return X_train_out, X_test_out, out_dim, t_reduce


# ═══════════════════════════════════════════════════════════════════════
#  OUR NOVEL METHOD 3: RDA+SMD (RDA + Spectral Margin Discriminants)
# ═══════════════════════════════════════════════════════════════════════

def method_rda_smd(X_train, y_train, X_test, n_components, **kw):
    """
    RDA + Spectral Margin Discriminants (OURS).

    Two-phase approach:
      1. SMD: Self-refining LDA that focuses on hard class pairs
      2. RDA: Residual recovery from orthogonal complement

    SMD uses LDA's own output to identify poorly-separated class pairs,
    then re-weights the between-class scatter to focus on them.
    """
    n_residual = kw.get("n_residual", 20)
    margin_fraction = kw.get("margin_fraction", 0.3)
    t0 = time.perf_counter()

    classes = np.unique(y_train)
    C = len(classes)
    N, D = X_train.shape
    n_comp = min(n_components, C - 1)
    global_mean = X_train.mean(axis=0)

    # Compute class statistics
    class_means = np.zeros((C, D), dtype=np.float64)
    class_counts = np.zeros(C)
    X_centered = np.zeros_like(X_train, dtype=np.float64)
    for i, c in enumerate(classes):
        mask = y_train == c
        class_means[i] = X_train[mask].mean(axis=0)
        class_counts[i] = mask.sum()
        X_centered[mask] = X_train[mask] - class_means[i]

    # Sw + Ledoit-Wolf
    Sw = X_centered.T @ X_centered / N
    subsample_size = min(10000, N)
    _, alpha = ledoit_wolf(X_centered[:subsample_size])
    target_mat = np.trace(Sw) / D * np.eye(D)
    Sw_reg = (1 - alpha) * Sw + alpha * target_mat

    eigvals, eigvecs = np.linalg.eigh(Sw_reg)
    eigvals = np.maximum(eigvals, 1e-10)
    Sw_inv_sqrt = eigvecs * (1.0 / np.sqrt(eigvals))[np.newaxis, :]

    # Standard Sb
    diffs = class_means - global_mean
    Sb = (diffs * class_counts[:, np.newaxis]).T @ diffs / N

    # ── SMD Pass 1: Standard LDA projection ──
    W1, _ = solve_whitened_eigen(Sb, Sw_inv_sqrt, n_comp)

    # ── SMD Pass 2: Find hard class pairs ──
    class_means_proj = (class_means - global_mean) @ W1
    n_total_pairs = C * (C - 1) // 2
    n_hard = max(1, int(n_total_pairs * margin_fraction))

    pair_dists = []
    for i in range(C):
        for j in range(i + 1, C):
            dist = np.linalg.norm(class_means_proj[i] - class_means_proj[j])
            pair_dists.append((i, j, dist))
    pair_dists.sort(key=lambda x: x[2])

    # ── SMD Pass 3: Margin-weighted Sb ──
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

    # ── SMD Pass 4: Re-solve ──
    W_smd, _ = solve_whitened_eigen(Sb_margin, Sw_inv_sqrt, n_comp)
    X_train_smd = (X_train - global_mean) @ W_smd
    X_test_smd = (X_test - global_mean) @ W_smd

    # ── RDA: Residual from vanilla LDA ──
    lda = LinearDiscriminantAnalysis(n_components=n_comp)
    lda.fit(X_train, y_train)
    W_lda = lda.scalings_[:, :n_comp]
    Q, _ = np.linalg.qr(W_lda)

    X_train_res = X_train - (X_train @ Q) @ Q.T
    X_test_res = X_test - (X_test @ Q) @ Q.T

    n_res = min(n_residual, X_train.shape[1] - n_comp - 1)
    if n_res < 1:
        n_res = 1
    pca_res = PCA(n_components=n_res, random_state=SEED)
    X_train_res_pca = pca_res.fit_transform(X_train_res)
    X_test_res_pca = pca_res.transform(X_test_res)

    X_train_out = np.hstack([X_train_smd, X_train_res_pca])
    X_test_out = np.hstack([X_test_smd, X_test_res_pca])

    t_reduce = time.perf_counter() - t0
    return X_train_out, X_test_out, X_train_out.shape[1], t_reduce


# ═══════════════════════════════════════════════════════════════════════
#  METHOD REGISTRY
# ═══════════════════════════════════════════════════════════════════════

def get_method_configs(n_components: int, feature_dim: int) -> List[Dict]:
    """
    Build ordered list of methods to run.
    n_components = C-1 (99 for CIFAR-100, 199 for Tiny ImageNet).
    feature_dim = backbone feature dimensionality.
    """
    # Adaptive residual dim based on feature space size
    n_residual = 20 if feature_dim <= 576 else (25 if feature_dim <= 1280 else 30)

    methods = [
        # ── Controls ──
        {"name": "Full", "fn": method_full, "category": "control",
         "kwargs": {}},
        {"name": f"PCA ({n_components}D)", "fn": method_pca, "category": "control",
         "kwargs": {}},

        # ── Classical Supervised DR ──
        {"name": f"LDA ({n_components}D)", "fn": method_lda, "category": "classical",
         "kwargs": {}},
        {"name": f"PCA+LDA ({n_components}D)", "fn": method_pca_lda, "category": "classical",
         "kwargs": {}},
        {"name": f"R-LDA ({n_components}D)", "fn": method_rlda, "category": "classical",
         "kwargs": {}},

        # ── Modern Academic ──
        {"name": f"LFDA ({n_components}D)", "fn": method_lfda, "category": "academic",
         "kwargs": {"k": 7, "pca_preprocess": 300}},
        {"name": f"NCA ({n_components}D)", "fn": method_nca, "category": "academic",
         "kwargs": {"pca_preprocess": 200, "max_fit_samples": 10000, "max_iter": 100}},

        # ── Our Novel Methods ──
        {"name": f"RDA ({n_components}+{n_residual}D)", "fn": method_rda, "category": "ours",
         "kwargs": {"n_residual": n_residual}},
        {"name": f"DSB (2 rounds, {n_components}D)", "fn": method_dsb, "category": "ours",
         "kwargs": {"n_rounds": 2}},
        {"name": f"RDA+SMD ({n_components}+{n_residual}D)", "fn": method_rda_smd,
         "category": "ours",
         "kwargs": {"n_residual": n_residual, "margin_fraction": 0.3}},
    ]
    return methods


# ═══════════════════════════════════════════════════════════════════════
#  MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════

def run_benchmark(backbone: str, dataset: str) -> pd.DataFrame:
    """Run full benchmark for one backbone × one dataset."""
    feature_dim = BACKBONES[backbone]["feature_dim"]

    # Dataset config
    if dataset == "cifar100":
        n_classes = 100
        load_fn = lambda: get_or_extract_cifar100(backbone)
        dataset_label = "CIFAR-100"
    elif dataset == "tiny_imagenet":
        n_classes = 200
        load_fn = lambda: get_or_extract_tiny_imagenet(backbone)
        dataset_label = "Tiny ImageNet"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    n_components = n_classes - 1  # LDA max = C-1

    print("\n" + "═" * 78)
    print(f"  ACADEMIC BENCHMARK — {backbone.upper()} × {dataset_label}")
    print(f"  Features: {feature_dim}D → {n_components}D (LDA max)")
    print(f"  Classes: {n_classes}")
    print("═" * 78)

    # ── Load & standardize features ──
    print("\n  Loading features...")
    X_train, y_train, X_test, y_test, fdim = load_fn()
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    print(f"  train={X_train_s.shape}, test={X_test_s.shape}")

    # ── Get method list ──
    methods = get_method_configs(n_components, feature_dim)

    results = []
    print(f"\n  {'Method':<30s} {'Acc':>7s}  {'Dim':>5s}  {'t_red':>6s}  {'t_clf':>6s}  {'Total':>7s}  Category")
    print("  " + "─" * 76)

    for cfg in methods:
        name = cfg["name"]
        fn = cfg["fn"]
        category = cfg["category"]
        kwargs = cfg["kwargs"]

        print(f"  {name:<30s}", end="", flush=True)
        try:
            X_tr, X_te, dim, t_reduce = fn(
                X_train_s, y_train, X_test_s, n_components, **kwargs
            )
            acc, t_classify = evaluate(X_tr, y_train, X_te, y_test)
            t_total = t_reduce + t_classify

            results.append({
                "backbone": backbone,
                "dataset": dataset,
                "method": name,
                "category": category,
                "accuracy": round(acc, 2),
                "dim": dim,
                "time_reduce": round(t_reduce, 2),
                "time_classify": round(t_classify, 2),
                "time_total": round(t_total, 2),
                "n_classes": n_classes,
                "feature_dim": feature_dim,
            })

            print(f" {acc:6.2f}%  {dim:5d}  {t_reduce:6.1f}s  {t_classify:6.1f}s  {t_total:7.1f}s  {category}")

        except Exception as e:
            print(f" ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "backbone": backbone, "dataset": dataset,
                "method": f"{name} (FAILED)", "category": category,
                "accuracy": 0.0, "dim": 0,
                "time_reduce": 0, "time_classify": 0, "time_total": 0,
                "n_classes": n_classes, "feature_dim": feature_dim,
            })

    df = pd.DataFrame(results)

    # ── Summary ──
    print("\n" + "─" * 78)
    print(f"  RANKING — {backbone.upper()} × {dataset_label} (by accuracy)")
    print("─" * 78)

    df_sorted = df.sort_values("accuracy", ascending=False).reset_index(drop=True)

    # Reference points
    lda_rows = df_sorted[df_sorted["method"].str.startswith("LDA")]
    lda_acc = lda_rows["accuracy"].values[0] if len(lda_rows) > 0 else 0
    full_rows = df_sorted[df_sorted["method"].str.startswith("Full")]
    full_acc = full_rows["accuracy"].values[0] if len(full_rows) > 0 else 0

    for rank, (_, row) in enumerate(df_sorted.iterrows(), 1):
        acc = row["accuracy"]
        if acc >= full_acc and row["method"] != "Full":
            icon = "🟢"
        elif acc >= lda_acc:
            icon = "🟡"
        else:
            icon = "🔴"

        delta_lda = acc - lda_acc
        delta_full = acc - full_acc
        cat_tag = f"[{row['category']}]".ljust(12)
        print(f"  {rank:2d}. {icon} {acc:6.2f}%  (Δlda={delta_lda:+.2f}, Δfull={delta_full:+.2f})  "
              f"|  {row['dim']:4d}D  {row['time_total']:6.1f}s  {cat_tag} {row['method']}")

    # Highlight best from each category
    print()
    for cat in ["classical", "academic", "ours"]:
        cat_df = df_sorted[df_sorted["category"] == cat]
        if len(cat_df) > 0:
            best = cat_df.iloc[0]
            print(f"  🏆 Best {cat}: {best['method']} = {best['accuracy']:.2f}% "
                  f"({best['accuracy'] - lda_acc:+.2f}% vs LDA)")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Academic benchmark: our methods vs established LDA variants"
    )
    parser.add_argument("--backbone", type=str, default=None,
                        help="Backbone name or 'all'. Default: resnet18+resnet50")
    parser.add_argument("--dataset", type=str, default="cifar100",
                        choices=["cifar100", "tiny_imagenet"],
                        help="Dataset to benchmark on")
    args = parser.parse_args()

    # Determine backbones
    if args.backbone is None:
        backbones = ["resnet18", "resnet50"]
    elif args.backbone == "all":
        backbones = list(BACKBONES.keys())
    elif args.backbone in BACKBONES:
        backbones = [args.backbone]
    else:
        print(f"Unknown backbone: {args.backbone}")
        print(f"Available: {list(BACKBONES.keys())} or 'all'")
        sys.exit(1)

    all_dfs = []
    for bb in backbones:
        df = run_benchmark(bb, args.dataset)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    # ── Cross-backbone summary ──
    if len(all_dfs) > 1:
        print("\n" + "═" * 78)
        print("  CROSS-BACKBONE SUMMARY")
        print("═" * 78)

        for bb in backbones:
            bb_df = combined[combined["backbone"] == bb]
            fdim = BACKBONES[bb]["feature_dim"]
            lda_row = bb_df[bb_df["method"].str.startswith("LDA")]
            lda_acc = lda_row["accuracy"].values[0] if len(lda_row) > 0 else 0
            full_row = bb_df[bb_df["method"].str.startswith("Full")]
            full_acc = full_row["accuracy"].values[0] if len(full_row) > 0 else 0

            best_ours = bb_df[bb_df["category"] == "ours"].sort_values("accuracy", ascending=False)
            best_acad = bb_df[bb_df["category"] == "academic"].sort_values("accuracy", ascending=False)

            print(f"\n  {bb.upper()} ({fdim}D):")
            print(f"    Full features:  {full_acc:.2f}%")
            print(f"    Vanilla LDA:    {lda_acc:.2f}%")
            if len(best_acad) > 0:
                ba = best_acad.iloc[0]
                print(f"    Best academic:  {ba['accuracy']:.2f}%  ({ba['method']})  "
                      f"Δlda={ba['accuracy']-lda_acc:+.2f}%")
            if len(best_ours) > 0:
                bo = best_ours.iloc[0]
                print(f"    Best ours:      {bo['accuracy']:.2f}%  ({bo['method']})  "
                      f"Δlda={bo['accuracy']-lda_acc:+.2f}%")
                if len(best_acad) > 0:
                    gap = bo["accuracy"] - ba["accuracy"]
                    print(f"    Ours vs academic best: {gap:+.2f}%")

    # ── Save results ──
    os.makedirs("results/academic_benchmark", exist_ok=True)
    csv_path = f"results/academic_benchmark/{args.dataset}_benchmark.csv"
    combined.to_csv(csv_path, index=False)
    print(f"\n  📄 Results saved to: {csv_path}")

    print("\n" + "═" * 78)
    print("  ✅ ACADEMIC BENCHMARK COMPLETE")
    print("═" * 78)


if __name__ == "__main__":
    main()
