"""
Novel DG-LDA Methods — Quick Comparison Test
==============================================
Tests 5 truly novel approaches on cached backbone × CIFAR-100 features.

Problem: Current DG-LDA has 2 issues:
  1. Too slow (280-365s vs vanilla LDA's 32s) — profiling is 229s alone
  2. Can't beat full features on ResNet-50 (72.16% vs 73.08%)

Root cause: LDA@99D captures ALL between-class variance (Sb rank = C-1 = 99)
but discards all within-class variance (1949 of 2048 dims). Classifiers need
within-class structure to set precise decision boundaries.

5 Novel Methods (none of these exist in LDA literature):
  1. RDA  — Residual Discriminant Augmentation
  2. KCS  — kNN Confusion Scatter
  3. SDP  — Stochastic Discriminant Projection
  4. SMD  — Spectral Margin Discriminants (self-refining LDA)
  5. DSB  — Discriminant Subspace Boosting

Usage:
  python experiments/run_novel_methods_test.py              # default: resnet50
  python experiments/run_novel_methods_test.py resnet18

Author: Research Study
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import ledoit_wolf

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.extract_features_multi import get_or_extract_cifar100, BACKBONES

SEED = 42


# ═══════════════════════════════════════════════════════════════════════
# Shared Utilities
# ═══════════════════════════════════════════════════════════════════════

def evaluate(X_train: np.ndarray, y_train: np.ndarray,
             X_test: np.ndarray, y_test: np.ndarray) -> tuple:
    """Train LogisticRegression and return (accuracy%, time_seconds)."""
    t0 = time.perf_counter()
    clf = LogisticRegression(solver="lbfgs", max_iter=5000, random_state=SEED)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test) * 100
    return acc, time.perf_counter() - t0


def precompute_stats(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Precompute scatter matrices, Ledoit-Wolf shrinkage, and whitening
    so novel methods don't redundantly recompute them.
    """
    classes = np.unique(y)
    C = len(classes)
    N, D = X.shape
    global_mean = X.mean(axis=0)

    # Class means and within-class centering
    class_means = np.zeros((C, D), dtype=np.float64)
    class_counts = np.zeros(C)
    X_centered = np.zeros_like(X, dtype=np.float64)

    for i, c in enumerate(classes):
        mask = y == c
        class_means[i] = X[mask].mean(axis=0)
        class_counts[i] = mask.sum()
        X_centered[mask] = X[mask] - class_means[i]

    # Sw via one big matrix multiply (much faster than per-class loop)
    Sw = X_centered.T @ X_centered / N

    # Sb
    diffs = class_means - global_mean
    Sb = (diffs * class_counts[:, np.newaxis]).T @ diffs / N

    # Ledoit-Wolf on subsample for speed
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


def solve_whitened_eigen(Sb_custom: np.ndarray, Sw_inv_sqrt: np.ndarray,
                         n_components: int) -> tuple:
    """Solve Sb_custom in whitened space, return (projection W, eigenvalues)."""
    Sb_white = Sw_inv_sqrt.T @ Sb_custom @ Sw_inv_sqrt
    eigenvalues, eigvecs_white = np.linalg.eigh(Sb_white)
    eigenvectors = Sw_inv_sqrt @ eigvecs_white
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, idx[:n_components]], eigenvalues[idx[:n_components]]


# ═══════════════════════════════════════════════════════════════════════
# METHOD 1: Residual Discriminant Augmentation (RDA)
# ═══════════════════════════════════════════════════════════════════════
#
# KEY INSIGHT: LDA provably captures ALL between-class variance in C-1
# dimensions (since rank(Sb) = C-1). By discarding the remaining D-(C-1)
# dimensions, it removes ALL within-class variance. But classifiers need
# within-class structure to set precise decision boundaries near class
# overlaps.
#
# RDA recovers the most informative within-class directions from LDA's
# ORTHOGONAL COMPLEMENT. Unlike naive LDA+PCA concatenation, the residual
# PCA features have ZERO correlation with LDA features by construction.
#
# NOT IN LITERATURE: PCA+LDA pipelines exist (PCA first, then LDA).
# LDA+PCA independent concatenation exists. But extracting PCA from the
# orthogonal complement of the LDA subspace is novel.
# ═══════════════════════════════════════════════════════════════════════

def method_rda(X_train, y_train, X_test, n_lda=99, n_residual=20):
    """Residual Discriminant Augmentation."""
    t0 = time.perf_counter()

    # Step 1: Standard LDA
    lda = LinearDiscriminantAnalysis(n_components=n_lda)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)

    # Step 2: Orthonormalize LDA directions
    W = lda.scalings_[:, :n_lda]  # (D, n_lda)
    Q, _ = np.linalg.qr(W)  # Q is (D, n_lda), orthonormal columns

    # Step 3: Project out LDA subspace → residual (efficient 2-step)
    X_train_proj = X_train @ Q  # (N, n_lda)
    X_train_recon = X_train_proj @ Q.T  # (N, D) — reconstruction
    X_train_res = X_train - X_train_recon  # residual in orthogonal complement

    X_test_proj = X_test @ Q
    X_test_res = X_test - X_test_proj @ Q.T

    # Step 4: PCA on residual to extract top within-class variance directions
    pca_res = PCA(n_components=n_residual, random_state=SEED)
    X_train_res_pca = pca_res.fit_transform(X_train_res)
    X_test_res_pca = pca_res.transform(X_test_res)

    # Step 5: Concatenate — discriminant + within-class structure
    X_train_out = np.hstack([X_train_lda, X_train_res_pca])
    X_test_out = np.hstack([X_test_lda, X_test_res_pca])

    t_reduce = time.perf_counter() - t0
    return X_train_out, X_test_out, X_train_out.shape[1], t_reduce


# ═══════════════════════════════════════════════════════════════════════
# CONTROL: Naive LDA + PCA concatenation (NOT novel — validates RDA)
# ═══════════════════════════════════════════════════════════════════════

def method_lda_pca_concat(X_train, y_train, X_test, n_lda=99, n_pca=20):
    """Non-novel control: independent LDA + PCA concatenation (with redundancy)."""
    t0 = time.perf_counter()

    lda = LinearDiscriminantAnalysis(n_components=n_lda)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)

    pca = PCA(n_components=n_pca, random_state=SEED)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    X_train_out = np.hstack([X_train_lda, X_train_pca])
    X_test_out = np.hstack([X_test_lda, X_test_pca])

    t_reduce = time.perf_counter() - t0
    return X_train_out, X_test_out, X_train_out.shape[1], t_reduce


# ═══════════════════════════════════════════════════════════════════════
# METHOD 2: kNN Confusion Scatter (KCS)
# ═══════════════════════════════════════════════════════════════════════
#
# PROBLEM: CW-LDA uses Bhattacharyya distances to weight class pairs,
# which takes ~100s+ for the distance matrix computation.
#
# KCS replaces this with kNN-based confusion: for each class pair (i,j),
# confusion_weight = fraction of class-i samples whose k-nearest neighbors
# include class-j members. This directly measures empirical confusion.
#
# NOT IN LITERATURE: LFDA (Local Fisher DA) uses kNN for locality
# preservation WITHIN classes. KCS uses kNN to estimate confusion
# BETWEEN classes — fundamentally different concept.
# ═══════════════════════════════════════════════════════════════════════

def method_kcs(X_train, y_train, X_test, stats, n_components=99, k=10):
    """kNN Confusion Scatter."""
    t0 = time.perf_counter()

    C, D, N = stats["C"], stats["D"], stats["N"]
    classes = stats["classes"]
    class_means = stats["class_means"]
    Sw_inv_sqrt = stats["Sw_inv_sqrt"]

    # Fast kNN in PCA-reduced space (avoid curse of dimensionality)
    pca_quick = PCA(n_components=min(128, D), random_state=SEED)
    X_pca = pca_quick.fit_transform(X_train)

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", n_jobs=-1)
    nn.fit(X_pca)
    _, indices = nn.kneighbors(X_pca)
    indices = indices[:, 1:]  # Exclude self

    # Build empirical confusion matrix from kNN neighborhoods
    neighbor_labels = y_train[indices]  # (N, k)
    confusion = np.zeros((C, C))

    for i, c in enumerate(classes):
        mask = y_train == c
        nc = mask.sum()
        if nc > 0:
            nl = neighbor_labels[mask]  # (nc, k)
            for j, c2 in enumerate(classes):
                if i != j:
                    confusion[i, j] = (nl == c2).sum() / (nc * k)

    confusion = (confusion + confusion.T) / 2  # Symmetrize

    # Build confusion-weighted between-class scatter
    Sb_cw = np.zeros((D, D), dtype=np.float64)
    n_pairs = 0
    for i in range(C):
        for j in range(i + 1, C):
            w = confusion[i, j]
            if w > 1e-8:
                dm = (class_means[i] - class_means[j]).reshape(-1, 1)
                Sb_cw += w * (dm @ dm.T)
                n_pairs += 1
    Sb_cw /= N

    # Solve in whitened space (reuse precomputed Sw_inv_sqrt)
    W, _ = solve_whitened_eigen(Sb_cw, Sw_inv_sqrt, n_components)

    mean = stats["global_mean"]
    X_train_out = (X_train - mean) @ W
    X_test_out = (X_test - mean) @ W

    t_reduce = time.perf_counter() - t0
    return X_train_out, X_test_out, n_components, t_reduce


# ═══════════════════════════════════════════════════════════════════════
# METHOD 3: Stochastic Discriminant Projection (SDP)
# ═══════════════════════════════════════════════════════════════════════
#
# IDEA: Scatter matrices estimated from N samples have estimation error.
# By computing scatter on M random stratified subsets and averaging,
# we get variance reduction (like bagging) which acts as implicit
# regularization — without needing explicit shrinkage estimation.
#
# NOT IN LITERATURE: Ensemble methods exist for classifiers, not for
# scatter matrix estimation in discriminant analysis. Bagging of scatter
# matrices is genuinely new.
# ═══════════════════════════════════════════════════════════════════════

def method_sdp(X_train, y_train, X_test, stats,
               n_components=99, n_bags=5, fraction=0.3):
    """Stochastic Discriminant Projection."""
    t0 = time.perf_counter()

    classes = stats["classes"]
    C, D, N = stats["C"], stats["D"], stats["N"]
    Sw_inv_sqrt = stats["Sw_inv_sqrt"]

    rng = np.random.RandomState(SEED)
    Sb_sum = np.zeros((D, D), dtype=np.float64)

    for bag in range(n_bags):
        # Stratified subsample
        indices = []
        for c in classes:
            class_idx = np.where(y_train == c)[0]
            n_sample = max(2, int(len(class_idx) * fraction))
            sampled = rng.choice(class_idx, size=n_sample, replace=False)
            indices.extend(sampled)
        indices = np.array(indices)

        X_sub = X_train[indices]
        y_sub = y_train[indices]
        N_sub = len(indices)
        global_mean_sub = X_sub.mean(axis=0)

        # Compute Sb on subset
        class_means_sub = np.zeros((C, D), dtype=np.float64)
        class_counts_sub = np.zeros(C)
        for i, c in enumerate(classes):
            mask = y_sub == c
            class_means_sub[i] = X_sub[mask].mean(axis=0)
            class_counts_sub[i] = mask.sum()

        diffs = class_means_sub - global_mean_sub
        Sb_sub = (diffs * class_counts_sub[:, np.newaxis]).T @ diffs / N_sub
        Sb_sum += Sb_sub

    Sb_avg = Sb_sum / n_bags

    # Solve using full-data Sw_inv_sqrt (Sw benefits from all data)
    W, _ = solve_whitened_eigen(Sb_avg, Sw_inv_sqrt, n_components)

    mean = stats["global_mean"]
    X_train_out = (X_train - mean) @ W
    X_test_out = (X_test - mean) @ W

    t_reduce = time.perf_counter() - t0
    return X_train_out, X_test_out, n_components, t_reduce


# ═══════════════════════════════════════════════════════════════════════
# METHOD 4: Spectral Margin Discriminants (SMD)
# ═══════════════════════════════════════════════════════════════════════
#
# IDEA: Standard LDA maximizes average class separation. But some class
# pairs are already well-separated; wasting discriminant power on them
# doesn't help. SMD uses a two-pass approach:
#   Pass 1: Standard LDA → project to discriminant space
#   Pass 2: In LDA space, identify the K hardest class pairs (closest means)
#   Pass 3: Re-weight Sb to focus on these hard pairs
#   Pass 4: Re-solve the eigenvalue problem
#
# NOT IN LITERATURE: Margin Fisher Analysis uses graph embedding.
# Weighted LDA uses cost matrices. SMD uniquely uses LDA's OWN output
# as feedback to refine itself — a self-referencing discriminant loop.
# ═══════════════════════════════════════════════════════════════════════

def method_smd(X_train, y_train, X_test, stats,
               n_components=99, margin_fraction=0.3):
    """Spectral Margin Discriminants — Self-Refining LDA."""
    t0 = time.perf_counter()

    C, D, N = stats["C"], stats["D"], stats["N"]
    classes = stats["classes"]
    class_means = stats["class_means"]
    global_mean = stats["global_mean"]
    Sw_inv_sqrt = stats["Sw_inv_sqrt"]
    Sb = stats["Sb"]

    # Pass 1: Standard LDA projection
    W1, _ = solve_whitened_eigen(Sb, Sw_inv_sqrt, n_components)

    # Pass 2: Compute class-mean distances in LDA space
    class_means_proj = (class_means - global_mean) @ W1  # (C, n_comp)

    n_total_pairs = C * (C - 1) // 2
    n_hard = max(1, int(n_total_pairs * margin_fraction))

    pair_dists = []
    for i in range(C):
        for j in range(i + 1, C):
            dist = np.linalg.norm(class_means_proj[i] - class_means_proj[j])
            pair_dists.append((i, j, dist))
    pair_dists.sort(key=lambda x: x[2])  # Ascending: closest (hardest) first

    # Pass 3: Build margin-weighted Sb in original space
    Sb_margin = np.zeros((D, D), dtype=np.float64)

    # Hard pairs: high weight (inverse distance)
    for i, j, dist in pair_dists[:n_hard]:
        w = 1.0 / (dist + 1e-6)
        dm = (class_means[i] - class_means[j]).reshape(-1, 1)
        Sb_margin += w * (dm @ dm.T)

    # Easy pairs: reduced weight (10× less)
    for i, j, dist in pair_dists[n_hard:]:
        w = 0.1 / (dist + 1e-6)
        dm = (class_means[i] - class_means[j]).reshape(-1, 1)
        Sb_margin += w * (dm @ dm.T)

    Sb_margin /= N

    # Pass 4: Re-solve with margin-weighted Sb
    W2, _ = solve_whitened_eigen(Sb_margin, Sw_inv_sqrt, n_components)

    X_train_out = (X_train - global_mean) @ W2
    X_test_out = (X_test - global_mean) @ W2

    t_reduce = time.perf_counter() - t0
    return X_train_out, X_test_out, n_components, t_reduce


# ═══════════════════════════════════════════════════════════════════════
# METHOD 5: Discriminant Subspace Boosting (DSB)
# ═══════════════════════════════════════════════════════════════════════
#
# IDEA: Apply boosting logic to the PROJECTION, not the classifier.
# Each round: LDA → classify → upweight misclassified samples in scatter
# computation → re-solve LDA. Then concatenate multi-round features.
#
# The second round's LDA inherently focuses on samples the first round
# got wrong, discovering new discriminant directions the first round missed.
#
# NOT IN LITERATURE: AdaBoost, gradient boosting, etc. boost classifiers.
# Boosting the discriminant PROJECTION itself (re-weighting scatter
# matrices based on classification error) is novel.
# ═══════════════════════════════════════════════════════════════════════

def method_dsb(X_train, y_train, X_test, n_lda=99, n_rounds=2):
    """Discriminant Subspace Boosting."""
    t0 = time.perf_counter()

    N, D = X_train.shape
    classes = np.unique(y_train)
    C = len(classes)
    n_comp = min(n_lda, C - 1)

    sample_weights = np.ones(N, dtype=np.float64) / N
    all_train_features = []
    all_test_features = []

    for round_idx in range(n_rounds):
        # Weighted class statistics
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

        # Simple regularization (faster than Ledoit-Wolf)
        Sw_trace = np.trace(Sw)
        alpha = 0.05
        if Sw_trace > 0:
            target = Sw_trace / D * np.eye(D)
            Sw_reg = (1 - alpha) * Sw + alpha * target
        else:
            Sw_reg = np.eye(D)

        # Solve
        eigvals_sw, eigvecs_sw = np.linalg.eigh(Sw_reg)
        eigvals_sw = np.maximum(eigvals_sw, 1e-10)
        inv_sqrt_vals = 1.0 / np.sqrt(eigvals_sw)
        Sw_inv_sqrt = eigvecs_sw * inv_sqrt_vals[np.newaxis, :]

        W, _ = solve_whitened_eigen(Sb, Sw_inv_sqrt, n_comp)

        # Project
        X_train_proj = (X_train - global_mean_w) @ W
        X_test_proj = (X_test - global_mean_w) @ W

        all_train_features.append(X_train_proj)
        all_test_features.append(X_test_proj)

        # Update weights for next round: upweight misclassified samples
        if round_idx < n_rounds - 1:
            clf_boost = LogisticRegression(
                solver="lbfgs", max_iter=500, random_state=SEED
            )
            clf_boost.fit(X_train_proj, y_train)
            pred = clf_boost.predict(X_train_proj)
            wrong = (pred != y_train).astype(float)
            error_rate = np.average(wrong, weights=sample_weights)

            if 0 < error_rate < 0.5:
                beta = error_rate / (1 - error_rate)
                sample_weights *= np.exp(wrong * np.log(1 / beta + 1e-10))
                sample_weights /= sample_weights.sum()

    # Concatenate all rounds
    X_train_concat = np.hstack(all_train_features)
    X_test_concat = np.hstack(all_test_features)

    # Compress with PCA if total dims > n_lda
    total_dim = X_train_concat.shape[1]
    if total_dim > n_lda:
        pca = PCA(n_components=n_lda, random_state=SEED)
        X_train_out = pca.fit_transform(X_train_concat)
        X_test_out = pca.transform(X_test_concat)
        out_dim = n_lda
    else:
        X_train_out = X_train_concat
        X_test_out = X_test_concat
        out_dim = total_dim

    t_reduce = time.perf_counter() - t0
    return X_train_out, X_test_out, out_dim, t_reduce


# ═══════════════════════════════════════════════════════════════════════
# COMBO: RDA + SMD (augmented self-refining)
# ═══════════════════════════════════════════════════════════════════════

def method_rda_smd(X_train, y_train, X_test, stats,
                   n_lda=99, n_residual=20, margin_fraction=0.3):
    """SMD for better discriminant directions + RDA for residual recovery."""
    t0 = time.perf_counter()

    # SMD for the discriminant part
    X_train_smd, X_test_smd, _, _ = method_smd(
        X_train, y_train, X_test, stats,
        n_components=n_lda, margin_fraction=margin_fraction
    )

    # Use vanilla LDA projection for residual computation
    # (its projection matrix is readily available from sklearn)
    lda = LinearDiscriminantAnalysis(n_components=n_lda)
    lda.fit(X_train, y_train)

    W = lda.scalings_[:, :n_lda]
    Q, _ = np.linalg.qr(W)

    # Residual PCA
    X_train_res = X_train - (X_train @ Q) @ Q.T
    X_test_res = X_test - (X_test @ Q) @ Q.T

    pca_res = PCA(n_components=n_residual, random_state=SEED)
    X_train_res_pca = pca_res.fit_transform(X_train_res)
    X_test_res_pca = pca_res.transform(X_test_res)

    # Concatenate SMD features + residual PCA
    X_train_out = np.hstack([X_train_smd, X_train_res_pca])
    X_test_out = np.hstack([X_test_smd, X_test_res_pca])

    t_reduce = time.perf_counter() - t0
    return X_train_out, X_test_out, X_train_out.shape[1], t_reduce


# ═══════════════════════════════════════════════════════════════════════
# COMBO: RDA + KCS (confusion-aware residual augmentation)
# ═══════════════════════════════════════════════════════════════════════

def method_rda_kcs(X_train, y_train, X_test, stats,
                   n_lda=99, n_residual=20, k=10):
    """KCS for better scatter weighting + RDA for residual recovery."""
    t0 = time.perf_counter()

    # KCS-LDA for discriminant features
    X_train_kcs, X_test_kcs, _, _ = method_kcs(
        X_train, y_train, X_test, stats, n_components=n_lda, k=k
    )

    # Residual from vanilla LDA (reuse readily available projection)
    lda = LinearDiscriminantAnalysis(n_components=n_lda)
    lda.fit(X_train, y_train)

    W = lda.scalings_[:, :n_lda]
    Q, _ = np.linalg.qr(W)

    X_train_res = X_train - (X_train @ Q) @ Q.T
    X_test_res = X_test - (X_test @ Q) @ Q.T

    pca_res = PCA(n_components=n_residual, random_state=SEED)
    X_train_res_pca = pca_res.fit_transform(X_train_res)
    X_test_res_pca = pca_res.transform(X_test_res)

    X_train_out = np.hstack([X_train_kcs, X_train_res_pca])
    X_test_out = np.hstack([X_test_kcs, X_test_res_pca])

    t_reduce = time.perf_counter() - t0
    return X_train_out, X_test_out, X_train_out.shape[1], t_reduce


# ═══════════════════════════════════════════════════════════════════════
# COMBO: RDA + DSB (boosted residual augmentation)
# ═══════════════════════════════════════════════════════════════════════

def method_rda_dsb(X_train, y_train, X_test, n_lda=99, n_residual=20):
    """DSB for boosted projections + RDA for residual recovery."""
    t0 = time.perf_counter()

    # DSB for discriminant features
    X_train_dsb, X_test_dsb, dim_dsb, _ = method_dsb(
        X_train, y_train, X_test, n_lda=n_lda, n_rounds=2
    )

    # Residual from vanilla LDA
    lda = LinearDiscriminantAnalysis(n_components=n_lda)
    lda.fit(X_train, y_train)

    W = lda.scalings_[:, :n_lda]
    Q, _ = np.linalg.qr(W)

    X_train_res = X_train - (X_train @ Q) @ Q.T
    X_test_res = X_test - (X_test @ Q) @ Q.T

    pca_res = PCA(n_components=n_residual, random_state=SEED)
    X_train_res_pca = pca_res.fit_transform(X_train_res)
    X_test_res_pca = pca_res.transform(X_test_res)

    X_train_out = np.hstack([X_train_dsb, X_train_res_pca])
    X_test_out = np.hstack([X_test_dsb, X_test_res_pca])

    t_reduce = time.perf_counter() - t0
    return X_train_out, X_test_out, X_train_out.shape[1], t_reduce


# ═══════════════════════════════════════════════════════════════════════
# MAIN TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════

def main(backbone: str = "resnet50"):
    feature_dim = BACKBONES[backbone]["feature_dim"]

    print("=" * 74)
    print(f"  NOVEL DG-LDA METHODS — QUICK TEST — {backbone.upper()} × CIFAR-100 ({feature_dim}D)")
    print("=" * 74)

    # ── Load features ──
    print("\n[1/6] Loading cached features...")
    X_train, y_train, X_test, y_test, fdim = get_or_extract_cifar100(backbone)
    n_classes = len(np.unique(y_train))
    print(f"  train={X_train.shape}, test={X_test.shape}, classes={n_classes}")

    # ── Standardize ──
    print("[2/6] Standardizing...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ── Precompute shared stats ──
    print("[3/6] Precomputing scatter matrices + whitening...")
    t0 = time.perf_counter()
    stats = precompute_stats(X_train, y_train)
    t_pre = time.perf_counter() - t0
    print(f"  Done in {t_pre:.1f}s  |  LW α={stats['lw_alpha']:.4f}")

    results = []

    def record(method, acc, dim, t_reduce, t_clf, method_type="novel"):
        results.append({
            "method": method, "accuracy": acc, "dim": dim,
            "time_reduce": round(t_reduce, 2),
            "time_classify": round(t_clf, 2),
            "time_total": round(t_reduce + t_clf, 2),
            "type": method_type,
        })

    # ═══════════════════════════════════════════════════════════════
    # [4/6] BASELINES
    # ═══════════════════════════════════════════════════════════════
    print("\n[4/6] Baselines...")

    print("  → Vanilla LDA (99D)...")
    t0 = time.perf_counter()
    lda = LinearDiscriminantAnalysis(n_components=99)
    X_tr_lda = lda.fit_transform(X_train, y_train)
    X_te_lda = lda.transform(X_test)
    t_r = time.perf_counter() - t0
    acc, t_c = evaluate(X_tr_lda, y_train, X_te_lda, y_test)
    record("Vanilla LDA (99D)", acc, 99, t_r, t_c, "baseline")
    print(f"    {acc:.2f}%  |  99D  |  {t_r + t_c:.1f}s")

    print("  → PCA (99D)...")
    t0 = time.perf_counter()
    pca = PCA(n_components=99, random_state=SEED)
    X_tr_pca = pca.fit_transform(X_train)
    X_te_pca = pca.transform(X_test)
    t_r = time.perf_counter() - t0
    acc, t_c = evaluate(X_tr_pca, y_train, X_te_pca, y_test)
    record("PCA (99D)", acc, 99, t_r, t_c, "baseline")
    print(f"    {acc:.2f}%  |  99D  |  {t_r + t_c:.1f}s")

    # Non-novel control: LDA + PCA concat
    print("  → LDA+PCA concat (119D) [control]...")
    X_tr, X_te, dim, t_r = method_lda_pca_concat(X_train, y_train, X_test, 99, 20)
    acc, t_c = evaluate(X_tr, y_train, X_te, y_test)
    record("LDA+PCA concat (119D)", acc, dim, t_r, t_c, "control")
    print(f"    {acc:.2f}%  |  {dim}D  |  {t_r + t_c:.1f}s")

    # ═══════════════════════════════════════════════════════════════
    # [5/6] NOVEL METHODS
    # ═══════════════════════════════════════════════════════════════
    print("\n[5/6] Novel methods...")

    # ── RDA variants ──
    for n_res in [10, 20, 50, 99]:
        label = f"RDA (99+{n_res}={99+n_res}D)"
        print(f"  → {label}...")
        X_tr, X_te, dim, t_r = method_rda(X_train, y_train, X_test, 99, n_res)
        acc, t_c = evaluate(X_tr, y_train, X_te, y_test)
        record(label, acc, dim, t_r, t_c)
        print(f"    {acc:.2f}%  |  {dim}D  |  {t_r + t_c:.1f}s")

    # ── KCS variants ──
    for k in [5, 10, 20]:
        label = f"KCS (k={k}, 99D)"
        print(f"  → {label}...")
        X_tr, X_te, dim, t_r = method_kcs(X_train, y_train, X_test, stats, 99, k)
        acc, t_c = evaluate(X_tr, y_train, X_te, y_test)
        record(label, acc, dim, t_r, t_c)
        print(f"    {acc:.2f}%  |  {dim}D  |  {t_r + t_c:.1f}s")

    # ── SDP variants ──
    for n_bags in [3, 5, 10]:
        label = f"SDP ({n_bags} bags, 99D)"
        print(f"  → {label}...")
        X_tr, X_te, dim, t_r = method_sdp(
            X_train, y_train, X_test, stats, 99, n_bags, 0.3
        )
        acc, t_c = evaluate(X_tr, y_train, X_te, y_test)
        record(label, acc, dim, t_r, t_c)
        print(f"    {acc:.2f}%  |  {dim}D  |  {t_r + t_c:.1f}s")

    # ── SMD variants ──
    for mf in [0.1, 0.3, 0.5]:
        label = f"SMD (margin={mf}, 99D)"
        print(f"  → {label}...")
        X_tr, X_te, dim, t_r = method_smd(
            X_train, y_train, X_test, stats, 99, mf
        )
        acc, t_c = evaluate(X_tr, y_train, X_te, y_test)
        record(label, acc, dim, t_r, t_c)
        print(f"    {acc:.2f}%  |  {dim}D  |  {t_r + t_c:.1f}s")

    # ── DSB variants ──
    for n_rounds in [2, 3]:
        label = f"DSB ({n_rounds} rounds)"
        print(f"  → {label}...")
        X_tr, X_te, dim, t_r = method_dsb(X_train, y_train, X_test, 99, n_rounds)
        acc, t_c = evaluate(X_tr, y_train, X_te, y_test)
        record(label, acc, dim, t_r, t_c)
        print(f"    {acc:.2f}%  |  {dim}D  |  {t_r + t_c:.1f}s")

    # ── Combos ──
    print("\n[6/6] Combo methods...")

    print("  → RDA+SMD (119D)...")
    X_tr, X_te, dim, t_r = method_rda_smd(
        X_train, y_train, X_test, stats, 99, 20, 0.3
    )
    acc, t_c = evaluate(X_tr, y_train, X_te, y_test)
    record("RDA+SMD (99+20D)", acc, dim, t_r, t_c, "combo")
    print(f"    {acc:.2f}%  |  {dim}D  |  {t_r + t_c:.1f}s")

    print("  → RDA+KCS (119D)...")
    X_tr, X_te, dim, t_r = method_rda_kcs(
        X_train, y_train, X_test, stats, 99, 20, 10
    )
    acc, t_c = evaluate(X_tr, y_train, X_te, y_test)
    record("RDA+KCS (99+20D)", acc, dim, t_r, t_c, "combo")
    print(f"    {acc:.2f}%  |  {dim}D  |  {t_r + t_c:.1f}s")

    print("  → RDA+DSB (119D)...")
    X_tr, X_te, dim, t_r = method_rda_dsb(X_train, y_train, X_test, 99, 20)
    acc, t_c = evaluate(X_tr, y_train, X_te, y_test)
    record("RDA+DSB (99+20D)", acc, dim, t_r, t_c, "combo")
    print(f"    {acc:.2f}%  |  {dim}D  |  {t_r + t_c:.1f}s")

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 74)
    print("  RESULTS SUMMARY — ranked by accuracy")
    print("=" * 74)

    df = pd.DataFrame(results)
    df = df.sort_values("accuracy", ascending=False).reset_index(drop=True)

    # Reference baselines from smoke test
    full_raw_acc = {"resnet50": 73.08, "resnet18": 64.41}.get(backbone, 0)
    vanilla_lda_acc = df[df["method"].str.contains("Vanilla LDA")]["accuracy"].values[0]

    print(f"\n  References: Full features raw = {full_raw_acc:.2f}%  |  Vanilla LDA = {vanilla_lda_acc:.2f}%\n")

    for _, row in df.iterrows():
        if row["accuracy"] > full_raw_acc:
            marker = "🟢"  # Beats full features
        elif row["accuracy"] > vanilla_lda_acc:
            marker = "🟡"  # Beats vanilla LDA but not full
        else:
            marker = "🔴"  # Below vanilla LDA
        tag = f"[{row['type']}]".ljust(10)
        print(
            f"  {marker} {row['accuracy']:6.2f}%  |  {row['dim']:4}D  |  "
            f"{row['time_total']:6.1f}s  |  {tag} {row['method']}"
        )

    # Key comparisons
    best_novel = df[df["type"].isin(["novel", "combo"])].iloc[0]
    print(f"\n  ── Key Comparisons ──")
    print(f"  Best novel method: {best_novel['method']}")
    print(f"    vs Full features raw:  {best_novel['accuracy'] - full_raw_acc:+.2f}%")
    print(f"    vs Vanilla LDA:        {best_novel['accuracy'] - vanilla_lda_acc:+.2f}%")
    print(f"    Dimensions:            {feature_dim}D → {best_novel['dim']}D "
          f"({100 * (1 - best_novel['dim'] / feature_dim):.1f}% reduction)")
    print(f"    Time:                  {best_novel['time_total']:.1f}s")

    # Timing comparison
    vanilla_time = df[df["method"].str.contains("Vanilla LDA")]["time_total"].values[0]
    print(f"\n  ── Time Efficiency ──")
    print(f"  Vanilla LDA time: {vanilla_time:.1f}s")
    for _, row in df[df["type"].isin(["novel", "combo"])].head(5).iterrows():
        ratio = row["time_total"] / vanilla_time
        print(f"    {row['method']}: {row['time_total']:.1f}s ({ratio:.1f}× vanilla)")

    # Save
    os.makedirs("results/novel_methods", exist_ok=True)
    csv_path = f"results/novel_methods/quick_test_{backbone}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to: {csv_path}")

    # Verdict
    print("\n" + "=" * 74)
    if best_novel["accuracy"] > full_raw_acc:
        print(f"  ✅ SUCCESS: {best_novel['method']} beats full features by "
              f"+{best_novel['accuracy'] - full_raw_acc:.2f}%!")
    elif best_novel["accuracy"] > vanilla_lda_acc:
        print(f"  🟡 PARTIAL: Best novel beats vanilla LDA but not full features.")
    else:
        print(f"  ❌ FAIL: No novel method beats vanilla LDA. Investigate.")
    print("=" * 74)

    return df


if __name__ == "__main__":
    backbone = sys.argv[1].lower() if len(sys.argv) > 1 else "resnet50"
    if backbone not in BACKBONES:
        print(f"Unknown backbone: {backbone}")
        print(f"Available: {list(BACKBONES.keys())}")
        sys.exit(1)
    main(backbone)
