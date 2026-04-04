"""
Phase 3: Statistical Rigor & Additional Experiments
=====================================================

All-in-one script for Phase 3 experiments. Designed to run efficiently on a
single GCP VM and produce all results needed for the paper.

EXPERIMENTS:
  3.1 — Multi-seed significance tests (5 seeds × 7 key methods × 8 configs)
  3.2 — Data efficiency study (4 fractions × 5 key methods × 2 key configs)
  3.3 — Per-class analysis (top-5/bottom-5 classes benefiting from DSB/LDA)
  3.4 — Computational cost analysis (accuracy vs time Pareto, auto-generated)

DESIGN DECISIONS (accuracy-time balance):
  - Drop NCA from significance tests (always Pareto-dominated, 17× slower)
  - Drop RDA+SMD from focus (Pareto 1/8, avg +122s for +0.10%)
  - Keep 7 methods: Full, PCA, LDA, R-LDA, LFDA, RDA, DSB
  - DSB is our best (accuracy champion, 6/8 Pareto, consistent +0.3%)
  - RDA is our versatile option (5/8 Pareto, good acc/time tradeoff)
  - LDA is the core story (6/8 Pareto, best efficiency among supervised)

Usage:
  python experiments/run_phase3_experiments.py --all                    # everything
  python experiments/run_phase3_experiments.py --significance           # 3.1 only
  python experiments/run_phase3_experiments.py --data-efficiency        # 3.2 only
  python experiments/run_phase3_experiments.py --per-class              # 3.3 only
  python experiments/run_phase3_experiments.py --cost-analysis          # 3.4 only
"""

import os
import sys
import time
import warnings
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.covariance import ledoit_wolf
from scipy.linalg import eigh as scipy_eigh
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Project imports ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.extract_features_multi import (
    get_or_extract_cifar100,
    get_or_extract_tiny_imagenet,
    BACKBONES,
)

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

SEEDS = [42, 123, 456, 789, 1024]  # 5 seeds for significance tests

# Focus methods for significance: Pareto-relevant + key comparisons
SIGNIFICANCE_METHODS = ["Full", "PCA", "LDA", "R-LDA", "LFDA", "RDA", "DSB"]

# Data efficiency configs: 2 representative backbones × 1 dataset
DATA_EFFICIENCY_CONFIGS = [
    ("resnet18", "cifar100"),     # low-dim backbone (512D)
    ("resnet50", "cifar100"),     # high-dim backbone (2048D)
]
DATA_FRACTIONS = [0.1, 0.25, 0.5, 1.0]
DATA_EFFICIENCY_METHODS = ["Full", "PCA", "LDA", "DSB", "RDA"]

# Per-class analysis: 2 configs where DSB wins most
PER_CLASS_CONFIGS = [
    ("resnet18", "cifar100"),
    ("mobilenetv3", "cifar100"),
]

ALL_BACKBONES = list(BACKBONES.keys())
ALL_DATASETS = ["cifar100", "tiny_imagenet"]

RESULTS_DIR = "results/phase3"


# ═══════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES (from run_academic_benchmark.py — self-contained)
# ═══════════════════════════════════════════════════════════════════════

def evaluate(X_train: np.ndarray, y_train: np.ndarray,
             X_test: np.ndarray, y_test: np.ndarray,
             seed: int = 42) -> Tuple[float, float]:
    """Standardize, train LogReg, return (accuracy%, classify_seconds)."""
    t0 = time.perf_counter()
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    clf = LogisticRegression(solver="lbfgs", max_iter=5000, random_state=seed)
    clf.fit(X_train_std, y_train)
    acc = clf.score(X_test_std, y_test) * 100
    return acc, time.perf_counter() - t0


def evaluate_per_class(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       seed: int = 42) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return (overall_acc, per_class_acc_array, classes)."""
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    clf = LogisticRegression(solver="lbfgs", max_iter=5000, random_state=seed)
    clf.fit(X_train_std, y_train)
    preds = clf.predict(X_test_std)
    classes = np.unique(y_test)
    per_class = np.zeros(len(classes))
    for i, c in enumerate(classes):
        mask = y_test == c
        per_class[i] = (preds[mask] == c).mean() * 100
    return (preds == y_test).mean() * 100, per_class, classes


def solve_whitened_eigen(Sb: np.ndarray, Sw_inv_sqrt: np.ndarray,
                         n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    """Solve Sb in whitened Sw space → (projection W, eigenvalues)."""
    Sb_white = Sw_inv_sqrt.T @ Sb @ Sw_inv_sqrt
    eigenvalues, eigvecs_white = np.linalg.eigh(Sb_white)
    eigenvectors = Sw_inv_sqrt @ eigvecs_white
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, idx[:n_components]], eigenvalues[idx[:n_components]]


# ═══════════════════════════════════════════════════════════════════════
#  METHOD IMPLEMENTATIONS (self-contained, seed-parameterized)
# ═══════════════════════════════════════════════════════════════════════

def method_full(X_train, y_train, X_test, n_components, seed=42, **kw):
    return X_train.copy(), X_test.copy(), X_train.shape[1], 0.0


def method_pca(X_train, y_train, X_test, n_components, seed=42, **kw):
    t0 = time.perf_counter()
    pca = PCA(n_components=n_components, random_state=seed)
    X_tr = pca.fit_transform(X_train)
    X_te = pca.transform(X_test)
    return X_tr, X_te, n_components, time.perf_counter() - t0


def method_lda(X_train, y_train, X_test, n_components, seed=42, **kw):
    t0 = time.perf_counter()
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_tr = lda.fit_transform(X_train, y_train)
    X_te = lda.transform(X_test)
    return X_tr, X_te, n_components, time.perf_counter() - t0


def method_rlda(X_train, y_train, X_test, n_components, seed=42, **kw):
    t0 = time.perf_counter()
    lda = LinearDiscriminantAnalysis(
        n_components=n_components, solver="eigen", shrinkage="auto"
    )
    X_tr = lda.fit_transform(X_train, y_train)
    X_te = lda.transform(X_test)
    return X_tr, X_te, n_components, time.perf_counter() - t0


def method_lfda(X_train, y_train, X_test, n_components, seed=42, **kw):
    t0 = time.perf_counter()
    k = kw.get("k", 7)
    pca_preprocess = kw.get("pca_preprocess", 300)
    N, D = X_train.shape
    classes = np.unique(y_train)
    C = len(classes)

    pca_pre = None
    if D > pca_preprocess:
        pca_pre = PCA(n_components=pca_preprocess, random_state=seed)
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
    X_c_data = X_work - mean_work
    St = X_c_data.T @ X_c_data / N
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


def method_rda(X_train, y_train, X_test, n_components, seed=42, **kw):
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
    pca_res = PCA(n_components=n_res, random_state=seed)
    X_train_res_pca = pca_res.fit_transform(X_train_res)
    X_test_res_pca = pca_res.transform(X_test_res)
    X_train_out = np.hstack([X_train_lda, X_train_res_pca])
    X_test_out = np.hstack([X_test_lda, X_test_res_pca])
    return X_train_out, X_test_out, X_train_out.shape[1], time.perf_counter() - t0


def method_dsb(X_train, y_train, X_test, n_components, seed=42, **kw):
    n_rounds = kw.get("n_rounds", 2)
    t0 = time.perf_counter()
    N, D = X_train.shape
    classes = np.unique(y_train)
    C = len(classes)
    n_comp = min(n_components, C - 1)
    sample_weights = np.ones(N, dtype=np.float64) / N
    all_train_features, all_test_features = [], []

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
            clf_boost = LogisticRegression(solver="lbfgs", max_iter=500, random_state=seed)
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
        pca = PCA(n_components=n_comp, random_state=seed)
        X_train_out = pca.fit_transform(X_train_concat)
        X_test_out = pca.transform(X_test_concat)
        out_dim = n_comp
    else:
        X_train_out = X_train_concat
        X_test_out = X_test_concat
        out_dim = X_train_concat.shape[1]
    return X_train_out, X_test_out, out_dim, time.perf_counter() - t0


# ═══════════════════════════════════════════════════════════════════════
#  METHOD REGISTRY
# ═══════════════════════════════════════════════════════════════════════

def get_method(name: str, n_components: int, feature_dim: int) -> Dict:
    """Get a single method config by name."""
    n_residual = 20 if feature_dim <= 576 else (25 if feature_dim <= 1280 else 30)
    registry = {
        "Full": {"fn": method_full, "kwargs": {}, "label": "Full"},
        "PCA": {"fn": method_pca, "kwargs": {}, "label": f"PCA ({n_components}D)"},
        "LDA": {"fn": method_lda, "kwargs": {}, "label": f"LDA ({n_components}D)"},
        "R-LDA": {"fn": method_rlda, "kwargs": {}, "label": f"R-LDA ({n_components}D)"},
        "LFDA": {"fn": method_lfda, "kwargs": {"k": 7, "pca_preprocess": 300},
                 "label": f"LFDA ({n_components}D)"},
        "RDA": {"fn": method_rda, "kwargs": {"n_residual": n_residual},
                "label": f"RDA ({n_components}+{n_residual}D)"},
        "DSB": {"fn": method_dsb, "kwargs": {"n_rounds": 2},
                "label": f"DSB (2 rounds, {n_components}D)"},
    }
    return registry[name]


def load_features(backbone: str, dataset: str):
    """Load features, return (X_train_std, y_train, X_test_std, y_test, n_classes, feature_dim)."""
    if dataset == "cifar100":
        X_train, y_train, X_test, y_test, fdim = get_or_extract_cifar100(backbone)
        n_classes = 100
    elif dataset == "tiny_imagenet":
        X_train, y_train, X_test, y_test, fdim = get_or_extract_tiny_imagenet(backbone)
        n_classes = 200
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, y_train, X_test_s, y_test, n_classes, fdim


# ═══════════════════════════════════════════════════════════════════════
#  3.1 — MULTI-SEED SIGNIFICANCE TESTS
# ═══════════════════════════════════════════════════════════════════════

def run_significance_tests():
    """
    5 seeds × 7 methods × 8 configs = 280 runs.
    For each config, compute paired t-test and Wilcoxon signed-rank test
    for all method pairs vs LDA.

    Seed variation comes from the LogisticRegression random_state, which
    affects the classifier training. Reduction methods (LDA, PCA) are
    deterministic but the classifier isn't (due to data ordering in lbfgs).
    We also shuffle the training data per seed for additional variation.
    """
    print("\n" + "=" * 80)
    print("  PHASE 3.1: MULTI-SEED SIGNIFICANCE TESTS")
    print(f"  {len(SEEDS)} seeds × {len(SIGNIFICANCE_METHODS)} methods × "
          f"{len(ALL_BACKBONES)}bb × {len(ALL_DATASETS)}ds = "
          f"{len(SEEDS) * len(SIGNIFICANCE_METHODS) * len(ALL_BACKBONES) * len(ALL_DATASETS)} total runs")
    print("=" * 80)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = []
    run_count = 0
    total_runs = len(ALL_BACKBONES) * len(ALL_DATASETS) * len(SIGNIFICANCE_METHODS) * len(SEEDS)

    for backbone in ALL_BACKBONES:
        for dataset in ALL_DATASETS:
            print(f"\n  ── {backbone} × {dataset} ──")
            X_train, y_train, X_test, y_test, n_classes, fdim = load_features(backbone, dataset)
            n_components = n_classes - 1

            for method_name in SIGNIFICANCE_METHODS:
                mcfg = get_method(method_name, n_components, fdim)

                for seed in SEEDS:
                    run_count += 1
                    # Shuffle training data with seed for variation
                    rng = np.random.RandomState(seed)
                    perm = rng.permutation(len(y_train))
                    X_tr_s = X_train[perm]
                    y_tr_s = y_train[perm]

                    t_start = time.perf_counter()
                    try:
                        X_tr_proj, X_te_proj, dim, t_reduce = mcfg["fn"](
                            X_tr_s, y_tr_s, X_test, n_components,
                            seed=seed, **mcfg["kwargs"]
                        )
                        acc, t_classify = evaluate(X_tr_proj, y_tr_s, X_te_proj, y_test, seed=seed)
                        t_total = t_reduce + t_classify
                    except Exception as e:
                        print(f"    ❌ {method_name} seed={seed}: {e}")
                        acc, dim, t_reduce, t_classify, t_total = 0.0, 0, 0.0, 0.0, 0.0

                    all_results.append({
                        "backbone": backbone, "dataset": dataset,
                        "method": method_name, "label": mcfg["label"],
                        "seed": seed, "accuracy": round(acc, 3),
                        "dim": dim, "time_reduce": round(t_reduce, 2),
                        "time_classify": round(t_classify, 2),
                        "time_total": round(t_total, 2),
                    })

                    if run_count % 10 == 0 or run_count == total_runs:
                        print(f"    [{run_count}/{total_runs}] {method_name} seed={seed}: "
                              f"{acc:.2f}% in {t_total:.1f}s")

    # Save raw results
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(RESULTS_DIR, "multi_seed_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  📄 Raw results: {csv_path}")

    # ── Statistical tests ──
    print("\n  ── STATISTICAL SIGNIFICANCE (vs LDA) ──")
    sig_results = []

    for backbone in ALL_BACKBONES:
        for dataset in ALL_DATASETS:
            mask_config = (df["backbone"] == backbone) & (df["dataset"] == dataset)
            lda_accs = df[mask_config & (df["method"] == "LDA")].sort_values("seed")["accuracy"].values

            if len(lda_accs) != len(SEEDS):
                continue

            for method_name in SIGNIFICANCE_METHODS:
                if method_name == "LDA":
                    continue
                method_accs = df[mask_config & (df["method"] == method_name)].sort_values("seed")["accuracy"].values
                if len(method_accs) != len(SEEDS):
                    continue

                # Paired t-test
                t_stat, p_ttest = stats.ttest_rel(method_accs, lda_accs)

                # Wilcoxon signed-rank test
                diffs = method_accs - lda_accs
                if np.all(diffs == 0):
                    p_wilcoxon = 1.0
                    w_stat = 0.0
                else:
                    try:
                        w_stat, p_wilcoxon = stats.wilcoxon(diffs)
                    except ValueError:
                        p_wilcoxon = 1.0
                        w_stat = 0.0

                mean_diff = method_accs.mean() - lda_accs.mean()
                method_mean = method_accs.mean()
                method_std = method_accs.std()
                lda_mean = lda_accs.mean()
                lda_std = lda_accs.std()

                # Mean time
                method_times = df[mask_config & (df["method"] == method_name)]["time_total"].values
                lda_times = df[mask_config & (df["method"] == "LDA")]["time_total"].values
                time_ratio = method_times.mean() / max(lda_times.mean(), 0.1)

                sig_results.append({
                    "backbone": backbone, "dataset": dataset,
                    "method": method_name,
                    "method_mean": round(method_mean, 3),
                    "method_std": round(method_std, 3),
                    "lda_mean": round(lda_mean, 3),
                    "lda_std": round(lda_std, 3),
                    "mean_diff": round(mean_diff, 3),
                    "t_stat": round(t_stat, 4),
                    "p_ttest": round(p_ttest, 6),
                    "w_stat": round(w_stat, 4) if w_stat else 0,
                    "p_wilcoxon": round(p_wilcoxon, 6),
                    "significant_005": p_ttest < 0.05,
                    "significant_001": p_ttest < 0.01,
                    "time_ratio_vs_lda": round(time_ratio, 2),
                })

                sig_mark = "***" if p_ttest < 0.01 else ("**" if p_ttest < 0.05 else "ns")
                print(f"    {backbone}/{dataset}: {method_name} vs LDA: "
                      f"Δ={mean_diff:+.3f}% p={p_ttest:.4f} {sig_mark} "
                      f"(time: {time_ratio:.1f}× LDA)")

    sig_df = pd.DataFrame(sig_results)
    sig_path = os.path.join(RESULTS_DIR, "significance_tests.csv")
    sig_df.to_csv(sig_path, index=False)
    print(f"\n  📄 Significance tests: {sig_path}")

    return df, sig_df


# ═══════════════════════════════════════════════════════════════════════
#  3.2 — DATA EFFICIENCY STUDY
# ═══════════════════════════════════════════════════════════════════════

def run_data_efficiency():
    """
    Test how methods perform with limited training data.
    4 fractions × 5 methods × 2 configs × 3 seeds = 120 runs.
    """
    print("\n" + "=" * 80)
    print("  PHASE 3.2: DATA EFFICIENCY STUDY")
    print(f"  {len(DATA_FRACTIONS)} fractions × {len(DATA_EFFICIENCY_METHODS)} methods × "
          f"{len(DATA_EFFICIENCY_CONFIGS)} configs × 3 seeds")
    print("=" * 80)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    efficiency_seeds = SEEDS[:3]  # Use 3 seeds for efficiency study
    results = []
    run_count = 0

    for backbone, dataset in DATA_EFFICIENCY_CONFIGS:
        print(f"\n  ── {backbone} × {dataset} ──")
        X_train_full, y_train_full, X_test, y_test, n_classes, fdim = load_features(backbone, dataset)
        n_components = n_classes - 1

        for fraction in DATA_FRACTIONS:
            for seed in efficiency_seeds:
                # Stratified subsample
                rng = np.random.RandomState(seed)
                n_per_class = max(2, int(fraction * len(y_train_full) / n_classes))
                indices = []
                for c in range(n_classes):
                    c_idx = np.where(y_train_full == c)[0]
                    n_take = min(n_per_class, len(c_idx))
                    indices.extend(rng.choice(c_idx, size=n_take, replace=False))
                indices = np.array(indices)
                X_train_sub = X_train_full[indices]
                y_train_sub = y_train_full[indices]

                for method_name in DATA_EFFICIENCY_METHODS:
                    run_count += 1
                    # Adjust n_components for small subsets
                    n_comp_eff = min(n_components, len(np.unique(y_train_sub)) - 1)
                    if n_comp_eff < 1:
                        n_comp_eff = 1
                    mcfg = get_method(method_name, n_comp_eff, fdim)

                    try:
                        X_tr_proj, X_te_proj, dim, t_reduce = mcfg["fn"](
                            X_train_sub, y_train_sub, X_test, n_comp_eff,
                            seed=seed, **mcfg["kwargs"]
                        )
                        acc, t_classify = evaluate(X_tr_proj, y_train_sub, X_te_proj, y_test, seed=seed)
                        t_total = t_reduce + t_classify
                    except Exception as e:
                        print(f"    ❌ {method_name} frac={fraction} seed={seed}: {e}")
                        acc, dim, t_reduce, t_classify, t_total = 0.0, 0, 0.0, 0.0, 0.0

                    results.append({
                        "backbone": backbone, "dataset": dataset,
                        "method": method_name, "fraction": fraction,
                        "n_train_samples": len(y_train_sub),
                        "seed": seed, "accuracy": round(acc, 3),
                        "dim": dim, "time_total": round(t_total, 2),
                    })

                    if run_count % 5 == 0:
                        print(f"    [{run_count}] {method_name} frac={fraction:.0%} "
                              f"seed={seed}: {acc:.2f}%")

    df = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, "data_efficiency.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  📄 Data efficiency: {csv_path}")

    # Summary
    print("\n  ── SUMMARY ──")
    for backbone, dataset in DATA_EFFICIENCY_CONFIGS:
        print(f"\n  {backbone} × {dataset}:")
        mask = (df["backbone"] == backbone) & (df["dataset"] == dataset)
        for method_name in DATA_EFFICIENCY_METHODS:
            print(f"    {method_name:>6s}: ", end="")
            for frac in DATA_FRACTIONS:
                accs = df[mask & (df["method"] == method_name) & (df["fraction"] == frac)]["accuracy"]
                if len(accs) > 0:
                    print(f"{frac:.0%}={accs.mean():.1f}%  ", end="")
            print()

    return df


# ═══════════════════════════════════════════════════════════════════════
#  3.3 — PER-CLASS ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def run_per_class_analysis():
    """
    Detailed per-class accuracy for key methods on key configs.
    Identifies which classes benefit most from DSB vs LDA vs Full.
    """
    print("\n" + "=" * 80)
    print("  PHASE 3.3: PER-CLASS ANALYSIS")
    print("=" * 80)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_class_results = []

    for backbone, dataset in PER_CLASS_CONFIGS:
        print(f"\n  ── {backbone} × {dataset} ──")
        X_train, y_train, X_test, y_test, n_classes, fdim = load_features(backbone, dataset)
        n_components = n_classes - 1

        methods_to_test = ["Full", "PCA", "LDA", "DSB", "RDA"]
        per_class_accs = {}

        for method_name in methods_to_test:
            mcfg = get_method(method_name, n_components, fdim)
            X_tr_proj, X_te_proj, dim, t_reduce = mcfg["fn"](
                X_train, y_train, X_test, n_components, seed=42, **mcfg["kwargs"]
            )
            overall_acc, class_accs, classes = evaluate_per_class(
                X_tr_proj, y_train, X_te_proj, y_test, seed=42
            )
            per_class_accs[method_name] = class_accs
            print(f"    {method_name}: overall={overall_acc:.2f}%")

            for i, c in enumerate(classes):
                all_class_results.append({
                    "backbone": backbone, "dataset": dataset,
                    "method": method_name, "class_id": int(c),
                    "class_accuracy": round(class_accs[i], 2),
                    "overall_accuracy": round(overall_acc, 2),
                })

        # ── DSB vs LDA: which classes benefit? ──
        dsb_accs = per_class_accs.get("DSB", np.zeros(n_classes))
        lda_accs = per_class_accs.get("LDA", np.zeros(n_classes))
        full_accs = per_class_accs.get("Full", np.zeros(n_classes))

        dsb_lda_diff = dsb_accs - lda_accs
        top5_dsb = np.argsort(dsb_lda_diff)[::-1][:5]
        bot5_dsb = np.argsort(dsb_lda_diff)[:5]

        print(f"\n    Top 5 classes where DSB > LDA:")
        for i in top5_dsb:
            print(f"      Class {classes[i]:3d}: DSB={dsb_accs[i]:.1f}%, "
                  f"LDA={lda_accs[i]:.1f}%, Δ={dsb_lda_diff[i]:+.1f}%")

        print(f"    Bottom 5 classes where DSB < LDA:")
        for i in bot5_dsb:
            print(f"      Class {classes[i]:3d}: DSB={dsb_accs[i]:.1f}%, "
                  f"LDA={lda_accs[i]:.1f}%, Δ={dsb_lda_diff[i]:+.1f}%")

        # LDA vs Full
        lda_full_diff = lda_accs - full_accs
        top5_lda = np.argsort(lda_full_diff)[::-1][:5]
        print(f"\n    Top 5 classes where LDA > Full:")
        for i in top5_lda:
            print(f"      Class {classes[i]:3d}: LDA={lda_accs[i]:.1f}%, "
                  f"Full={full_accs[i]:.1f}%, Δ={lda_full_diff[i]:+.1f}%")

    df = pd.DataFrame(all_class_results)
    csv_path = os.path.join(RESULTS_DIR, "per_class_analysis.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  📄 Per-class analysis: {csv_path}")

    return df


# ═══════════════════════════════════════════════════════════════════════
#  3.4 — COMPUTATIONAL COST ANALYSIS (from existing + new data)
# ═══════════════════════════════════════════════════════════════════════

def run_cost_analysis():
    """
    Compute accuracy vs time Pareto analysis.
    Uses Phase 2 results + multi-seed timing data from 3.1.
    """
    print("\n" + "=" * 80)
    print("  PHASE 3.4: COMPUTATIONAL COST ANALYSIS")
    print("=" * 80)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load Phase 2 results
    p2_path = "results/academic_benchmark/all_benchmarks.csv"
    if os.path.exists(p2_path):
        p2_df = pd.read_csv(p2_path)
    else:
        print("  ⚠️  Phase 2 results not found — using multi-seed data only")
        p2_df = pd.DataFrame()

    # Load multi-seed results if available
    ms_path = os.path.join(RESULTS_DIR, "multi_seed_results.csv")
    if os.path.exists(ms_path):
        ms_df = pd.read_csv(ms_path)
    else:
        print("  ⚠️  Multi-seed results not found — using Phase 2 data only")
        ms_df = pd.DataFrame()

    # Build cost analysis from multi-seed data (more reliable timing with means)
    cost_results = []

    if len(ms_df) > 0:
        for backbone in ALL_BACKBONES:
            for dataset in ALL_DATASETS:
                mask = (ms_df["backbone"] == backbone) & (ms_df["dataset"] == dataset)
                config_df = ms_df[mask]

                methods_in_config = config_df["method"].unique()
                method_stats = []

                for method_name in methods_in_config:
                    m_df = config_df[config_df["method"] == method_name]
                    mean_acc = m_df["accuracy"].mean()
                    std_acc = m_df["accuracy"].std()
                    mean_time = m_df["time_total"].mean()
                    std_time = m_df["time_total"].std()
                    mean_dim = m_df["dim"].mean()
                    mean_t_reduce = m_df["time_reduce"].mean()
                    mean_t_classify = m_df["time_classify"].mean()

                    method_stats.append({
                        "backbone": backbone, "dataset": dataset,
                        "method": method_name,
                        "mean_accuracy": round(mean_acc, 3),
                        "std_accuracy": round(std_acc, 3),
                        "mean_time": round(mean_time, 2),
                        "std_time": round(std_time, 2),
                        "mean_dim": round(mean_dim),
                        "mean_time_reduce": round(mean_t_reduce, 2),
                        "mean_time_classify": round(mean_t_classify, 2),
                        "efficiency": round(mean_acc / max(mean_time, 0.1), 3),
                    })

                # Pareto frontier: no other method is both faster AND more accurate
                for i, s in enumerate(method_stats):
                    is_pareto = True
                    for j, s2 in enumerate(method_stats):
                        if i != j:
                            if s2["mean_accuracy"] >= s["mean_accuracy"] and s2["mean_time"] <= s["mean_time"]:
                                if s2["mean_accuracy"] > s["mean_accuracy"] or s2["mean_time"] < s["mean_time"]:
                                    is_pareto = False
                                    break
                    s["is_pareto"] = is_pareto

                cost_results.extend(method_stats)

    elif len(p2_df) > 0:
        # Fallback: use Phase 2 single-run data
        for _, row in p2_df.iterrows():
            cost_results.append({
                "backbone": row["backbone"], "dataset": row["dataset"],
                "method": row["method"], "mean_accuracy": row["accuracy"],
                "std_accuracy": 0, "mean_time": row["time_total"],
                "std_time": 0, "mean_dim": row["dim"],
                "mean_time_reduce": row["time_reduce"],
                "mean_time_classify": row["time_classify"],
                "efficiency": round(row["accuracy"] / max(row["time_total"], 0.1), 3),
                "is_pareto": False,  # will compute below
            })

    cost_df = pd.DataFrame(cost_results)

    if len(cost_df) > 0:
        csv_path = os.path.join(RESULTS_DIR, "cost_analysis.csv")
        cost_df.to_csv(csv_path, index=False)
        print(f"\n  📄 Cost analysis: {csv_path}")

        # Print Pareto summary
        print("\n  ── PARETO-OPTIMAL METHODS (faster AND more accurate than all others) ──")
        for backbone in ALL_BACKBONES:
            for dataset in ALL_DATASETS:
                mask = (cost_df["backbone"] == backbone) & (cost_df["dataset"] == dataset)
                config = cost_df[mask]
                pareto = config[config["is_pareto"] == True]
                if len(pareto) > 0:
                    methods = ", ".join(f"{r['method']}({r['mean_accuracy']:.1f}%/{r['mean_time']:.0f}s)"
                                        for _, r in pareto.iterrows())
                    print(f"    {backbone}/{dataset}: {methods}")

        # Efficiency ranking
        print("\n  ── EFFICIENCY RANKING (accuracy / time) ──")
        eff = cost_df.groupby("method").agg(
            mean_efficiency=("efficiency", "mean"),
            mean_acc=("mean_accuracy", "mean"),
            mean_time=("mean_time", "mean"),
        ).sort_values("mean_efficiency", ascending=False)
        for method, row in eff.iterrows():
            print(f"    {method:>8s}: eff={row['mean_efficiency']:.2f}  "
                  f"(avg acc={row['mean_acc']:.1f}%, avg time={row['mean_time']:.0f}s)")

    return cost_df


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Statistical rigor experiments")
    parser.add_argument("--all", action="store_true", help="Run all Phase 3 experiments")
    parser.add_argument("--significance", action="store_true", help="3.1: Multi-seed significance")
    parser.add_argument("--data-efficiency", action="store_true", help="3.2: Data efficiency study")
    parser.add_argument("--per-class", action="store_true", help="3.3: Per-class analysis")
    parser.add_argument("--cost-analysis", action="store_true", help="3.4: Cost analysis")
    args = parser.parse_args()

    # Default: run all if no specific flag
    run_all = args.all or not any([args.significance, args.data_efficiency,
                                    args.per_class, args.cost_analysis])

    t_session_start = time.perf_counter()

    if run_all or args.significance:
        run_significance_tests()

    if run_all or args.data_efficiency:
        run_data_efficiency()

    if run_all or args.per_class:
        run_per_class_analysis()

    if run_all or args.cost_analysis:
        run_cost_analysis()

    t_session = time.perf_counter() - t_session_start
    print(f"\n{'=' * 80}")
    print(f"  ✅ PHASE 3 COMPLETE — Total time: {t_session/60:.1f} minutes")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
