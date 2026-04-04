"""
Winners Smoke Test — Validate top novel methods across backbones
=================================================================
Runs the 3 winning novel methods + baselines on both ResNet-18 and ResNet-50
(and optionally all 4 backbones) to verify gains generalize.

Winners from quick test (ResNet-50):
  1. RDA (99+20=119D)   — 72.45%, 42.6s, +0.39% over vanilla LDA
  2. DSB (2 rounds, 99D) — 72.29%, 30.7s, +0.23% over vanilla LDA
  3. RDA+SMD (119D)      — 72.59%, 75.8s, +0.53% over vanilla LDA

Usage:
  python experiments/run_winners_smoke_test.py                    # resnet18 + resnet50
  python experiments/run_winners_smoke_test.py resnet18            # single backbone
  python experiments/run_winners_smoke_test.py all                 # all 4 backbones

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

def evaluate(X_train, y_train, X_test, y_test):
    """Train LogisticRegression and return (accuracy%, time_seconds)."""
    t0 = time.perf_counter()
    clf = LogisticRegression(solver="lbfgs", max_iter=5000, random_state=SEED)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test) * 100
    return acc, time.perf_counter() - t0


def solve_whitened_eigen(Sb_custom, Sw_inv_sqrt, n_components):
    """Solve Sb in whitened Sw space, return (projection, eigenvalues)."""
    Sb_white = Sw_inv_sqrt.T @ Sb_custom @ Sw_inv_sqrt
    eigenvalues, eigvecs_white = np.linalg.eigh(Sb_white)
    eigenvectors = Sw_inv_sqrt @ eigvecs_white
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, idx[:n_components]], eigenvalues[idx[:n_components]]


# ═══════════════════════════════════════════════════════════════════════
# Winner 1: RDA — Residual Discriminant Augmentation
# ═══════════════════════════════════════════════════════════════════════

def method_rda(X_train, y_train, X_test, n_lda=99, n_residual=20):
    """LDA + PCA on orthogonal complement (zero-redundancy augmentation)."""
    t0 = time.perf_counter()

    lda = LinearDiscriminantAnalysis(n_components=n_lda)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)

    # Orthonormalize LDA directions
    W = lda.scalings_[:, :n_lda]
    Q, _ = np.linalg.qr(W)

    # Project out LDA subspace → residual in orthogonal complement
    X_train_res = X_train - (X_train @ Q) @ Q.T
    X_test_res = X_test - (X_test @ Q) @ Q.T

    # PCA on residual
    pca_res = PCA(n_components=n_residual, random_state=SEED)
    X_train_res_pca = pca_res.fit_transform(X_train_res)
    X_test_res_pca = pca_res.transform(X_test_res)

    X_train_out = np.hstack([X_train_lda, X_train_res_pca])
    X_test_out = np.hstack([X_test_lda, X_test_res_pca])

    t_reduce = time.perf_counter() - t0
    return X_train_out, X_test_out, X_train_out.shape[1], t_reduce


# ═══════════════════════════════════════════════════════════════════════
# Winner 2: DSB — Discriminant Subspace Boosting
# ═══════════════════════════════════════════════════════════════════════

def method_dsb(X_train, y_train, X_test, n_lda=99, n_rounds=2):
    """Boosted projections: re-weight samples by classification error."""
    t0 = time.perf_counter()

    N, D = X_train.shape
    classes = np.unique(y_train)
    C = len(classes)
    n_comp = min(n_lda, C - 1)

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

        # Quick regularization
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

    if X_train_concat.shape[1] > n_lda:
        pca = PCA(n_components=n_lda, random_state=SEED)
        X_train_out = pca.fit_transform(X_train_concat)
        X_test_out = pca.transform(X_test_concat)
        out_dim = n_lda
    else:
        X_train_out = X_train_concat
        X_test_out = X_test_concat
        out_dim = X_train_concat.shape[1]

    t_reduce = time.perf_counter() - t0
    return X_train_out, X_test_out, out_dim, t_reduce


# ═══════════════════════════════════════════════════════════════════════
# Winner 3: RDA+SMD — Augmented Self-Refining Discriminants
# ═══════════════════════════════════════════════════════════════════════

def method_rda_smd(X_train, y_train, X_test, n_lda=99, n_residual=20, margin_fraction=0.3):
    """SMD for better discriminants + RDA for residual recovery."""
    t0 = time.perf_counter()

    classes = np.unique(y_train)
    C = len(classes)
    N, D = X_train.shape
    global_mean = X_train.mean(axis=0)

    # Compute class means
    class_means = np.zeros((C, D), dtype=np.float64)
    class_counts = np.zeros(C)
    X_centered = np.zeros_like(X_train, dtype=np.float64)
    for i, c in enumerate(classes):
        mask = y_train == c
        class_means[i] = X_train[mask].mean(axis=0)
        class_counts[i] = mask.sum()
        X_centered[mask] = X_train[mask] - class_means[i]

    # Sw + regularization
    Sw = X_centered.T @ X_centered / N
    subsample_size = min(10000, N)
    _, alpha = ledoit_wolf(X_centered[:subsample_size])
    target = np.trace(Sw) / D * np.eye(D)
    Sw_reg = (1 - alpha) * Sw + alpha * target

    eigvals, eigvecs = np.linalg.eigh(Sw_reg)
    eigvals = np.maximum(eigvals, 1e-10)
    Sw_inv_sqrt = eigvecs * (1.0 / np.sqrt(eigvals))[np.newaxis, :]

    # Sb standard
    diffs = class_means - global_mean
    Sb = (diffs * class_counts[:, np.newaxis]).T @ diffs / N

    # SMD Pass 1: Standard LDA projection
    W1, _ = solve_whitened_eigen(Sb, Sw_inv_sqrt, n_lda)

    # SMD Pass 2: Find hard class pairs in LDA space
    class_means_proj = (class_means - global_mean) @ W1
    n_total_pairs = C * (C - 1) // 2
    n_hard = max(1, int(n_total_pairs * margin_fraction))

    pair_dists = []
    for i in range(C):
        for j in range(i + 1, C):
            dist = np.linalg.norm(class_means_proj[i] - class_means_proj[j])
            pair_dists.append((i, j, dist))
    pair_dists.sort(key=lambda x: x[2])

    # SMD Pass 3: Margin-weighted Sb
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

    # SMD Pass 4: Re-solve
    W_smd, _ = solve_whitened_eigen(Sb_margin, Sw_inv_sqrt, n_lda)

    X_train_smd = (X_train - global_mean) @ W_smd
    X_test_smd = (X_test - global_mean) @ W_smd

    # RDA: Residual from vanilla LDA projection
    lda = LinearDiscriminantAnalysis(n_components=n_lda)
    lda.fit(X_train, y_train)
    W_lda = lda.scalings_[:, :n_lda]
    Q, _ = np.linalg.qr(W_lda)

    X_train_res = X_train - (X_train @ Q) @ Q.T
    X_test_res = X_test - (X_test @ Q) @ Q.T

    pca_res = PCA(n_components=n_residual, random_state=SEED)
    X_train_res_pca = pca_res.fit_transform(X_train_res)
    X_test_res_pca = pca_res.transform(X_test_res)

    X_train_out = np.hstack([X_train_smd, X_train_res_pca])
    X_test_out = np.hstack([X_test_smd, X_test_res_pca])

    t_reduce = time.perf_counter() - t0
    return X_train_out, X_test_out, X_train_out.shape[1], t_reduce


# ═══════════════════════════════════════════════════════════════════════
# Main Runner
# ═══════════════════════════════════════════════════════════════════════

def run_for_backbone(backbone: str):
    """Run all winners + baselines on one backbone × CIFAR-100."""
    feature_dim = BACKBONES[backbone]["feature_dim"]
    n_classes = 100
    max_comp = n_classes - 1  # 99

    print("\n" + "=" * 74)
    print(f"  WINNERS SMOKE TEST — {backbone.upper()} × CIFAR-100 ({feature_dim}D)")
    print("=" * 74)

    # ── Load & standardize ──
    print("\n  Loading features...")
    X_train, y_train, X_test, y_test, fdim = get_or_extract_cifar100(backbone)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    print(f"  train={X_train.shape}, test={X_test.shape}")

    results = []

    def run_method(label, method_fn, method_type="novel", **kwargs):
        print(f"  → {label}...", end="", flush=True)
        try:
            X_tr, X_te, dim, t_r = method_fn(**kwargs)
            acc, t_c = evaluate(X_tr, y_train, X_te, y_test)
            total = t_r + t_c
            results.append({
                "backbone": backbone, "method": label, "accuracy": round(acc, 2),
                "dim": dim, "time_reduce": round(t_r, 2), "time_classify": round(t_c, 2),
                "time_total": round(total, 2), "type": method_type,
            })
            print(f"  {acc:.2f}%  |  {dim}D  |  {total:.1f}s")
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            import traceback; traceback.print_exc()
            results.append({
                "backbone": backbone, "method": f"{label} (FAILED)", "accuracy": 0.0,
                "dim": 0, "time_reduce": 0, "time_classify": 0,
                "time_total": 0, "type": method_type,
            })

    # ── Baselines ──
    print("\n  Baselines:")

    # Full features raw
    print(f"  → Full raw ({feature_dim}D)...", end="", flush=True)
    acc_full_raw, t_full_raw = evaluate(X_train, y_train, X_test, y_test)
    results.append({"backbone": backbone, "method": f"Full raw ({feature_dim}D)",
                     "accuracy": round(acc_full_raw, 2), "dim": feature_dim,
                     "time_reduce": 0, "time_classify": round(t_full_raw, 2),
                     "time_total": round(t_full_raw, 2), "type": "baseline"})
    print(f"  {acc_full_raw:.2f}%  |  {feature_dim}D  |  {t_full_raw:.1f}s")

    # Full features scaled
    print(f"  → Full scaled ({feature_dim}D)...", end="", flush=True)
    acc_full_s, t_full_s = evaluate(X_train_s, y_train, X_test_s, y_test)
    results.append({"backbone": backbone, "method": f"Full scaled ({feature_dim}D)",
                     "accuracy": round(acc_full_s, 2), "dim": feature_dim,
                     "time_reduce": 0, "time_classify": round(t_full_s, 2),
                     "time_total": round(t_full_s, 2), "type": "baseline"})
    print(f"  {acc_full_s:.2f}%  |  {feature_dim}D  |  {t_full_s:.1f}s")

    # Vanilla LDA
    print(f"  → Vanilla LDA (99D)...", end="", flush=True)
    t0 = time.perf_counter()
    lda = LinearDiscriminantAnalysis(n_components=max_comp)
    X_tr_lda = lda.fit_transform(X_train_s, y_train)
    X_te_lda = lda.transform(X_test_s)
    t_r = time.perf_counter() - t0
    acc_lda, t_c = evaluate(X_tr_lda, y_train, X_te_lda, y_test)
    results.append({"backbone": backbone, "method": "Vanilla LDA (99D)",
                     "accuracy": round(acc_lda, 2), "dim": max_comp,
                     "time_reduce": round(t_r, 2), "time_classify": round(t_c, 2),
                     "time_total": round(t_r + t_c, 2), "type": "baseline"})
    print(f"  {acc_lda:.2f}%  |  99D  |  {t_r + t_c:.1f}s")

    # PCA
    print(f"  → PCA (99D)...", end="", flush=True)
    t0 = time.perf_counter()
    pca = PCA(n_components=max_comp, random_state=SEED)
    X_tr_pca = pca.fit_transform(X_train_s)
    X_te_pca = pca.transform(X_test_s)
    t_r = time.perf_counter() - t0
    acc_pca, t_c = evaluate(X_tr_pca, y_train, X_te_pca, y_test)
    results.append({"backbone": backbone, "method": "PCA (99D)",
                     "accuracy": round(acc_pca, 2), "dim": max_comp,
                     "time_reduce": round(t_r, 2), "time_classify": round(t_c, 2),
                     "time_total": round(t_r + t_c, 2), "type": "baseline"})
    print(f"  {acc_pca:.2f}%  |  99D  |  {t_r + t_c:.1f}s")

    # ── Novel Winners ──
    print("\n  Novel Winners:")

    # RDA variants
    for n_res in [10, 20, 30]:
        total_d = max_comp + n_res
        run_method(
            f"RDA (99+{n_res}={total_d}D)",
            lambda nr=n_res: method_rda(X_train_s, y_train, X_test_s, max_comp, nr),
        )

    # DSB
    run_method(
        "DSB (2 rounds, 99D)",
        lambda: method_dsb(X_train_s, y_train, X_test_s, max_comp, 2),
    )

    # RDA+SMD variants
    for n_res in [10, 20, 30]:
        total_d = max_comp + n_res
        run_method(
            f"RDA+SMD (99+{n_res}={total_d}D)",
            lambda nr=n_res: method_rda_smd(X_train_s, y_train, X_test_s, max_comp, nr, 0.3),
            method_type="combo",
        )

    # ── Summary ──
    print("\n" + "-" * 74)
    print(f"  SUMMARY — {backbone.upper()} (ranked by accuracy)")
    print("-" * 74)

    df = pd.DataFrame(results)
    df = df.sort_values("accuracy", ascending=False).reset_index(drop=True)

    vanilla_lda_acc = df[df["method"].str.contains("Vanilla LDA")]["accuracy"].values[0]
    full_raw_acc = df[df["method"].str.contains("Full raw")]["accuracy"].values[0]

    for _, row in df.iterrows():
        if row["accuracy"] > full_raw_acc:
            marker = "🟢"
        elif row["accuracy"] > vanilla_lda_acc:
            marker = "🟡"
        else:
            marker = "🔴"
        delta_lda = row["accuracy"] - vanilla_lda_acc
        tag = f"[{row['type']}]".ljust(10)
        print(f"  {marker} {row['accuracy']:6.2f}%  ({delta_lda:+.2f}% vs LDA)  |  "
              f"{row['dim']:4}D  |  {row['time_total']:6.1f}s  |  {tag} {row['method']}")

    # Best novel
    novel_df = df[df["type"].isin(["novel", "combo"])]
    if len(novel_df) > 0:
        best = novel_df.iloc[0]
        print(f"\n  🏆 Best novel: {best['method']}")
        print(f"     vs Full raw:    {best['accuracy'] - full_raw_acc:+.2f}%")
        print(f"     vs Vanilla LDA: {best['accuracy'] - vanilla_lda_acc:+.2f}%")
        print(f"     Dim reduction:  {feature_dim}D → {best['dim']}D "
              f"({100 * (1 - best['dim'] / feature_dim):.1f}%)")

    return df


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "all":
            backbones = list(BACKBONES.keys())
        elif arg in BACKBONES:
            backbones = [arg]
        else:
            print(f"Unknown: {arg}. Available: {list(BACKBONES.keys())} or 'all'")
            sys.exit(1)
    else:
        backbones = ["resnet18", "resnet50"]

    all_dfs = []
    for bb in backbones:
        df = run_for_backbone(bb)
        all_dfs.append(df)

    # ── Cross-backbone comparison ──
    if len(all_dfs) > 1:
        print("\n" + "=" * 74)
        print("  CROSS-BACKBONE COMPARISON")
        print("=" * 74)

        combined = pd.concat(all_dfs, ignore_index=True)

        # For each backbone, show best novel vs baselines
        for bb in backbones:
            bb_df = combined[combined["backbone"] == bb]
            novel_df = bb_df[bb_df["type"].isin(["novel", "combo"])]
            if len(novel_df) == 0:
                continue
            best = novel_df.iloc[0]
            vanilla = bb_df[bb_df["method"].str.contains("Vanilla LDA")].iloc[0]
            full_raw = bb_df[bb_df["method"].str.contains("Full raw")].iloc[0]
            fdim = BACKBONES[bb]["feature_dim"]

            print(f"\n  {bb.upper()} ({fdim}D):")
            print(f"    Full raw:      {full_raw['accuracy']:.2f}%")
            print(f"    Vanilla LDA:   {vanilla['accuracy']:.2f}%")
            print(f"    Best novel:    {best['accuracy']:.2f}%  ({best['method']})")
            print(f"      vs Full raw:    {best['accuracy'] - full_raw['accuracy']:+.2f}%")
            print(f"      vs Vanilla LDA: {best['accuracy'] - vanilla['accuracy']:+.2f}%")
            vanilla_time = vanilla["time_total"]
            print(f"      Time: {best['time_total']:.1f}s ({best['time_total']/vanilla_time:.1f}× vanilla)")

        # Save combined
        os.makedirs("results/novel_methods", exist_ok=True)
        csv_path = "results/novel_methods/winners_smoke_test.csv"
        combined.to_csv(csv_path, index=False)
        print(f"\n  Results saved to: {csv_path}")
    else:
        os.makedirs("results/novel_methods", exist_ok=True)
        csv_path = f"results/novel_methods/winners_{backbones[0]}.csv"
        all_dfs[0].to_csv(csv_path, index=False)
        print(f"\n  Results saved to: {csv_path}")

    print("\n" + "=" * 74)
    print("  ✅ WINNERS SMOKE TEST COMPLETE")
    print("=" * 74)


if __name__ == "__main__":
    main()
