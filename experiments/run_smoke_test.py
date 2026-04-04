"""
DG-LDA Smoke Test — Quick Validation Before Full-Scale Experiments
====================================================================
Runs a focused test on a given backbone × CIFAR-100 to verify:
  1. Feature extraction pipeline works
  2. All DG-LDA components produce valid output
  3. DG-LDA modes beat established baselines
  4. Results are in expected ranges vs. academic benchmarks

Usage:
  python experiments/run_smoke_test.py                  # default: resnet18
  python experiments/run_smoke_test.py resnet50          # specific backbone
  python experiments/run_smoke_test.py all               # all 4 backbones

Expected run time: ~5-10 min per backbone (feature extraction + 6 methods)

Academic Reference Points (frozen ImageNet-pretrained linear probe):
  - ResNet-18 → CIFAR-100 Full features: ~64.41% (our confirmed)
  - ResNet-50 → CIFAR-100 Full features: ~79.3% (Kornblith 2019)
  - Our vanilla LDA @ 99D with StandardScaler: 66.88%
  - Our PCA @ 99D: ~62-64%

Author: Research Study
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.extract_features_multi import get_or_extract_cifar100, BACKBONES
from reduction.dg_lda import DGLDA, reduce_with_dglda
from reduction.feature_profiler import profile_feature_space


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════
DATASET = "cifar100"
SEED = 42
MAX_COMPONENTS = 99  # C - 1 for CIFAR-100

# Academic benchmarks for comparison
BENCHMARKS = {
    "ResNet-18 Full (our confirmed)": 64.41,
    "ResNet-18 LDA@99 + Scaler (our confirmed)": 66.88,
    "ResNet-50 Supervised LP (Kornblith 2019)": 79.3,
    "ResNet-50 SimCLR LP (Ericsson 2021)": 72.1,
    "CLIP ViT-B/32 zero-shot (Radford 2021)": 65.1,
}


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    standardize: bool = True,
) -> tuple:
    """Train LogisticRegression and return accuracy + timing."""
    t0 = time.perf_counter()

    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    clf = LogisticRegression(solver="lbfgs", max_iter=5000, n_jobs=-1, random_state=SEED)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test) * 100
    elapsed = time.perf_counter() - t0
    return acc, elapsed


def run_smoke_test(backbone: str = "resnet18"):
    """Run comprehensive smoke test on a given backbone × CIFAR-100."""

    feature_dim_expected = BACKBONES[backbone]["feature_dim"]

    print("=" * 70)
    print(f"DG-LDA SMOKE TEST — {backbone.upper()} × CIFAR-100 ({feature_dim_expected}D)")
    print("=" * 70)
    t_global_start = time.perf_counter()

    results = []

    # ─── Step 1: Extract Features ───
    print("\n[1/5] Extracting features...")
    t0 = time.perf_counter()
    X_train, y_train, X_test, y_test, feature_dim = get_or_extract_cifar100(backbone)
    t_extract = time.perf_counter() - t0
    print(f"  Done: train={X_train.shape}, test={X_test.shape}, dim={feature_dim}")
    print(f"  Extraction time: {t_extract:.1f}s")

    n_classes = len(np.unique(y_train))
    print(f"  Classes: {n_classes}")

    # ─── Step 2: Baseline — Full Features ───
    print("\n[2/5] Baseline methods...")

    # Full features (no reduction)
    print("  → Full features (no reduction)...")
    acc_full, t_full = train_and_evaluate(X_train, y_train, X_test, y_test, standardize=True)
    results.append({"method": f"Full Features ({feature_dim}D)", "accuracy": acc_full, "dim": feature_dim,
                     "time": t_full, "notes": "StandardScaler + LogReg"})
    print(f"    Accuracy: {acc_full:.2f}%  |  Time: {t_full:.2f}s")

    # Full features without standardization (to match our archived 64.41%)
    acc_full_raw, t_full_raw = train_and_evaluate(X_train, y_train, X_test, y_test, standardize=False)
    results.append({"method": f"Full Features raw ({feature_dim}D)", "accuracy": acc_full_raw, "dim": feature_dim,
                     "time": t_full_raw, "notes": "No scaling + LogReg"})
    print(f"    (raw, no scaler): {acc_full_raw:.2f}%")

    # PCA @ 99 components
    print("  → PCA @ 99 components...")
    t0 = time.perf_counter()
    scaler_pca = StandardScaler()
    X_train_s = scaler_pca.fit_transform(X_train)
    X_test_s = scaler_pca.transform(X_test)
    pca = PCA(n_components=MAX_COMPONENTS, random_state=SEED)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca = pca.transform(X_test_s)
    t_pca_reduce = time.perf_counter() - t0
    acc_pca, t_pca_clf = train_and_evaluate(X_train_pca, y_train, X_test_pca, y_test, standardize=False)
    results.append({"method": "PCA (99D)", "accuracy": acc_pca, "dim": MAX_COMPONENTS,
                     "time": t_pca_reduce + t_pca_clf, "notes": "StandardScaler → PCA → LogReg"})
    print(f"    Accuracy: {acc_pca:.2f}%  |  Time: {t_pca_reduce + t_pca_clf:.2f}s")

    # Vanilla LDA @ 99 components
    print("  → Vanilla LDA @ 99 components...")
    t0 = time.perf_counter()
    lda_sklearn = LinearDiscriminantAnalysis(n_components=MAX_COMPONENTS)
    X_train_lda = lda_sklearn.fit_transform(X_train_s, y_train)
    X_test_lda = lda_sklearn.transform(X_test_s)
    t_lda_reduce = time.perf_counter() - t0
    acc_lda, t_lda_clf = train_and_evaluate(X_train_lda, y_train, X_test_lda, y_test, standardize=False)
    results.append({"method": "Vanilla LDA (99D)", "accuracy": acc_lda, "dim": MAX_COMPONENTS,
                     "time": t_lda_reduce + t_lda_clf, "notes": "StandardScaler → sklearn LDA → LogReg"})
    print(f"    Accuracy: {acc_lda:.2f}%  |  Time: {t_lda_reduce + t_lda_clf:.2f}s")

    # ─── Step 3: Feature Profiling ───
    print("\n[3/5] Feature space profiling...")
    t0 = time.perf_counter()
    profile = profile_feature_space(X_train, y_train, backbone=backbone, dataset=DATASET, verbose=True)
    t_profile = time.perf_counter() - t0
    print(f"\n  Profiling time: {t_profile:.2f}s")
    print(f"  Feature Complexity Score: {profile.feature_complexity_score:.4f}")
    print(f"  Sw condition number: {profile.sw_condition_number:.1f}")
    print(f"  Covariance heterogeneity: {profile.covariance_heterogeneity:.4f}")
    print(f"  Sw effective rank: {profile.sw_effective_rank:.1f}")
    print(f"  Sb effective rank: {profile.sb_effective_rank:.1f}")
    print(f"  DIR @50%: {profile.dir_at_50:.4f}, @90%: {profile.dir_at_90:.4f}, @95%: {profile.dir_at_95:.4f}")

    # ─── Step 4: DG-LDA Modes ───
    print("\n[4/5] DG-LDA methods...")

    dglda_modes = [
        ("DG-LDA [ablation_vanilla]", "ablation_vanilla"),
        ("DG-LDA [regularized]", "regularized"),
        ("DG-LDA [cw_only]", "cw_only"),
        ("DG-LDA [full]", "full"),
    ]

    for label, mode in dglda_modes:
        print(f"  → {label}...")
        try:
            t0 = time.perf_counter()
            # Use standardized features for fair comparison
            X_tr_red, X_te_red, dglda = reduce_with_dglda(
                X_train_s, y_train, X_test_s,
                mode=mode, backbone=backbone, dataset=DATASET, verbose=False,
            )
            t_reduce = time.perf_counter() - t0
            n_comp = X_tr_red.shape[1]

            acc_dglda, t_clf = train_and_evaluate(X_tr_red, y_train, X_te_red, y_test, standardize=False)
            total_time = t_reduce + t_clf

            detail = ""
            if dglda.result_:
                detail = f"n_comp={dglda.result_.selected_components}, α={dglda.result_.shrinkage_alpha:.4f}"

            results.append({
                "method": f"{label} ({n_comp}D)", "accuracy": acc_dglda, "dim": n_comp,
                "time": total_time, "notes": detail,
            })
            print(f"    Accuracy: {acc_dglda:.2f}%  |  Dim: {n_comp}  |  Time: {total_time:.2f}s  |  {detail}")

        except Exception as e:
            print(f"    ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "method": f"{label} (FAILED)", "accuracy": 0.0, "dim": 0,
                "time": 0.0, "notes": str(e),
            })

    # ─── Step 5: Summary ───
    print("\n" + "=" * 70)
    print("SMOKE TEST RESULTS SUMMARY")
    print("=" * 70)

    df = pd.DataFrame(results)
    df = df.sort_values("accuracy", ascending=False).reset_index(drop=True)

    print("\n── Methods (ranked by accuracy) ──\n")
    for _, row in df.iterrows():
        marker = "🟢" if row["accuracy"] > acc_full else "🔴"
        print(f"  {marker} {row['accuracy']:6.2f}%  |  {row['dim']:4}D  |  {row['time']:6.2f}s  |  {row['method']}")

    # Compare against academic benchmarks
    best_dglda = df[df["method"].str.contains("DG-LDA")]["accuracy"].max()
    print("\n── Comparison with Academic Benchmarks ──\n")
    for name, ref_acc in BENCHMARKS.items():
        gap = best_dglda - ref_acc
        symbol = "✅" if gap > 0 else "❌"
        print(f"  {symbol} DG-LDA best ({best_dglda:.2f}%) vs {name} ({ref_acc:.1f}%): {gap:+.2f}%")

    # Key metrics
    print("\n── Key Metrics ──\n")
    full_acc = df[df["method"].str.contains("Full Features") & ~df["method"].str.contains("raw")]["accuracy"].values[0]
    best_dglda_row = df[df["method"].str.contains("DG-LDA")].iloc[0]
    pca_acc = df[df["method"].str.contains("PCA")]["accuracy"].values[0]
    vanilla_lda_acc = df[df["method"].str.contains("Vanilla LDA")]["accuracy"].values[0]

    print(f"  DG-LDA best vs Full features: {best_dglda - full_acc:+.2f}%")
    print(f"  DG-LDA best vs Vanilla LDA:   {best_dglda - vanilla_lda_acc:+.2f}%")
    print(f"  DG-LDA best vs PCA:           {best_dglda - pca_acc:+.2f}%")
    print(f"  Dimensionality reduction:      {feature_dim}D → {best_dglda_row['dim']}D ({100 * (1 - best_dglda_row['dim'] / feature_dim):.1f}% reduction)")

    t_total = time.perf_counter() - t_global_start
    print(f"\n  Total smoke test time: {t_total:.1f}s ({t_total/60:.1f} min)")

    # Save results
    os.makedirs("results/smoke_test", exist_ok=True)
    csv_path = f"results/smoke_test/smoke_test_{backbone}_cifar100.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to: {csv_path}")

    # Verdict
    print("\n" + "=" * 70)
    if best_dglda > full_acc and best_dglda > vanilla_lda_acc:
        print("✅ VERDICT: PASS — DG-LDA beats both Full features and Vanilla LDA")
    elif best_dglda > full_acc:
        print("⚠️  VERDICT: PARTIAL — DG-LDA beats Full features but not Vanilla LDA")
    elif best_dglda > vanilla_lda_acc:
        print("⚠️  VERDICT: PARTIAL — DG-LDA beats Vanilla LDA but not Full features")
    else:
        print("❌ VERDICT: FAIL — DG-LDA does not beat baselines. Investigate.")
    print("=" * 70)

    return df


if __name__ == "__main__":
    # Parse backbone from command line
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "all":
            backbones_to_test = list(BACKBONES.keys())
        elif arg in BACKBONES:
            backbones_to_test = [arg]
        else:
            print(f"Unknown backbone: {arg}")
            print(f"Available: {list(BACKBONES.keys())} or 'all'")
            sys.exit(1)
    else:
        backbones_to_test = ["resnet18"]

    all_results = []
    for bb in backbones_to_test:
        df = run_smoke_test(backbone=bb)
        df["backbone"] = bb
        all_results.append(df)
        print()

    if len(all_results) > 1:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv("results/smoke_test/smoke_test_all_backbones.csv", index=False)
        print(f"\nCombined results saved to results/smoke_test/smoke_test_all_backbones.csv")
