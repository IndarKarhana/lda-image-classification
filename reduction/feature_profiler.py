"""
Feature Space Profiler (FSP) — DG-LDA Component 1
===================================================
Analyzes the geometric and statistical properties of frozen CNN feature spaces
to predict whether LDA will be effective and guide its configuration.

Metrics computed:
  1. Eigenvalue spectrum of Sw and Sb (effective rank, spectral decay)
  2. Class-conditional covariance heterogeneity (LDA assumes equal Σ_i)
  3. Pairwise Bhattacharyya distance matrix between classes
  4. Multivariate Gaussianity per class (Mardia's skewness & kurtosis)
  5. Feature Complexity Score (FCS) — single summary scalar

Author: Research Study
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.covariance import LedoitWolf
import warnings


@dataclass
class FeatureProfile:
    """Container for all profiling metrics of a feature space."""
    backbone: str
    dataset: str
    feature_dim: int
    n_classes: int
    n_samples: int
    dim_class_ratio: float

    # Eigenvalue spectrum
    sw_eigenvalues: np.ndarray = field(repr=False)
    sb_eigenvalues: np.ndarray = field(repr=False)
    sw_effective_rank: float = 0.0
    sb_effective_rank: float = 0.0
    sw_condition_number: float = 0.0
    sw_spectral_decay_rate: float = 0.0

    # Discriminant eigenvalues (Sw^{-1} Sb)
    discriminant_eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    dir_at_50: float = 0.0  # Discriminant Information Ratio at 50% components
    dir_at_90: float = 0.0
    dir_at_95: float = 0.0
    optimal_components_95: int = 0

    # Covariance heterogeneity
    covariance_heterogeneity: float = 0.0
    class_cov_norms: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)

    # Bhattacharyya distances
    bhattacharyya_matrix: np.ndarray = field(default_factory=lambda: np.array([[]]), repr=False)
    mean_bhattacharyya: float = 0.0
    min_bhattacharyya: float = 0.0
    confused_pairs: list = field(default_factory=list)

    # Gaussianity
    mardia_skewness_pvalues: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    fraction_non_gaussian: float = 0.0

    # Summary
    feature_complexity_score: float = 0.0
    lda_benefit_prediction: str = ""


def compute_scatter_matrices(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray], np.ndarray]:
    """
    Compute within-class (Sw) and between-class (Sb) scatter matrices.

    Args:
        X: Features, shape (N, D)
        y: Labels, shape (N,)

    Returns:
        Sw: Within-class scatter, (D, D)
        Sb: Between-class scatter, (D, D)
        class_means: dict mapping label -> mean vector
        global_mean: overall mean vector
    """
    classes = np.unique(y)
    D = X.shape[1]
    global_mean = X.mean(axis=0)

    Sw = np.zeros((D, D), dtype=np.float64)
    Sb = np.zeros((D, D), dtype=np.float64)
    class_means = {}

    for c in classes:
        Xc = X[y == c]
        nc = Xc.shape[0]
        mc = Xc.mean(axis=0)
        class_means[c] = mc

        # Within-class: sum of (x - mc)(x - mc)^T
        diff = Xc - mc
        Sw += diff.T @ diff

        # Between-class: nc * (mc - mu)(mc - mu)^T
        dm = (mc - global_mean).reshape(-1, 1)
        Sb += nc * (dm @ dm.T)

    return Sw, Sb, class_means, global_mean


def eigenvalue_spectrum_analysis(
    Sw: np.ndarray, Sb: np.ndarray, n_classes: int
) -> Dict:
    """
    Analyze eigenvalue spectra of Sw, Sb, and Sw^{-1}Sb.

    Returns dict with effective ranks, condition numbers, discriminant eigenvalues,
    and Discriminant Information Ratios (DIR).
    """
    results = {}

    # Sw eigenvalues
    sw_eigs = np.linalg.eigvalsh(Sw)
    sw_eigs = np.sort(sw_eigs)[::-1]
    sw_eigs_pos = sw_eigs[sw_eigs > 1e-12]

    results["sw_eigenvalues"] = sw_eigs

    # Effective rank: exp(entropy of normalized eigenvalues)
    if len(sw_eigs_pos) > 0:
        p = sw_eigs_pos / sw_eigs_pos.sum()
        results["sw_effective_rank"] = np.exp(-np.sum(p * np.log(p + 1e-30)))
        results["sw_condition_number"] = sw_eigs_pos[0] / sw_eigs_pos[-1]

        # Spectral decay: slope of log-eigenvalues
        log_eigs = np.log(sw_eigs_pos + 1e-30)
        x = np.arange(len(log_eigs))
        if len(x) > 1:
            slope = np.polyfit(x, log_eigs, 1)[0]
            results["sw_spectral_decay_rate"] = abs(slope)
        else:
            results["sw_spectral_decay_rate"] = 0.0
    else:
        results["sw_effective_rank"] = 0.0
        results["sw_condition_number"] = np.inf
        results["sw_spectral_decay_rate"] = 0.0

    # Sb eigenvalues
    sb_eigs = np.linalg.eigvalsh(Sb)
    sb_eigs = np.sort(sb_eigs)[::-1]
    sb_eigs_pos = sb_eigs[sb_eigs > 1e-12]
    results["sb_eigenvalues"] = sb_eigs

    if len(sb_eigs_pos) > 0:
        p = sb_eigs_pos / sb_eigs_pos.sum()
        results["sb_effective_rank"] = np.exp(-np.sum(p * np.log(p + 1e-30)))
    else:
        results["sb_effective_rank"] = 0.0

    # Discriminant eigenvalues: solve generalized eigenvalue problem
    # Use regularized Sw to avoid singularity
    reg = 1e-6 * np.trace(Sw) / Sw.shape[0]
    Sw_reg = Sw + reg * np.eye(Sw.shape[0])

    try:
        from scipy.linalg import eigh
        disc_eigs, _ = eigh(Sb, Sw_reg)
        disc_eigs = np.sort(np.real(disc_eigs))[::-1]
        # Only C-1 are meaningful
        disc_eigs = disc_eigs[: n_classes - 1]
        disc_eigs = np.maximum(disc_eigs, 0)
    except Exception:
        disc_eigs = np.zeros(n_classes - 1)

    results["discriminant_eigenvalues"] = disc_eigs

    # Discriminant Information Ratio
    total = disc_eigs.sum()
    if total > 0:
        cumsum = np.cumsum(disc_eigs) / total
        results["dir_at_50"] = cumsum[min(len(cumsum) - 1, (n_classes - 1) // 2)]
        results["dir_at_90"] = cumsum[min(len(cumsum) - 1, int(0.9 * (n_classes - 1)))]
        results["dir_at_95"] = cumsum[min(len(cumsum) - 1, int(0.95 * (n_classes - 1)))]

        # Optimal components at 95% threshold
        idx_95 = np.searchsorted(cumsum, 0.95)
        results["optimal_components_95"] = int(min(idx_95 + 1, n_classes - 1))
    else:
        results["dir_at_50"] = 0.0
        results["dir_at_90"] = 0.0
        results["dir_at_95"] = 0.0
        results["optimal_components_95"] = n_classes - 1

    return results


def covariance_heterogeneity(
    X: np.ndarray, y: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Measure how much class-conditional covariances differ.

    LDA assumes all classes share the same covariance. This measures violation.
    Uses the coefficient of variation of Frobenius norms of class covariances.

    Returns:
        heterogeneity: CoV of class covariance Frobenius norms (0 = perfect LDA assumption)
        class_cov_norms: array of Frobenius norms per class
    """
    classes = np.unique(y)
    cov_norms = []

    for c in classes:
        Xc = X[y == c]
        if Xc.shape[0] < 2:
            continue
        # Use Ledoit-Wolf for stable covariance estimation
        try:
            lw = LedoitWolf().fit(Xc)
            cov_norm = np.linalg.norm(lw.covariance_, "fro")
        except Exception:
            cov = np.cov(Xc, rowvar=False)
            cov_norm = np.linalg.norm(cov, "fro")
        cov_norms.append(cov_norm)

    cov_norms = np.array(cov_norms)
    if cov_norms.std() == 0:
        return 0.0, cov_norms

    heterogeneity = cov_norms.std() / cov_norms.mean()
    return float(heterogeneity), cov_norms


def bhattacharyya_distance_matrix(
    X: np.ndarray, y: np.ndarray, max_classes: int = 200
) -> Tuple[np.ndarray, float, float, list]:
    """
    Compute pairwise Bhattacharyya distances between class distributions.

    Uses diagonal covariance approximation for efficiency in high-D.

    Args:
        X: Features (N, D)
        y: Labels (N,)
        max_classes: Maximum classes to compute (for efficiency)

    Returns:
        dist_matrix: (C, C) Bhattacharyya distance matrix
        mean_dist: average pairwise distance
        min_dist: minimum pairwise distance (most confused pair)
        confused_pairs: list of (class_i, class_j, distance) for closest pairs
    """
    classes = np.unique(y)[:max_classes]
    C = len(classes)
    dist_matrix = np.zeros((C, C))

    # Precompute class stats (diagonal covariance for efficiency)
    means = {}
    vars_ = {}
    for i, c in enumerate(classes):
        Xc = X[y == c]
        means[i] = Xc.mean(axis=0)
        vars_[i] = Xc.var(axis=0) + 1e-10  # diagonal covariance

    for i in range(C):
        for j in range(i + 1, C):
            # Bhattacharyya distance with diagonal covariance
            avg_var = 0.5 * (vars_[i] + vars_[j])
            diff_mean = means[i] - means[j]

            # Term 1: Mahalanobis-like term
            term1 = 0.125 * np.sum(diff_mean**2 / avg_var)

            # Term 2: Covariance divergence term
            term2 = 0.5 * np.sum(np.log(avg_var) - 0.5 * np.log(vars_[i]) - 0.5 * np.log(vars_[j]))

            dist_matrix[i, j] = term1 + term2
            dist_matrix[j, i] = dist_matrix[i, j]

    # Extract statistics
    upper_tri = dist_matrix[np.triu_indices(C, k=1)]
    mean_dist = float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0.0
    min_dist = float(np.min(upper_tri)) if len(upper_tri) > 0 else 0.0

    # Find most confused pairs (smallest distances)
    n_confused = min(20, len(upper_tri))
    if n_confused > 0:
        flat_idx = np.argsort(upper_tri)[:n_confused]
        tri_indices = np.triu_indices(C, k=1)
        confused_pairs = [
            (int(classes[tri_indices[0][idx]]),
             int(classes[tri_indices[1][idx]]),
             float(upper_tri[idx]))
            for idx in flat_idx
        ]
    else:
        confused_pairs = []

    return dist_matrix, mean_dist, min_dist, confused_pairs


def test_gaussianity(
    X: np.ndarray, y: np.ndarray, n_subsample: int = 50
) -> Tuple[np.ndarray, float]:
    """
    Test multivariate Gaussianity per class using Mardia's skewness.

    Uses PCA-reduced features (to 20D) for computational feasibility.

    Args:
        X: Features (N, D)
        y: Labels (N,)
        n_subsample: Number of dimensions to keep for testing

    Returns:
        pvalues: p-values for each class (small = non-Gaussian)
        fraction_non_gaussian: fraction of classes with p < 0.05
    """
    from sklearn.decomposition import PCA
    from scipy import stats

    classes = np.unique(y)
    pvalues = []

    # Reduce dimensionality for feasibility
    n_comp = min(n_subsample, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=n_comp)
    X_reduced = pca.fit_transform(X)

    for c in classes:
        Xc = X_reduced[y == c]
        nc, p = Xc.shape

        if nc < p + 2:
            pvalues.append(1.0)  # Can't test, assume Gaussian
            continue

        # Mardia's multivariate skewness (simplified)
        # b1p = (1/n^2) * sum_i sum_j ((xi-mu)' Sigma^-1 (xj-mu))^3
        try:
            mc = Xc.mean(axis=0)
            centered = Xc - mc
            cov = np.cov(centered, rowvar=False)
            cov_inv = np.linalg.pinv(cov)

            # Subsample for speed
            n_sub = min(nc, 200)
            idx = np.random.choice(nc, n_sub, replace=False) if nc > n_sub else np.arange(nc)
            sub = centered[idx]

            mah = sub @ cov_inv @ sub.T  # (n_sub, n_sub) matrix of Mahalanobis products
            b1p = np.mean(mah**3)

            # Under H0 (Gaussian): n*b1p/6 ~ chi2(p*(p+1)*(p+2)/6)
            test_stat = n_sub * b1p / 6.0
            df = p * (p + 1) * (p + 2) / 6.0
            pval = 1.0 - stats.chi2.cdf(test_stat, df)
            pvalues.append(float(pval))
        except Exception:
            pvalues.append(1.0)

    pvalues = np.array(pvalues)
    fraction_ng = float(np.mean(pvalues < 0.05))

    return pvalues, fraction_ng


def compute_feature_complexity_score(profile_metrics: Dict) -> Tuple[float, str]:
    """
    Compute a single Feature Complexity Score (FCS) summarizing how
    challenging the feature space is for LDA.

    Higher FCS = more challenging = LDA less likely to help without adaptation.

    Components (weighted):
      - dim_class_ratio: Higher = harder for LDA (weight: 0.25)
      - covariance_heterogeneity: Higher = violates LDA assumption (weight: 0.25)
      - sw_condition_number (log): Higher = ill-conditioned Sw (weight: 0.20)
      - fraction_non_gaussian: Higher = violates Gaussian assumption (weight: 0.15)
      - 1 - dir_at_95: Less info in top 95% = LDA less effective (weight: 0.15)

    Returns:
        fcs: Score in [0, 1], higher = more complex
        prediction: "high_benefit", "moderate_benefit", or "low_benefit"
    """
    # Normalize each component to [0, 1] range with saturation
    dcr = min(profile_metrics.get("dim_class_ratio", 5.0) / 30.0, 1.0)
    cov_het = min(profile_metrics.get("covariance_heterogeneity", 0.0) / 1.0, 1.0)

    cond = profile_metrics.get("sw_condition_number", 1e6)
    cond_norm = min(np.log10(max(cond, 1.0)) / 10.0, 1.0)

    fng = profile_metrics.get("fraction_non_gaussian", 0.0)

    dir95 = profile_metrics.get("dir_at_95", 1.0)
    dir_component = 1.0 - dir95

    fcs = (
        0.25 * dcr
        + 0.25 * cov_het
        + 0.20 * cond_norm
        + 0.15 * fng
        + 0.15 * dir_component
    )

    if fcs < 0.3:
        prediction = "high_benefit"
    elif fcs < 0.55:
        prediction = "moderate_benefit"
    else:
        prediction = "low_benefit"

    return float(fcs), prediction


def profile_feature_space(
    X_train: np.ndarray,
    y_train: np.ndarray,
    backbone: str = "unknown",
    dataset: str = "unknown",
    verbose: bool = True,
) -> FeatureProfile:
    """
    Run the full Feature Space Profiler on a set of training features.

    Args:
        X_train: Training features, shape (N, D)
        y_train: Training labels, shape (N,)
        backbone: Name of the backbone model
        dataset: Name of the dataset
        verbose: Print progress

    Returns:
        FeatureProfile dataclass with all metrics
    """
    N, D = X_train.shape
    classes = np.unique(y_train)
    C = len(classes)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Feature Space Profiler: {backbone} on {dataset}")
        print(f"  Samples: {N}, Features: {D}, Classes: {C}")
        print(f"  Dim/Class ratio: {D/C:.2f}")
        print(f"{'='*60}")

    # Step 1: Scatter matrices
    if verbose:
        print("  [1/5] Computing scatter matrices...")
    Sw, Sb, class_means, global_mean = compute_scatter_matrices(X_train, y_train)

    # Step 2: Eigenvalue spectrum analysis
    if verbose:
        print("  [2/5] Eigenvalue spectrum analysis...")
    eig_results = eigenvalue_spectrum_analysis(Sw, Sb, C)

    # Step 3: Covariance heterogeneity
    if verbose:
        print("  [3/5] Covariance heterogeneity...")
    cov_het, cov_norms = covariance_heterogeneity(X_train, y_train)

    # Step 4: Bhattacharyya distances
    if verbose:
        print("  [4/5] Bhattacharyya distance matrix...")
    bhat_matrix, mean_bhat, min_bhat, confused_pairs = bhattacharyya_distance_matrix(
        X_train, y_train
    )

    # Step 5: Gaussianity test
    if verbose:
        print("  [5/5] Gaussianity testing...")
    gauss_pvals, frac_ng = test_gaussianity(X_train, y_train)

    # Compute FCS
    metrics = {
        "dim_class_ratio": D / C,
        "covariance_heterogeneity": cov_het,
        "sw_condition_number": eig_results.get("sw_condition_number", 1e6),
        "fraction_non_gaussian": frac_ng,
        "dir_at_95": eig_results.get("dir_at_95", 1.0),
    }
    fcs, prediction = compute_feature_complexity_score(metrics)

    if verbose:
        print(f"\n  ✅ Feature Complexity Score: {fcs:.4f} → {prediction}")
        print(f"     Dim/Class ratio: {D/C:.2f}")
        print(f"     Cov heterogeneity: {cov_het:.4f}")
        print(f"     Sw condition number: {eig_results['sw_condition_number']:.2e}")
        print(f"     Sw effective rank: {eig_results['sw_effective_rank']:.1f}")
        print(f"     Fraction non-Gaussian: {frac_ng:.2%}")
        print(f"     DIR@95%: {eig_results['dir_at_95']:.4f}")
        print(f"     Optimal components (95%): {eig_results['optimal_components_95']}")
        print(f"     Mean Bhattacharyya dist: {mean_bhat:.4f}")
        print(f"     Min Bhattacharyya dist: {min_bhat:.4f}")

    profile = FeatureProfile(
        backbone=backbone,
        dataset=dataset,
        feature_dim=D,
        n_classes=C,
        n_samples=N,
        dim_class_ratio=D / C,
        sw_eigenvalues=eig_results["sw_eigenvalues"],
        sb_eigenvalues=eig_results["sb_eigenvalues"],
        sw_effective_rank=eig_results["sw_effective_rank"],
        sb_effective_rank=eig_results["sb_effective_rank"],
        sw_condition_number=eig_results["sw_condition_number"],
        sw_spectral_decay_rate=eig_results["sw_spectral_decay_rate"],
        discriminant_eigenvalues=eig_results["discriminant_eigenvalues"],
        dir_at_50=eig_results["dir_at_50"],
        dir_at_90=eig_results["dir_at_90"],
        dir_at_95=eig_results["dir_at_95"],
        optimal_components_95=eig_results["optimal_components_95"],
        covariance_heterogeneity=cov_het,
        class_cov_norms=cov_norms,
        bhattacharyya_matrix=bhat_matrix,
        mean_bhattacharyya=mean_bhat,
        min_bhattacharyya=min_bhat,
        confused_pairs=confused_pairs,
        mardia_skewness_pvalues=gauss_pvals,
        fraction_non_gaussian=frac_ng,
        feature_complexity_score=fcs,
        lda_benefit_prediction=prediction,
    )

    return profile
