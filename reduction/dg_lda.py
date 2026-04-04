"""
Distribution-Guided LDA (DG-LDA) — Orchestrator
==================================================
Combines all DG-LDA components into a unified reducer that:

  1. Profiles the feature space (Feature Space Profiler)
  2. Selects optimal components (Adaptive Component Selection)
  3. Applies backbone-adaptive regularization (Regularized LDA)
  4. Uses confusion-weighted scatter (CW-LDA)

This is the main class users should instantiate.

Author: Research Study
"""

import numpy as np
import time
from typing import Tuple, Optional, Dict
from dataclasses import dataclass, field

from reduction.feature_profiler import (
    profile_feature_space,
    compute_scatter_matrices,
    FeatureProfile,
)
from reduction.adaptive_components import (
    select_components_adaptive,
    select_components_by_dir,
    discriminant_information_ratio,
)
from reduction.regularized_lda import RegularizedLDA
from reduction.cw_lda import ConfusionWeightedLDA


@dataclass
class DGLDAResult:
    """Container for DG-LDA fitting results with full diagnostics."""
    profile: FeatureProfile
    selected_components: int
    selection_strategy: str
    shrinkage_alpha: float
    weighting_mode: str
    eigenvalues: np.ndarray = field(repr=False)
    fit_time: float = 0.0
    transform_time: float = 0.0


class DGLDA:
    """
    Distribution-Guided LDA (DG-LDA).

    A self-configuring LDA variant that analyzes feature space geometry
    to automatically:
      - Select the optimal number of components
      - Apply appropriate regularization
      - Weight class-pair separations by confusion level

    Usage:
        dglda = DGLDA(mode="full")
        X_train_red = dglda.fit_transform(X_train, y_train, backbone="resnet50", dataset="cifar100")
        X_test_red = dglda.transform(X_test)
        print(dglda.result_)  # Full diagnostics
    """

    def __init__(
        self,
        mode: str = "full",
        n_components: Optional[int] = None,
        shrinkage: str = "auto",
        weighting: str = "bhattacharyya",
        temperature: float = 1.0,
        dir_threshold: float = 0.95,
        verbose: bool = True,
    ):
        """
        Args:
            mode: DG-LDA operating mode:
                - "full": Profile → Adaptive components → Regularized CW-LDA (default)
                - "regularized": Fixed components + Regularized LDA (no CW, no profiling)
                - "cw_only": Fixed components + CW-LDA (no regularization adaptation)
                - "ablation_vanilla": Standard LDA (for fair comparison baseline)
            n_components: Override component count. None = auto-select in "full" mode.
            shrinkage: "auto" (Ledoit-Wolf), "condition_based", "manual"
            weighting: "bhattacharyya", "softmax", "uniform"
            temperature: Softmax temperature (only for weighting="softmax")
            dir_threshold: DIR threshold for component selection
            verbose: Print progress
        """
        self.mode = mode
        self.n_components = n_components
        self.shrinkage = shrinkage
        self.weighting = weighting
        self.temperature = temperature
        self.dir_threshold = dir_threshold
        self.verbose = verbose

        self._reducer = None
        self._is_fitted = False
        self.result_: Optional[DGLDAResult] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        backbone: str = "unknown",
        dataset: str = "unknown",
    ) -> "DGLDA":
        """
        Fit DG-LDA to training data.

        The full pipeline:
          1. Feature Space Profiling (if mode="full")
          2. Adaptive Component Selection (if n_components is None)
          3. Confusion-Weighted Regularized LDA fitting

        Args:
            X: Training features (N, D)
            y: Training labels (N,)
            backbone: Backbone name (for profiling metadata)
            dataset: Dataset name (for profiling metadata)
        """
        t_start = time.perf_counter()
        classes = np.unique(y)
        C = len(classes)
        N, D = X.shape

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"DG-LDA [{self.mode}]: Fitting on {backbone}/{dataset}")
            print(f"  N={N}, D={D}, C={C}")
            print(f"{'='*60}")

        # ─── Step 1: Feature Space Profiling ───
        if self.mode == "full":
            profile = profile_feature_space(
                X, y, backbone=backbone, dataset=dataset, verbose=self.verbose
            )
        else:
            # Minimal profiling for metadata
            profile = profile_feature_space(
                X, y, backbone=backbone, dataset=dataset, verbose=False
            )

        # ─── Step 2: Component Selection ───
        if self.n_components is not None:
            selected_n = min(self.n_components, C - 1)
            strategy = f"manual_override ({self.n_components})"
        elif self.mode in ("full",):
            # Use adaptive selection based on discriminant eigenvalues
            disc_eigs = profile.discriminant_eigenvalues
            selected_n, strategy = select_components_adaptive(
                disc_eigs,
                n_classes=C,
                feature_dim=D,
                dir_threshold=self.dir_threshold,
            )
            if self.verbose:
                print(f"\n  Adaptive component selection: {selected_n} (strategy: {strategy})")
        else:
            selected_n = C - 1
            strategy = f"default_max ({C - 1})"

        # ─── Step 3: Fit the appropriate reducer ───
        if self.mode == "ablation_vanilla":
            # Standard sklearn LDA for fair comparison
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            reducer = LinearDiscriminantAnalysis(n_components=selected_n)
            reducer.fit(X, y)
            self._reducer = reducer
            alpha_used = 0.0
            eigenvalues = getattr(reducer, "explained_variance_ratio_", np.array([]))
            weighting_used = "none"

        elif self.mode == "regularized":
            # Regularized LDA without confusion weighting
            reducer = RegularizedLDA(
                n_components=selected_n,
                shrinkage=self.shrinkage,
            )
            reducer.fit(X, y)
            self._reducer = reducer
            alpha_used = reducer.alpha_used_
            eigenvalues = reducer.eigenvalues_
            weighting_used = "none"

        elif self.mode in ("cw_only", "full"):
            # Confusion-Weighted LDA (with or without adaptive components)
            reducer = ConfusionWeightedLDA(
                n_components=selected_n,
                weighting=self.weighting,
                temperature=self.temperature,
                shrinkage=self.shrinkage if self.mode == "full" else "none",
            )
            reducer.fit(X, y)
            self._reducer = reducer
            alpha_used = reducer.alpha_used_ or 0.0
            eigenvalues = reducer.eigenvalues_
            weighting_used = self.weighting

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        t_fit = time.perf_counter() - t_start

        self.result_ = DGLDAResult(
            profile=profile,
            selected_components=selected_n,
            selection_strategy=strategy,
            shrinkage_alpha=alpha_used,
            weighting_mode=weighting_used,
            eigenvalues=eigenvalues if eigenvalues is not None else np.array([]),
            fit_time=t_fit,
        )

        self._is_fitted = True

        if self.verbose:
            print(f"\n  DG-LDA fit complete:")
            print(f"    Components: {selected_n}")
            print(f"    Shrinkage α: {alpha_used:.4f}")
            print(f"    Weighting: {weighting_used}")
            print(f"    Fit time: {t_fit:.2f}s")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features to DG-LDA subspace."""
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before transform()")
        t_start = time.perf_counter()
        result = self._reducer.transform(X)
        if self.result_ is not None:
            self.result_.transform_time = time.perf_counter() - t_start
        return result

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        backbone: str = "unknown",
        dataset: str = "unknown",
    ) -> np.ndarray:
        """Fit and transform training data."""
        self.fit(X, y, backbone=backbone, dataset=dataset)
        return self.transform(X)


def reduce_with_dglda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    mode: str = "full",
    n_components: Optional[int] = None,
    backbone: str = "unknown",
    dataset: str = "unknown",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, DGLDA]:
    """
    Convenience function for DG-LDA reduction.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        mode: "full", "regularized", "cw_only", "ablation_vanilla"
        n_components: Override component count (None = auto)
        backbone: Backbone name
        dataset: Dataset name
        verbose: Print progress

    Returns:
        X_train_reduced, X_test_reduced, fitted DGLDA
    """
    reducer = DGLDA(mode=mode, n_components=n_components, verbose=verbose)
    X_train_reduced = reducer.fit_transform(
        X_train, y_train, backbone=backbone, dataset=dataset
    )
    X_test_reduced = reducer.transform(X_test)
    return X_train_reduced, X_test_reduced, reducer
