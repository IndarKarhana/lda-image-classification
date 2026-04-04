"""
Backbone-Adaptive Regularized LDA (BAR-LDA) — DG-LDA Component 3
===================================================================
Applies Ledoit-Wolf shrinkage regularization to Sw before solving the
generalized eigenvalue problem. The shrinkage intensity is adapted based
on the feature space geometry (condition number, dim/sample ratio).

Key insight: Higher-dimensional backbones (ResNet-50 @ 2048D) have
ill-conditioned Sw matrices. Standard LDA fails silently by using SVD
solver, but regularized Sw yields better discriminant directions.

Author: Research Study
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.covariance import ledoit_wolf


class RegularizedLDA:
    """
    LDA with backbone-adaptive Ledoit-Wolf shrinkage regularization.

    Instead of sklearn's LDA (which handles singularity via SVD truncation),
    this explicitly regularizes Sw using Ledoit-Wolf shrinkage:

        Sw_reg = (1 - alpha) * Sw + alpha * (trace(Sw)/p) * I

    where alpha is either automatically estimated (Ledoit-Wolf) or set
    based on the feature space condition number.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        shrinkage: str = "auto",
        shrinkage_alpha: Optional[float] = None,
    ):
        """
        Args:
            n_components: Number of discriminant components. None = C-1.
            shrinkage: Strategy — "auto" (Ledoit-Wolf), "manual", or "condition_based"
            shrinkage_alpha: Manual shrinkage parameter in [0, 1]. Only used if shrinkage="manual".
        """
        self.n_components = n_components
        self.shrinkage = shrinkage
        self.shrinkage_alpha = shrinkage_alpha

        self._projection = None
        self._mean = None
        self._is_fitted = False
        self.alpha_used_ = None
        self.eigenvalues_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegularizedLDA":
        """
        Fit regularized LDA.

        1. Compute Sw and Sb
        2. Apply shrinkage regularization to Sw
        3. Solve generalized eigenvalue problem: Sb @ w = lambda * Sw_reg @ w
        4. Select top n_components eigenvectors

        Args:
            X: Training features (N, D)
            y: Training labels (N,)
        """
        classes = np.unique(y)
        C = len(classes)
        N, D = X.shape

        if self.n_components is None:
            self.n_components = C - 1

        self._mean = X.mean(axis=0)

        # Compute scatter matrices (normalized by N for proper output scale)
        # Without normalization, Sw eigenvalues ~ N * variance, causing
        # Sw^{-1/2} to produce tiny projections (std ~ 0.005 instead of ~ 1.0)
        Sw = np.zeros((D, D), dtype=np.float64)
        Sb = np.zeros((D, D), dtype=np.float64)
        global_mean = X.mean(axis=0)

        for c in classes:
            Xc = X[y == c]
            nc = Xc.shape[0]
            mc = Xc.mean(axis=0)
            diff = Xc - mc
            Sw += diff.T @ diff
            dm = (mc - global_mean).reshape(-1, 1)
            Sb += nc * (dm @ dm.T)

        # Normalize to covariance scale (consistent with sklearn LDA)
        Sw /= N
        Sb /= N

        # Determine shrinkage alpha
        if self.shrinkage == "auto":
            # Ledoit-Wolf on per-sample within-class centered data
            X_centered = np.zeros_like(X)
            for c in classes:
                mask = y == c
                X_centered[mask] = X[mask] - X[mask].mean(axis=0)

            _, alpha = ledoit_wolf(X_centered)
            self.alpha_used_ = alpha

        elif self.shrinkage == "condition_based":
            # Base alpha on condition number of Sw
            sw_eigs = np.linalg.eigvalsh(Sw)
            sw_eigs_pos = sw_eigs[sw_eigs > 1e-12]
            if len(sw_eigs_pos) > 1:
                cond = sw_eigs_pos[-1] / sw_eigs_pos[0]  # Note: eigvalsh is ascending
                # Map condition number to alpha: higher cond → more shrinkage
                log_cond = np.log10(max(cond, 1.0))
                alpha = min(0.5, max(0.01, log_cond / 20.0))
            else:
                alpha = 0.5
            self.alpha_used_ = alpha

        elif self.shrinkage == "manual":
            alpha = self.shrinkage_alpha if self.shrinkage_alpha is not None else 0.1
            self.alpha_used_ = alpha

        else:
            raise ValueError(f"Unknown shrinkage strategy: {self.shrinkage}")

        # Apply shrinkage
        target = np.trace(Sw) / D * np.eye(D)
        Sw_reg = (1 - self.alpha_used_) * Sw + self.alpha_used_ * target

        # Solve via whitening approach (more numerically stable than eigh(Sb, Sw))
        # 1. Eigendecompose Sw_reg
        eigvals_sw, eigvecs_sw = np.linalg.eigh(Sw_reg)
        # 2. Clip small eigenvalues for stability
        eigvals_sw = np.maximum(eigvals_sw, 1e-10)
        # 3. Compute Sw_reg^{-1/2}
        inv_sqrt = 1.0 / np.sqrt(eigvals_sw)
        Sw_inv_sqrt = eigvecs_sw * inv_sqrt[np.newaxis, :]  # (D, D) broadcast
        # 4. Whiten Sb: Sb_w = Sw^{-1/2} @ Sb @ Sw^{-1/2}
        Sb_white = Sw_inv_sqrt.T @ Sb @ Sw_inv_sqrt
        # 5. Standard symmetric eigenvalue problem on whitened Sb
        eigenvalues, eigvecs_white = np.linalg.eigh(Sb_white)
        # 6. Back-transform: w = Sw^{-1/2} @ w_white
        eigenvectors = Sw_inv_sqrt @ eigvecs_white

        # eigh returns ascending order, reverse for descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select top n_components
        self._projection = eigenvectors[:, : self.n_components]
        self.eigenvalues_ = eigenvalues[: self.n_components]
        self._is_fitted = True

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project features to discriminant subspace."""
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before transform()")
        return (X - self._mean) @ self._projection

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform training data."""
        self.fit(X, y)
        return self.transform(X)


def reduce_with_regularized_lda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_components: Optional[int] = None,
    shrinkage: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, RegularizedLDA]:
    """
    Convenience function for regularized LDA reduction.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        n_components: Number of components (None = C-1)
        shrinkage: "auto", "condition_based", or "manual"

    Returns:
        X_train_reduced, X_test_reduced, fitted reducer
    """
    reducer = RegularizedLDA(n_components=n_components, shrinkage=shrinkage)
    X_train_reduced = reducer.fit_transform(X_train, y_train)
    X_test_reduced = reducer.transform(X_test)
    return X_train_reduced, X_test_reduced, reducer
