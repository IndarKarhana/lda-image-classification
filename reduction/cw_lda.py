"""
Confusion-Weighted LDA (CW-LDA) — DG-LDA Component 4 (NOVEL)
===============================================================
Modifies the between-class scatter matrix to weight class pairs by their
confusion level (inverse Bhattacharyya distance).

Standard LDA:
    Sb = sum_i  n_i * (mu_i - mu) @ (mu_i - mu)^T

This treats all class separations equally. But for hard classification tasks
with many classes (100+), some class pairs are already well-separated and
don't need further discriminant attention. Meanwhile, confused pairs
(close in feature space) are underweighted.

CW-LDA:
    Sb_cw = sum_{i<j}  w_ij * (mu_i - mu_j) @ (mu_i - mu_j)^T

where w_ij = 1 / (bhattacharyya_distance(i, j) + epsilon)

This focuses the discriminant projection on separating confused class pairs.

Author: Research Study
"""

import numpy as np
from typing import Tuple, Optional, Dict
from sklearn.covariance import ledoit_wolf


class ConfusionWeightedLDA:
    """
    LDA with confusion-weighted between-class scatter.

    Key novelty: The between-class scatter matrix weights class pairs
    inversely proportional to their Bhattacharyya distance, focusing
    discriminant power on the most confused class pairs.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        weighting: str = "bhattacharyya",
        temperature: float = 1.0,
        shrinkage: str = "auto",
    ):
        """
        Args:
            n_components: Number of discriminant components. None = C-1.
            weighting: How to compute class-pair weights:
                - "bhattacharyya": Inverse Bhattacharyya distance
                - "uniform": Standard LDA (all pairs weighted equally, for ablation)
                - "softmax": Softmax of negative distances (soft attention)
            temperature: Temperature for softmax weighting. Lower = more focused.
            shrinkage: Sw regularization — "auto" (Ledoit-Wolf) or "none"
        """
        self.n_components = n_components
        self.weighting = weighting
        self.temperature = temperature
        self.shrinkage = shrinkage

        self._projection = None
        self._mean = None
        self._is_fitted = False
        self.eigenvalues_ = None
        self.weight_matrix_ = None
        self.alpha_used_ = None

    def _compute_pairwise_bhattacharyya(
        self, X: np.ndarray, y: np.ndarray, classes: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise Bhattacharyya distances using diagonal covariance.

        Returns:
            dist_matrix: (C, C) symmetric matrix of distances
        """
        C = len(classes)
        means = {}
        vars_ = {}

        for i, c in enumerate(classes):
            Xc = X[y == c]
            means[i] = Xc.mean(axis=0)
            vars_[i] = Xc.var(axis=0) + 1e-10

        dist_matrix = np.zeros((C, C))
        for i in range(C):
            for j in range(i + 1, C):
                avg_var = 0.5 * (vars_[i] + vars_[j])
                diff = means[i] - means[j]
                term1 = 0.125 * np.sum(diff**2 / avg_var)
                term2 = 0.5 * np.sum(
                    np.log(avg_var) - 0.5 * np.log(vars_[i]) - 0.5 * np.log(vars_[j])
                )
                dist_matrix[i, j] = term1 + term2
                dist_matrix[j, i] = dist_matrix[i, j]

        return dist_matrix

    def _compute_weights(self, dist_matrix: np.ndarray) -> np.ndarray:
        """
        Convert distance matrix to weight matrix.

        Args:
            dist_matrix: (C, C) pairwise distances

        Returns:
            weight_matrix: (C, C) pairwise weights (higher = more attention)
        """
        C = dist_matrix.shape[0]

        if self.weighting == "uniform":
            return np.ones((C, C))

        elif self.weighting == "bhattacharyya":
            # Inverse distance: closer pairs get higher weight
            epsilon = 1e-6
            weights = 1.0 / (dist_matrix + epsilon)
            np.fill_diagonal(weights, 0)
            # Normalize so weights sum to C*(C-1)/2
            total = weights[np.triu_indices(C, k=1)].sum()
            if total > 0:
                target_sum = C * (C - 1) / 2
                weights *= target_sum / total
            return weights

        elif self.weighting == "softmax":
            # Softmax attention on negative distances
            upper_tri = dist_matrix[np.triu_indices(C, k=1)]
            neg_dist = -upper_tri / self.temperature
            neg_dist -= neg_dist.max()  # Stability
            exp_vals = np.exp(neg_dist)
            softmax_weights = exp_vals / exp_vals.sum() * len(upper_tri)

            weights = np.zeros((C, C))
            idx = np.triu_indices(C, k=1)
            weights[idx] = softmax_weights
            weights += weights.T
            return weights

        else:
            raise ValueError(f"Unknown weighting: {self.weighting}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConfusionWeightedLDA":
        """
        Fit CW-LDA.

        1. Compute class means and Sw
        2. Compute Bhattacharyya distance matrix
        3. Convert to confusion weights
        4. Build weighted Sb: sum_{i<j} w_ij * (mu_i - mu_j)(mu_i - mu_j)^T
        5. Regularize Sw (Ledoit-Wolf)
        6. Solve generalized eigenvalue problem
        """
        classes = np.unique(y)
        C = len(classes)
        N, D = X.shape

        if self.n_components is None:
            self.n_components = C - 1

        self._mean = X.mean(axis=0)

        # Compute class means and Sw (normalized by N for numerical stability)
        class_means = {}
        Sw = np.zeros((D, D), dtype=np.float64)
        for c_idx, c in enumerate(classes):
            Xc = X[y == c]
            mc = Xc.mean(axis=0)
            class_means[c_idx] = mc
            diff = Xc - mc
            Sw += diff.T @ diff

        # Normalize to covariance scale (consistent with sklearn LDA)
        Sw /= N

        # Compute pairwise distances
        dist_matrix = self._compute_pairwise_bhattacharyya(X, y, classes)

        # Compute confusion weights
        weight_matrix = self._compute_weights(dist_matrix)
        self.weight_matrix_ = weight_matrix

        # Build confusion-weighted between-class scatter
        Sb_cw = np.zeros((D, D), dtype=np.float64)
        for i in range(C):
            for j in range(i + 1, C):
                w = weight_matrix[i, j]
                dm = (class_means[i] - class_means[j]).reshape(-1, 1)
                Sb_cw += w * (dm @ dm.T)

        # Normalize to covariance scale
        Sb_cw /= N

        # Regularize Sw
        if self.shrinkage == "auto":
            X_centered = np.zeros_like(X)
            for c_idx, c in enumerate(classes):
                mask = y == c
                X_centered[mask] = X[mask] - X[mask].mean(axis=0)
            _, alpha = ledoit_wolf(X_centered)
            self.alpha_used_ = alpha
        else:
            alpha = 0.0
            self.alpha_used_ = 0.0

        target = np.trace(Sw) / D * np.eye(D)
        Sw_reg = (1 - alpha) * Sw + alpha * target

        # Add small regularization for numerical stability
        Sw_reg += 1e-6 * np.eye(D)

        # Solve via whitening approach (more numerically stable than eigh(Sb, Sw))
        # 1. Eigendecompose Sw_reg
        eigvals_sw, eigvecs_sw = np.linalg.eigh(Sw_reg)
        # 2. Clip small eigenvalues for stability
        eigvals_sw = np.maximum(eigvals_sw, 1e-10)
        # 3. Compute Sw_reg^{-1/2}
        inv_sqrt = 1.0 / np.sqrt(eigvals_sw)
        Sw_inv_sqrt = eigvecs_sw * inv_sqrt[np.newaxis, :]  # (D, D) broadcast
        # 4. Whiten Sb: Sb_w = Sw^{-1/2} @ Sb_cw @ Sw^{-1/2}
        Sb_white = Sw_inv_sqrt.T @ Sb_cw @ Sw_inv_sqrt
        # 5. Standard symmetric eigenvalue problem on whitened Sb
        eigenvalues, eigvecs_white = np.linalg.eigh(Sb_white)
        # 6. Back-transform: w = Sw^{-1/2} @ w_white
        eigenvectors = Sw_inv_sqrt @ eigvecs_white

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self._projection = eigenvectors[:, : self.n_components]
        self.eigenvalues_ = eigenvalues[: self.n_components]
        self._is_fitted = True

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project features to CW-LDA discriminant subspace."""
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before transform()")
        return (X - self._mean) @ self._projection

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform training data."""
        self.fit(X, y)
        return self.transform(X)


def reduce_with_cw_lda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_components: Optional[int] = None,
    weighting: str = "bhattacharyya",
    temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, ConfusionWeightedLDA]:
    """
    Convenience function for CW-LDA reduction.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        n_components: Number of components (None = C-1)
        weighting: "bhattacharyya", "uniform", or "softmax"
        temperature: Temperature for softmax weighting

    Returns:
        X_train_reduced, X_test_reduced, fitted reducer
    """
    reducer = ConfusionWeightedLDA(
        n_components=n_components,
        weighting=weighting,
        temperature=temperature,
    )
    X_train_reduced = reducer.fit_transform(X_train, y_train)
    X_test_reduced = reducer.transform(X_test)
    return X_train_reduced, X_test_reduced, reducer
