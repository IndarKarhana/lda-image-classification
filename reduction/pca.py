"""
Principal Component Analysis (PCA) Module

Fits PCA on training features only (unsupervised baseline).
No label information is used during fitting.
"""

import numpy as np
from sklearn.decomposition import PCA


class PCAReducer:
    """
    PCA dimensionality reduction wrapper.
    
    PCA finds orthogonal directions of maximum variance.
    Unlike LDA, PCA is unsupervised (ignores class labels).
    """
    
    def __init__(self, n_components):
        """
        Initialize PCA reducer.
        
        Args:
            n_components: Number of principal components
        """
        self.n_components = n_components
        self.model = PCA(n_components=n_components)
        self._is_fitted = False
    
    def fit(self, X_train, y_train=None):
        """
        Fit PCA on training data only.
        
        Note: y_train is accepted for API consistency but ignored.
        
        Args:
            X_train: Training features, shape (N, D)
            y_train: Ignored (PCA is unsupervised)
        
        Returns:
            self
        """
        # y_train is ignored - PCA is unsupervised
        self.model.fit(X_train)
        self._is_fitted = True
        return self
    
    def transform(self, X):
        """
        Project features to PCA subspace.
        
        Args:
            X: Features to transform, shape (N, D)
        
        Returns:
            Projected features, shape (N, n_components)
        """
        if not self._is_fitted:
            raise RuntimeError("PCA must be fitted before transform")
        
        return self.model.transform(X)
    
    def fit_transform(self, X_train, y_train=None):
        """
        Fit PCA and transform training data.
        
        Args:
            X_train: Training features
            y_train: Ignored (for API consistency)
        
        Returns:
            Projected training features
        """
        self.fit(X_train, y_train)
        return self.transform(X_train)
    
    @property
    def explained_variance_ratio(self):
        """
        Return the ratio of variance explained by each component.
        """
        if not self._is_fitted:
            raise RuntimeError("PCA must be fitted first")
        return self.model.explained_variance_ratio_
    
    @property
    def cumulative_variance_ratio(self):
        """
        Return cumulative variance explained.
        """
        return np.cumsum(self.explained_variance_ratio)


def reduce_with_pca(X_train, y_train, X_test, n_components):
    """
    Convenience function to fit PCA and transform both train and test.
    
    Args:
        X_train: Training features
        y_train: Training labels (ignored, for API consistency)
        X_test: Test features
        n_components: Number of PCA components
    
    Returns:
        X_train_reduced: Projected training features
        X_test_reduced: Projected test features
        reducer: Fitted PCA reducer
    """
    reducer = PCAReducer(n_components=n_components)
    X_train_reduced = reducer.fit_transform(X_train, y_train)
    X_test_reduced = reducer.transform(X_test)
    
    return X_train_reduced, X_test_reduced, reducer


if __name__ == "__main__":
    # Quick test with dummy data
    np.random.seed(42)
    
    n_samples = 1000
    n_features = 512
    
    X = np.random.randn(n_samples, n_features)
    
    # Test PCA
    for n_comp in [2, 10, 50, 99, 200]:
        reducer = PCAReducer(n_components=n_comp)
        X_reduced = reducer.fit_transform(X)
        print(f"PCA with {n_comp} components: {X.shape} -> {X_reduced.shape}")
        print(f"  Variance explained: {reducer.cumulative_variance_ratio[-1]:.4f}")
    
    print("\nPCA module test passed!")
