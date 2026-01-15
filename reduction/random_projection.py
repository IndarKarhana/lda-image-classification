"""
Gaussian Random Projection Module

Projects features using random Gaussian matrices.
Baseline method - no data-dependent learning.
"""

import numpy as np
from sklearn.random_projection import GaussianRandomProjection


class RandomProjectionReducer:
    """
    Gaussian Random Projection dimensionality reduction wrapper.
    
    Projects data using a random matrix drawn from N(0, 1/n_components).
    This is a baseline method that preserves pairwise distances (Johnson-Lindenstrauss).
    """
    
    def __init__(self, n_components, random_state=None):
        """
        Initialize Random Projection reducer.
        
        Args:
            n_components: Target dimensionality
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        self.model = GaussianRandomProjection(
            n_components=n_components,
            random_state=random_state
        )
        self._is_fitted = False
    
    def fit(self, X_train, y_train=None):
        """
        Fit random projection on training data.
        
        Note: y_train is accepted for API consistency but ignored.
        Random projection matrix is independent of data.
        
        Args:
            X_train: Training features, shape (N, D)
            y_train: Ignored (RP is unsupervised)
        
        Returns:
            self
        """
        # y_train is ignored - RP is data-independent
        self.model.fit(X_train)
        self._is_fitted = True
        return self
    
    def transform(self, X):
        """
        Project features using random matrix.
        
        Args:
            X: Features to transform, shape (N, D)
        
        Returns:
            Projected features, shape (N, n_components)
        """
        if not self._is_fitted:
            raise RuntimeError("Random Projection must be fitted before transform")
        
        return self.model.transform(X)
    
    def fit_transform(self, X_train, y_train=None):
        """
        Fit and transform training data.
        
        Args:
            X_train: Training features
            y_train: Ignored (for API consistency)
        
        Returns:
            Projected training features
        """
        self.fit(X_train, y_train)
        return self.transform(X_train)
    
    @property
    def components(self):
        """
        Return the random projection matrix.
        """
        if not self._is_fitted:
            raise RuntimeError("Random Projection must be fitted first")
        return self.model.components_


def reduce_with_rp(X_train, y_train, X_test, n_components, random_state=None):
    """
    Convenience function to fit RP and transform both train and test.
    
    Args:
        X_train: Training features
        y_train: Training labels (ignored, for API consistency)
        X_test: Test features
        n_components: Target dimensionality
        random_state: Random seed
    
    Returns:
        X_train_reduced: Projected training features
        X_test_reduced: Projected test features
        reducer: Fitted RP reducer
    """
    reducer = RandomProjectionReducer(
        n_components=n_components,
        random_state=random_state
    )
    X_train_reduced = reducer.fit_transform(X_train, y_train)
    X_test_reduced = reducer.transform(X_test)
    
    return X_train_reduced, X_test_reduced, reducer


if __name__ == "__main__":
    # Quick test with dummy data
    np.random.seed(42)
    
    n_samples = 1000
    n_features = 512
    
    X = np.random.randn(n_samples, n_features)
    
    # Test Random Projection
    for n_comp in [2, 10, 50, 99, 200]:
        reducer = RandomProjectionReducer(n_components=n_comp, random_state=42)
        X_reduced = reducer.fit_transform(X)
        print(f"RP with {n_comp} components: {X.shape} -> {X_reduced.shape}")
    
    # Test reproducibility with same seed
    reducer1 = RandomProjectionReducer(n_components=50, random_state=42)
    reducer2 = RandomProjectionReducer(n_components=50, random_state=42)
    
    X1 = reducer1.fit_transform(X)
    X2 = reducer2.fit_transform(X)
    
    assert np.allclose(X1, X2), "Same seed should give same projection"
    print("\nReproducibility test passed!")
    
    print("\nRandom Projection module test passed!")
