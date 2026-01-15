"""
Linear Discriminant Analysis (LDA) Module

Fits LDA on training features only.
Maximum components for 100 classes = 99 (C - 1).
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class LDAReducer:
    """
    LDA dimensionality reduction wrapper.
    
    LDA finds linear combinations of features that maximize class separability.
    For C classes, maximum number of components is C - 1 = 99 for CIFAR-100.
    """
    
    def __init__(self, n_components):
        """
        Initialize LDA reducer.
        
        Args:
            n_components: Number of LDA components (1 to 99 for CIFAR-100)
        """
        if n_components > 99:
            raise ValueError(f"LDA can have at most 99 components for 100 classes, got {n_components}")
        
        self.n_components = n_components
        self.model = LinearDiscriminantAnalysis(n_components=n_components)
        self._is_fitted = False
    
    def fit(self, X_train, y_train):
        """
        Fit LDA on training data only.
        
        Args:
            X_train: Training features, shape (N, D)
            y_train: Training labels, shape (N,)
        
        Returns:
            self
        """
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        return self
    
    def transform(self, X):
        """
        Project features to LDA subspace.
        
        Args:
            X: Features to transform, shape (N, D)
        
        Returns:
            Projected features, shape (N, n_components)
        """
        if not self._is_fitted:
            raise RuntimeError("LDA must be fitted before transform")
        
        return self.model.transform(X)
    
    def fit_transform(self, X_train, y_train):
        """
        Fit LDA and transform training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
        
        Returns:
            Projected training features
        """
        self.fit(X_train, y_train)
        return self.transform(X_train)
    
    @property
    def explained_variance_ratio(self):
        """
        Return the ratio of between-class variance explained by each component.
        """
        if not self._is_fitted:
            raise RuntimeError("LDA must be fitted first")
        return self.model.explained_variance_ratio_


def reduce_with_lda(X_train, y_train, X_test, n_components):
    """
    Convenience function to fit LDA and transform both train and test.
    
    Args:
        X_train: Training features
        y_train: Training labels (used for fitting)
        X_test: Test features
        n_components: Number of LDA components
    
    Returns:
        X_train_reduced: Projected training features
        X_test_reduced: Projected test features
        reducer: Fitted LDA reducer
    """
    reducer = LDAReducer(n_components=n_components)
    X_train_reduced = reducer.fit_transform(X_train, y_train)
    X_test_reduced = reducer.transform(X_test)
    
    return X_train_reduced, X_test_reduced, reducer


if __name__ == "__main__":
    # Quick test with dummy data
    np.random.seed(42)
    
    # Simulate 100 classes
    n_samples = 1000
    n_features = 512
    n_classes = 100
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    # Test LDA
    for n_comp in [2, 10, 50, 99]:
        reducer = LDAReducer(n_components=n_comp)
        X_reduced = reducer.fit_transform(X, y)
        print(f"LDA with {n_comp} components: {X.shape} -> {X_reduced.shape}")
    
    print("\nLDA module test passed!")
