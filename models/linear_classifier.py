"""
Linear Classifier Module

Logistic Regression classifier for evaluation.
Same classifier used across all dimensionality reduction methods.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class LinearClassifier:
    """
    Logistic Regression classifier wrapper.
    
    Uses the same hyperparameters across all experiments for fair comparison.
    Does NOT tune hyperparameters on test data.
    """
    
    def __init__(self, random_state=None, max_iter=1000):
        """
        Initialize classifier.
        
        Args:
            random_state: Random seed for reproducibility
            max_iter: Maximum iterations for solver convergence
        """
        self.random_state = random_state
        self.max_iter = max_iter
        
        # Fixed hyperparameters for all experiments
        # Note: multi_class parameter removed in sklearn 1.7+
        # lbfgs solver handles multiclass natively
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            solver='lbfgs',           # Good for multiclass
            n_jobs=-1                  # Use all cores
        )
        self._is_fitted = False
    
    def fit(self, X_train, y_train):
        """
        Train classifier on training data.
        
        Args:
            X_train: Training features, shape (N, D)
            y_train: Training labels, shape (N,)
        
        Returns:
            self
        """
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X: Features to classify, shape (N, D)
        
        Returns:
            Predicted labels, shape (N,)
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted before predict")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Features to classify, shape (N, D)
        
        Returns:
            Class probabilities, shape (N, n_classes)
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted before predict_proba")
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        """
        Compute accuracy on given data.
        
        Args:
            X: Features
            y: True labels
        
        Returns:
            Accuracy score
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted before score")
        return self.model.score(X, y)


def train_and_evaluate(X_train, y_train, X_test, y_test, random_state=None):
    """
    Train classifier and evaluate on test set.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        random_state: Random seed
    
    Returns:
        accuracy: Test accuracy
        classifier: Trained classifier
    """
    classifier = LinearClassifier(random_state=random_state)
    classifier.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy, classifier


if __name__ == "__main__":
    # Quick test with dummy data
    np.random.seed(42)
    
    n_train = 1000
    n_test = 200
    n_features = 50
    n_classes = 100
    
    X_train = np.random.randn(n_train, n_features)
    y_train = np.random.randint(0, n_classes, n_train)
    X_test = np.random.randn(n_test, n_features)
    y_test = np.random.randint(0, n_classes, n_test)
    
    accuracy, clf = train_and_evaluate(X_train, y_train, X_test, y_test, random_state=42)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Verify reproducibility
    acc1, _ = train_and_evaluate(X_train, y_train, X_test, y_test, random_state=42)
    acc2, _ = train_and_evaluate(X_train, y_train, X_test, y_test, random_state=42)
    assert acc1 == acc2, "Same seed should give same accuracy"
    print("Reproducibility test passed!")
    
    print("\nLinear classifier module test passed!")
