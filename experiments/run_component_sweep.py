"""
Component Sweep Experiment
============================
Tests LDA and PCA at different numbers of components to validate
the claim that d=C-1 is the optimal choice.

Tests: d = [5, 10, 20, 40, 60, 80, 99] on CIFAR-100
Backbones: ResNet-18 (512D), ResNet-50 (2048D)
Seeds: 3 per config
"""

import os
import sys
import time
import csv
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

COMPONENT_VALUES = [5, 10, 20, 40, 60, 80, 99]
SEEDS = [42, 123, 456]
BACKBONES = ['resnet18', 'resnet50']
METHODS = ['LDA', 'PCA']

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'component_sweep')
FEATURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'features', 'saved')


def load_features(backbone: str):
    """Load cached CIFAR-100 features."""
    path = os.path.join(FEATURES_DIR, f'{backbone}_cifar100.npz')
    data = np.load(path)
    return data['X_train'], data['y_train'], data['X_test'], data['y_test']


def evaluate(X_train, y_train, X_test, y_test, method, n_components, seed):
    """Reduce + classify, return accuracy and time."""
    t0 = time.perf_counter()

    if method == 'LDA':
        reducer = LinearDiscriminantAnalysis(n_components=n_components)
        X_tr = reducer.fit_transform(X_train, y_train)
        X_te = reducer.transform(X_test)
    else:  # PCA
        reducer = PCA(n_components=n_components, random_state=seed)
        X_tr = reducer.fit_transform(X_train)
        X_te = reducer.transform(X_test)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    clf = LogisticRegression(solver='lbfgs', max_iter=5000, C=1.0,
                              n_jobs=-1, random_state=seed)
    clf.fit(X_tr, y_train)
    acc = clf.score(X_te, y_test) * 100
    elapsed = time.perf_counter() - t0

    return acc, elapsed


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, 'component_sweep.csv')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['backbone', 'method', 'n_components', 'seed', 'accuracy', 'time'])

    total = len(BACKBONES) * len(METHODS) * len(COMPONENT_VALUES) * len(SEEDS)
    done = 0

    for backbone in BACKBONES:
        print(f"\n{'='*60}")
        print(f"Loading {backbone} features...")
        X_train, y_train, X_test, y_test = load_features(backbone)
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

        # Also run Full baseline
        for seed in SEEDS:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train)
            X_te = scaler.transform(X_test)
            t0 = time.perf_counter()
            clf = LogisticRegression(solver='lbfgs', max_iter=5000, C=1.0,
                                      n_jobs=-1, random_state=seed)
            clf.fit(X_tr, y_train)
            acc = clf.score(X_te, y_test) * 100
            elapsed = time.perf_counter() - t0
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([backbone, 'Full', X_train.shape[1], seed, f'{acc:.2f}', f'{elapsed:.2f}'])
            print(f"  Full (d={X_train.shape[1]}), seed={seed}: {acc:.2f}% ({elapsed:.1f}s)")

        for method in METHODS:
            for n_comp in COMPONENT_VALUES:
                for seed in SEEDS:
                    acc, elapsed = evaluate(X_train, y_train, X_test, y_test,
                                            method, n_comp, seed)
                    done += 1
                    with open(csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([backbone, method, n_comp, seed, f'{acc:.2f}', f'{elapsed:.2f}'])
                    print(f"  [{done}/{total}] {backbone}/{method} d={n_comp} seed={seed}: {acc:.2f}% ({elapsed:.1f}s)")

    print(f"\nResults saved to {csv_path}")
    print("Done!")


if __name__ == '__main__':
    main()
