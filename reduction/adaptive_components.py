"""
Adaptive Component Selection (ACS) — DG-LDA Component 2
=========================================================
Instead of always using C-1 components, analyzes the discriminant eigenvalue
spectrum to select the optimal number of components.

Key idea: Higher-dimensional backbones have faster eigenvalue decay in the
discriminant space, so fewer effective components are needed.

Author: Research Study
"""

import numpy as np
from typing import Tuple, Optional


def discriminant_information_ratio(eigenvalues: np.ndarray) -> np.ndarray:
    """
    Compute cumulative Discriminant Information Ratio (DIR).

    DIR(d) = sum(lambda_1..lambda_d) / sum(all lambda)

    Similar to PCA's explained variance ratio, but for discriminant eigenvalues.

    Args:
        eigenvalues: Discriminant eigenvalues sorted descending, shape (C-1,)

    Returns:
        Cumulative DIR, shape (C-1,)
    """
    eigenvalues = np.maximum(eigenvalues, 0)
    total = eigenvalues.sum()
    if total == 0:
        return np.ones_like(eigenvalues)
    return np.cumsum(eigenvalues) / total


def select_components_by_dir(
    eigenvalues: np.ndarray,
    threshold: float = 0.95,
    min_components: int = 2,
) -> Tuple[int, float]:
    """
    Select number of components using DIR threshold.

    Finds the smallest d such that DIR(d) >= threshold.

    Args:
        eigenvalues: Discriminant eigenvalues, sorted descending
        threshold: DIR threshold (e.g. 0.95 for 95% discriminant info)
        min_components: Minimum number of components to return

    Returns:
        n_components: Optimal number of components
        actual_dir: The actual DIR at the selected d
    """
    dir_curve = discriminant_information_ratio(eigenvalues)
    idx = np.searchsorted(dir_curve, threshold)
    n_components = max(int(idx + 1), min_components)
    n_components = min(n_components, len(eigenvalues))
    actual_dir = float(dir_curve[n_components - 1])
    return n_components, actual_dir


def select_components_by_gap(
    eigenvalues: np.ndarray,
    gap_threshold: float = 0.1,
    min_components: int = 2,
) -> Tuple[int, float]:
    """
    Select components by finding the largest eigenvalue gap.

    Looks for the steepest drop in the eigenvalue spectrum — the "elbow".
    Components before the gap carry most discriminant information.

    Args:
        eigenvalues: Discriminant eigenvalues, sorted descending
        gap_threshold: Minimum relative gap to consider significant
        min_components: Minimum components

    Returns:
        n_components: Optimal number of components
        gap_ratio: The relative gap at the selected point
    """
    eigenvalues = np.maximum(eigenvalues, 0)
    if len(eigenvalues) < 2 or eigenvalues[0] == 0:
        return min_components, 0.0

    # Compute relative gaps: (lambda_i - lambda_{i+1}) / lambda_i
    gaps = np.diff(eigenvalues) / (eigenvalues[:-1] + 1e-30)
    gaps = np.abs(gaps)  # Ensure positive

    # Find the first gap exceeding threshold, starting from index 1
    # (skip the very first gap which is often large)
    significant_gaps = np.where(gaps[1:] > gap_threshold)[0] + 1

    if len(significant_gaps) > 0:
        # Take the first significant gap
        cut_idx = significant_gaps[0]
        n_components = max(int(cut_idx + 1), min_components)
    else:
        # No significant gap — use all components
        n_components = len(eigenvalues)

    n_components = min(n_components, len(eigenvalues))
    gap_ratio = float(gaps[n_components - 1]) if n_components - 1 < len(gaps) else 0.0

    return n_components, gap_ratio


def select_components_adaptive(
    eigenvalues: np.ndarray,
    n_classes: int,
    feature_dim: int,
    dir_threshold: float = 0.95,
    min_components: int = 2,
) -> Tuple[int, str]:
    """
    Adaptive component selection that chooses the best strategy based
    on feature space characteristics.

    Strategy selection:
    - If dim/classes > 20: Use stricter DIR (0.90) — many eigenvalues are noise
    - If dim/classes > 5:  Use standard DIR (0.95)
    - If dim/classes <= 5: Use relaxed DIR (0.99) — most eigenvalues carry info

    The DIR approach is more reliable than gap analysis for determining
    how many discriminant directions carry useful information.

    Safety floor: never select fewer than max(min_components, C//5) components
    to avoid catastrophic under-selection.

    Args:
        eigenvalues: Discriminant eigenvalues, sorted descending
        n_classes: Number of classes
        feature_dim: Original feature dimensionality
        dir_threshold: Base DIR threshold
        min_components: Minimum components

    Returns:
        n_components: Selected number of components
        strategy: Description of which strategy was used
    """
    ratio = feature_dim / n_classes

    # Safety floor: at least 20% of C-1, minimum 10 for 100+ class problems
    floor = max(min_components, n_classes // 5, 10 if n_classes >= 50 else 2)

    if ratio > 20:
        # Very high-dimensional (e.g. ResNet-50 @ 2048D, 100 classes)
        # Many eigenvalues are noise — stricter threshold
        n_comp, actual_dir = select_components_by_dir(
            eigenvalues, threshold=0.90, min_components=floor
        )
        strategy = f"strict_dir_0.90 (dim/class={ratio:.1f})"
    elif ratio > 5:
        # Medium-high dimensional (e.g. ResNet-18 @ 512D, EfficientNet @ 1280D)
        # Standard threshold captures key discriminant directions
        n_comp, actual_dir = select_components_by_dir(
            eigenvalues, threshold=dir_threshold, min_components=floor
        )
        strategy = f"standard_dir_{dir_threshold} (dim/class={ratio:.1f})"
    else:
        # Low-dimensional: most eigenvalues carry info, use relaxed threshold
        n_comp, actual_dir = select_components_by_dir(
            eigenvalues, threshold=0.99, min_components=floor
        )
        strategy = f"relaxed_dir_0.99 (dim/class={ratio:.1f})"

    # Never exceed C-1
    n_comp = min(n_comp, n_classes - 1)

    return n_comp, strategy
