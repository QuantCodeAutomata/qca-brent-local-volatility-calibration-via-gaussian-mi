"""
Custom k-medians clustering with cityblock (L1) distance.

The paper requires k-medians with cityblock distance, NOT k-means (Euclidean).
Standard scikit-learn KMeans uses Euclidean means → incorrect.
scikit-learn-extra KMedoids uses medoids (observed data points) → not medians.
Therefore a custom implementation is required.

Custom implementation — Context7 found no library equivalent for k-medians
with cityblock distance (paper specification).

Algorithm:
    1. Initialise k centroids from random data points (k-means++ style optional)
    2. Assignment: each point → nearest centroid in L1 distance
    3. Update: each centroid → coordinate-wise median of assigned points
    4. Repeat until stable or max_iter reached
    5. Run n_init restarts; keep solution with lowest total L1 objective
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist


@dataclass
class KMediansResult:
    """Output of k-medians clustering."""
    centroids: np.ndarray   # shape (k, d)
    labels: np.ndarray      # shape (n,)  cluster assignment
    inertia: float          # total L1 objective value
    n_iter: int             # iterations in best run
    seed: int


def _l1_objective(X: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> float:
    """Total L1 (cityblock) distance from each point to its assigned centroid."""
    total = 0.0
    for j in range(len(centroids)):
        mask = labels == j
        if mask.any():
            total += np.abs(X[mask] - centroids[j]).sum()
    return total


def _assign(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each point to nearest centroid in L1 distance."""
    # cdist with cityblock: shape (n, k)
    dist = cdist(X, centroids, metric="cityblock")
    return np.argmin(dist, axis=1)


def _update(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Update centroids as coordinate-wise median of assigned points."""
    d = X.shape[1]
    centroids = np.empty((k, d))
    for j in range(k):
        mask = labels == j
        if mask.any():
            centroids[j] = np.median(X[mask], axis=0)
        else:
            # Empty cluster: reinitialise to a random point
            centroids[j] = X[np.random.randint(len(X))]
    return centroids


def _single_run(
    X: np.ndarray,
    k: int,
    max_iter: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """
    Single k-medians run with random initialisation.

    Returns
    -------
    (centroids, labels, inertia, n_iter)
    """
    n = len(X)
    # Initialise by sampling k distinct points
    init_idx = rng.choice(n, size=k, replace=False)
    centroids = X[init_idx].copy()

    labels = np.full(n, -1, dtype=int)
    for it in range(max_iter):
        new_labels = _assign(X, centroids)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        centroids = _update(X, labels, k)
    else:
        labels = _assign(X, centroids)

    inertia = _l1_objective(X, centroids, labels)
    return centroids, labels, inertia, it + 1


def k_medians(
    X: np.ndarray,
    k: int = 6,
    n_init: int = 50,
    max_iter: int = 300,
    tol: float = 1e-6,
    seed: int = 42,
) -> KMediansResult:
    """
    k-medians clustering with cityblock (L1) distance.

    Parameters
    ----------
    X        : data matrix, shape (n_samples, n_features)
    k        : number of clusters (paper specifies k=6)
    n_init   : number of random restarts (50-100 recommended for stability)
    max_iter : maximum iterations per restart
    tol      : convergence tolerance (unused in discrete assignment; kept for API compat)
    seed     : base random seed for reproducibility

    Returns
    -------
    KMediansResult with centroids, labels, inertia, n_iter, seed
    """
    X = np.asarray(X, dtype=float)
    assert X.ndim == 2, "X must be 2-D"
    assert k <= len(X), "k must not exceed number of samples"

    rng = np.random.default_rng(seed)

    best_centroids = None
    best_labels = None
    best_inertia = np.inf
    best_n_iter = 0

    for _ in range(n_init):
        centroids, labels, inertia, n_iter = _single_run(X, k, max_iter, rng)
        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids.copy()
            best_labels = labels.copy()
            best_n_iter = n_iter

    return KMediansResult(
        centroids=best_centroids,
        labels=best_labels,
        inertia=best_inertia,
        n_iter=best_n_iter,
        seed=seed,
    )
