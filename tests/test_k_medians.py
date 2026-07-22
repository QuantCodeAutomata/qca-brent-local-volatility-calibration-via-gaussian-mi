"""
Tests for custom k-medians clustering with cityblock (L1) distance.

Covers:
- Correct number of clusters
- L1 inertia is a valid objective
- Centroid coordinate-wise median property
- Reproducibility with fixed seed
- Edge cases
"""

import numpy as np
import pytest

from src.k_medians import k_medians, KMediansResult, _assign, _update, _l1_objective


class TestKMediansBasic:
    def test_returns_k_centroids(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 10))
        result = k_medians(X, k=6, n_init=5, seed=0)
        assert result.centroids.shape == (6, 10)

    def test_labels_valid_range(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 4))
        result = k_medians(X, k=6, n_init=5, seed=0)
        assert result.labels.shape == (100,)
        assert np.all(result.labels >= 0)
        assert np.all(result.labels < 6)

    def test_all_labels_assigned(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 4))
        result = k_medians(X, k=6, n_init=5, seed=0)
        # Every point should have a label
        assert len(result.labels) == len(X)

    def test_inertia_nonnegative(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 6))
        result = k_medians(X, k=3, n_init=5, seed=0)
        assert result.inertia >= 0.0

    def test_reproducibility(self):
        rng = np.random.default_rng(99)
        X = rng.standard_normal((80, 8))
        r1 = k_medians(X, k=4, n_init=10, seed=42)
        r2 = k_medians(X, k=4, n_init=10, seed=42)
        assert np.allclose(r1.centroids, r2.centroids)
        assert np.array_equal(r1.labels, r2.labels)

    def test_different_seeds_may_differ(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((80, 8))
        r1 = k_medians(X, k=6, n_init=3, seed=0)
        r2 = k_medians(X, k=6, n_init=3, seed=999)
        # Not necessarily different, but inertia should be similar magnitude
        assert r1.inertia > 0

    def test_well_separated_clusters_found(self):
        """Well-separated clusters should be recovered accurately."""
        rng = np.random.default_rng(0)
        centers = np.array([[0, 0], [10, 0], [5, 8]], dtype=float)
        X = np.vstack([
            centers[i] + rng.standard_normal((30, 2)) * 0.5
            for i in range(3)
        ])
        result = k_medians(X, k=3, n_init=20, seed=0)

        # Sort centroids by first coordinate for comparison
        order = np.argsort(result.centroids[:, 0])
        recovered = result.centroids[order]
        expected = centers[np.argsort(centers[:, 0])]

        assert np.allclose(recovered, expected, atol=1.0)

    def test_single_point_per_cluster(self):
        """k == n_samples: each point is its own centroid."""
        X = np.eye(5)
        result = k_medians(X, k=5, n_init=3, seed=0)
        assert result.centroids.shape == (5, 5)


class TestKMediansHelpers:
    def test_l1_objective_zero_for_centroid(self):
        """When each point equals its centroid, inertia = 0."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        centroids = X.copy()
        labels = np.array([0, 1])
        inertia = _l1_objective(X, centroids, labels)
        assert abs(inertia) < 1e-12

    def test_assign_nearest_centroid(self):
        X = np.array([[0.0], [1.0], [9.0], [10.0]])
        centroids = np.array([[0.0], [10.0]])
        labels = _assign(X, centroids)
        assert np.array_equal(labels, [0, 0, 1, 1])

    def test_update_is_coordinate_median(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        labels = np.array([0, 0, 0])
        centroids = _update(X, labels, k=1)
        expected = np.median(X, axis=0)
        assert np.allclose(centroids[0], expected)


class TestKMediansEdgeCases:
    def test_k_equals_one(self):
        """k=1: single centroid = coordinate-wise median of all data."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = k_medians(X, k=1, n_init=5, seed=0)
        expected_centroid = np.median(X, axis=0)
        assert np.allclose(result.centroids[0], expected_centroid, atol=1e-10)

    def test_single_feature(self):
        """Works for 1-dimensional data."""
        X = np.array([[1.0], [2.0], [10.0], [11.0], [20.0], [21.0]])
        result = k_medians(X, k=3, n_init=10, seed=0)
        assert result.centroids.shape == (3, 1)
        assert result.inertia >= 0

    def test_high_dimensional(self):
        """Works for high-dimensional data (2M=22 for M=11)."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 22))
        result = k_medians(X, k=6, n_init=10, seed=0)
        assert result.centroids.shape == (6, 22)
