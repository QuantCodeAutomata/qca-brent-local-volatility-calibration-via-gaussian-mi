"""
Tests for Gaussian-mixture calibration (Exp 1 methodology).

Covers:
- Single-maturity calibration
- Monotone projection correctness
- Bucket vol computation
- Weights and means constraints
"""

import numpy as np
import pytest

from src.gaussian_mixture import (
    MixtureParams,
    calibrate_mixture_maturity,
    apply_monotone_projection,
    compute_bucket_vols,
    _softmax,
    _unpack_params,
)
from src.bachelier import bachelier_mixture_price, bachelier_price


class TestSoftmax:
    def test_sums_to_one(self):
        a = np.array([1.0, 2.0, -1.0])
        w = _softmax(a)
        assert abs(w.sum() - 1.0) < 1e-12

    def test_all_positive(self):
        a = np.array([1.0, 2.0, -1.0])
        w = _softmax(a)
        assert np.all(w > 0)

    def test_uniform_input_equal_weights(self):
        a = np.zeros(5)
        w = _softmax(a)
        assert np.allclose(w, 0.2)


class TestUnpackParams:
    def test_weights_sum_to_one(self):
        N = 3
        F = 80.0
        theta = np.zeros(N - 1 + N - 1 + N)
        weights, means, stds = _unpack_params(theta, N, F)
        assert abs(weights.sum() - 1.0) < 1e-12

    def test_means_martingale_constraint(self):
        """sum lambda_i mu_i = F."""
        N = 3
        F = 80.0
        rng = np.random.default_rng(0)
        theta = rng.standard_normal(N - 1 + N - 1 + N)
        weights, means, stds = _unpack_params(theta, N, F)
        fwd_constraint = float(np.dot(weights, means))
        assert abs(fwd_constraint - F) < 1e-8

    def test_stds_positive(self):
        N = 3
        F = 80.0
        rng = np.random.default_rng(42)
        theta = rng.standard_normal(N - 1 + N - 1 + N)
        weights, means, stds = _unpack_params(theta, N, F)
        assert np.all(stds > 0)


class TestCalibrateMixtureMaturity:
    """Test single-maturity calibration with synthetic data."""

    @pytest.fixture
    def synthetic_quotes(self):
        """Generate synthetic ATM-symmetric quotes from a known mixture."""
        F, T = 80.0, 0.5
        weights = np.array([0.35, 0.40, 0.25])
        means = np.array([78.5, 80.0, 82.14])  # Approximately martingale for these weights
        # Enforce exactly
        means[2] = (F - weights[0]*means[0] - weights[1]*means[1]) / weights[2]
        stds = np.array([2.0, 2.5, 3.0])

        strikes = F + np.linspace(-4, 4, 7)
        ivs = []
        for K in strikes:
            from src.bachelier import bachelier_mixture_impvol
            iv = bachelier_mixture_impvol(F, float(K), T, weights, means, stds, cp=1)
            ivs.append(iv if not np.isnan(iv) else 2.5)

        return {
            "F": F, "T": T,
            "strikes": strikes,
            "ivs": np.array(ivs),
            "cp": np.ones(len(strikes)),
            "weights": weights, "means": means, "stds": stds,
        }

    def test_weights_sum_to_one(self, synthetic_quotes):
        q = synthetic_quotes
        params = calibrate_mixture_maturity(
            F=q["F"], T=q["T"],
            strikes=q["strikes"], market_ivs=q["ivs"], call_flags=q["cp"],
            N=3, n_restarts=3, seed=0,
        )
        assert abs(params.weights.sum() - 1.0) < 1e-6

    def test_weights_in_zero_one(self, synthetic_quotes):
        q = synthetic_quotes
        params = calibrate_mixture_maturity(
            F=q["F"], T=q["T"],
            strikes=q["strikes"], market_ivs=q["ivs"], call_flags=q["cp"],
            N=3, n_restarts=3, seed=0,
        )
        assert np.all(params.weights >= 0)
        assert np.all(params.weights <= 1)

    def test_martingale_constraint(self, synthetic_quotes):
        """sum lambda_i mu_i ≈ F (forward constraint)."""
        q = synthetic_quotes
        params = calibrate_mixture_maturity(
            F=q["F"], T=q["T"],
            strikes=q["strikes"], market_ivs=q["ivs"], call_flags=q["cp"],
            N=3, n_restarts=3, seed=0,
        )
        fwd = float(np.dot(params.weights, params.means))
        assert abs(fwd - q["F"]) < 1e-4

    def test_rmse_reasonable(self, synthetic_quotes):
        """Calibration RMSE should be small for well-specified data."""
        q = synthetic_quotes
        params = calibrate_mixture_maturity(
            F=q["F"], T=q["T"],
            strikes=q["strikes"], market_ivs=q["ivs"], call_flags=q["cp"],
            N=3, n_restarts=5, seed=0,
        )
        assert params.rmse_iv < 1.0  # < 1 dollar normal-vol unit

    def test_stds_positive(self, synthetic_quotes):
        q = synthetic_quotes
        params = calibrate_mixture_maturity(
            F=q["F"], T=q["T"],
            strikes=q["strikes"], market_ivs=q["ivs"], call_flags=q["cp"],
            N=3, n_restarts=3, seed=0,
        )
        assert np.all(params.stds > 0)


class TestMonotoneProjection:
    def test_already_monotone_unchanged(self):
        """Already-increasing sequence stays the same after projection."""
        cum_vars = np.array([[0.1, 0.2, 0.35, 0.5, 0.7]])
        maturities = np.array([0.1, 0.25, 0.5, 0.75, 1.0])
        projected = apply_monotone_projection(cum_vars, maturities)
        assert np.allclose(projected, cum_vars, atol=1e-10)

    def test_decreasing_becomes_nondecreasing(self):
        """Decreasing sequence is made non-decreasing."""
        cum_vars = np.array([[0.5, 0.4, 0.3, 0.2, 0.1]])
        maturities = np.array([0.1, 0.25, 0.5, 0.75, 1.0])
        projected = apply_monotone_projection(cum_vars, maturities)
        diffs = np.diff(projected[0])
        assert np.all(diffs >= -1e-12)

    def test_multiple_components(self):
        """Each component is projected independently."""
        cum_vars = np.array([
            [0.1, 0.05, 0.3],  # non-monotone
            [0.2, 0.4, 0.6],   # already monotone
        ])
        maturities = np.array([0.25, 0.5, 1.0])
        projected = apply_monotone_projection(cum_vars, maturities)
        for i in range(2):
            assert np.all(np.diff(projected[i]) >= -1e-12)


class TestComputeBucketVols:
    def test_first_bucket(self):
        """γ²_{i,1} = V_{T1} / T_1."""
        N, M = 2, 3
        maturities = np.array([0.25, 0.5, 1.0])
        cum_vars = np.array([
            [0.5, 1.0, 2.0],
            [0.4, 0.8, 1.6],
        ])
        bucket_vars = compute_bucket_vols(cum_vars, maturities)
        expected_0_0 = cum_vars[0, 0] / maturities[0]
        assert abs(bucket_vars[0, 0] - expected_0_0) < 1e-10

    def test_subsequent_buckets(self):
        """γ²_{i,m} = (V_{Tm} - V_{T_{m-1}}) / (T_m - T_{m-1})."""
        maturities = np.array([0.25, 0.5, 1.0])
        cum_vars = np.array([[0.5, 1.2, 2.5]])
        bucket_vars = compute_bucket_vols(cum_vars, maturities)
        expected_1 = (cum_vars[0, 1] - cum_vars[0, 0]) / (maturities[1] - maturities[0])
        assert abs(bucket_vars[0, 1] - expected_1) < 1e-10

    def test_nonnegative_for_nondecreasing_input(self):
        """Non-decreasing cumulative variances yield non-negative bucket vars."""
        maturities = np.array([0.1, 0.25, 0.5, 0.75, 1.0])
        cum_vars = np.array([[0.1, 0.25, 0.55, 0.8, 1.1]])
        bucket_vars = compute_bucket_vols(cum_vars, maturities)
        assert np.all(bucket_vars >= -1e-12)
