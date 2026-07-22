"""
Tests for BrentLocalVolModel (Proposition 2.1 reconstruction).

Covers:
- sigma_B positivity at all (t, y)
- Correct weighted-average structure
- Vectorised consistency
- Build from calibrated params
"""

import numpy as np
import pytest

from src.gaussian_mixture import MixtureParams, compute_bucket_vols, apply_monotone_projection
from src.local_vol import BrentLocalVolModel, build_local_vol_model
from src.bachelier import EPSILON


@pytest.fixture
def simple_model():
    """2-component, 3-maturity model for unit tests."""
    N, M = 2, 3
    maturities = np.array([0.25, 0.5, 1.0])
    forwards = np.array([80.0, 80.0, 80.0])
    weights = np.column_stack([
        np.array([0.6, 0.4]),
        np.array([0.55, 0.45]),
        np.array([0.5, 0.5]),
    ])  # shape (N, M)

    cum_vars = np.array([
        [0.5, 1.1, 2.2],  # component 0
        [0.4, 0.9, 1.8],  # component 1
    ])  # (N, M)

    bucket_vars = compute_bucket_vols(cum_vars, maturities)

    return BrentLocalVolModel(
        maturities=maturities,
        forwards=forwards,
        weights=weights,
        cum_vars=cum_vars,
        bucket_vars=bucket_vars,
        N=N,
        M=M,
    )


class TestBrentLocalVolModel:
    def test_sigma_B_positive(self, simple_model):
        """σ_B(t, y) must be positive everywhere."""
        model = simple_model
        rng = np.random.default_rng(0)
        for _ in range(50):
            t = rng.uniform(0.01, 0.9)
            y = rng.uniform(60, 100)
            sigma = model.sigma_B(t, y)
            assert sigma > 0

    def test_sigma_B_at_zero_time_positive(self, simple_model):
        """σ_B at t≈0 must be finite and positive."""
        sigma = simple_model.sigma_B(1e-8, 80.0, mat_idx=0)
        assert np.isfinite(sigma)
        assert sigma > 0

    def test_sigma_B_vectorised_matches_scalar(self, simple_model):
        """Vectorised and scalar sigma_B must agree."""
        t = 0.3
        y_arr = np.array([75.0, 80.0, 85.0])
        mat_idx = 1

        scalar_vals = np.array([simple_model.sigma_B(t, float(y), mat_idx=mat_idx) for y in y_arr])
        vec_vals = simple_model.sigma_B_vectorised(t, y_arr, mat_idx=mat_idx)

        assert np.allclose(scalar_vals, vec_vals, rtol=1e-8)

    def test_sigma_B_extreme_strikes(self, simple_model):
        """Extreme strikes (far OTM) should not cause NaN or negative sigma."""
        model = simple_model
        t = 0.5
        for y in [1.0, 1000.0]:
            sigma = model.sigma_B(t, y, mat_idx=1)
            assert np.isfinite(sigma)
            assert sigma >= EPSILON

    def test_bucket_index_boundary(self, simple_model):
        """Bucket index should be within [0, M-1]."""
        model = simple_model
        for t in [0.0, 0.25, 0.26, 0.5, 0.51, 1.0, 1.5]:
            idx = model._bucket_index(t)
            assert 0 <= idx < model.M

    def test_cum_var_at_T_equals_calibrated(self, simple_model):
        """V_i(T_m) from _cum_var_at_t should equal cum_vars[i, m]."""
        model = simple_model
        for i in range(model.N):
            for m in range(model.M):
                T_m = model.maturities[m]
                v = model._cum_var_at_t(T_m, i)
                expected = model.cum_vars[i, m]
                assert abs(v - expected) < 1e-6, \
                    f"Component {i}, maturity {m}: got {v:.6f}, expected {expected:.6f}"

    def test_cum_var_at_zero_is_zero(self, simple_model):
        """V_i(0) = 0 for all components."""
        for i in range(simple_model.N):
            assert simple_model._cum_var_at_t(0.0, i) == 0.0


class TestBuildLocalVolModel:
    def test_builds_without_error(self):
        """build_local_vol_model assembles correctly from MixtureParams."""
        N, M = 2, 3
        maturities = np.array([0.25, 0.5, 1.0])
        forwards = np.array([80.0, 80.5, 81.0])

        mixture_params = [
            MixtureParams(
                maturity=maturities[m],
                forward=forwards[m],
                weights=np.array([0.5, 0.5]),
                means=np.array([79.0, 81.0]),
                stds=np.array([1.5, 2.0]),
                cum_variances=np.array([1.5**2 * (m+1) * 0.25, 2.0**2 * (m+1) * 0.25]),
                rmse_iv=0.01,
            )
            for m in range(M)
        ]

        cum_vars_raw = np.column_stack([p.cum_variances for p in mixture_params])
        cum_vars_projected = apply_monotone_projection(cum_vars_raw, maturities)
        bucket_vars = compute_bucket_vols(cum_vars_projected, maturities)

        model = build_local_vol_model(mixture_params, bucket_vars, cum_vars_projected)

        assert model.N == N
        assert model.M == M
        assert model.maturities.shape == (M,)
        assert model.weights.shape == (N, M)

    def test_model_sigma_B_is_callable(self):
        """Built model must be usable as sigma_B(t, y)."""
        N, M = 2, 2
        maturities = np.array([0.5, 1.0])
        forwards = np.array([80.0, 80.0])

        mixture_params = [
            MixtureParams(
                maturity=maturities[m],
                forward=forwards[m],
                weights=np.array([0.6, 0.4]),
                means=np.array([79.5, 81.25]),
                stds=np.array([2.0, 2.5]),
                cum_variances=np.array([4.0 * (m+1), 6.25 * (m+1)]),
                rmse_iv=0.005,
            )
            for m in range(M)
        ]

        cum_vars_raw = np.column_stack([p.cum_variances for p in mixture_params])
        cum_vars_projected = apply_monotone_projection(cum_vars_raw, maturities)
        bucket_vars = compute_bucket_vols(cum_vars_projected, maturities)
        model = build_local_vol_model(mixture_params, bucket_vars, cum_vars_projected)

        sigma = model.sigma_B(0.3, 80.0, mat_idx=0)
        assert sigma > 0 and np.isfinite(sigma)
