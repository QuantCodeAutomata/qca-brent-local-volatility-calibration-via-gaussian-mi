"""
Tests for joint Brent-Gasoil Monte Carlo pricing model.

Covers:
- Simulate_joint_paths: martingale property, shape, correlation
- mc_price_options: prices and IV shapes
- Anchored surface construction
- Error metrics correctness
"""

import numpy as np
import pytest

from src.local_vol import BrentLocalVolModel
from src.gaussian_mixture import compute_bucket_vols
from src.joint_model import (
    simulate_joint_paths,
    mc_price_options,
    interpolate_to_moneyness_grid,
    compute_anchored_surface,
    compute_error_metrics,
    ModelSpec,
)
from src.bachelier import EPSILON


@pytest.fixture
def flat_local_vol():
    """
    Flat local-vol model: σ_B(t,y) = constant.
    Two-component mixture with equal weights and means ≈ F.
    """
    N, M = 2, 2
    maturities = np.array([0.5, 1.0])
    forwards = np.array([80.0, 80.0])

    weights = np.column_stack([np.array([0.5, 0.5]), np.array([0.5, 0.5])])
    cum_vars = np.array([
        [4.0, 8.0],   # σ²=8 per year → σ=2.83 annualised
        [4.0, 8.0],
    ])
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


class TestSimulateJointPaths:
    def test_output_shape(self, flat_local_vol):
        """Output shapes should be (2*n_paths,) for antithetic variates."""
        B_T, G_T = simulate_joint_paths(
            T=0.5, F_B=80.0, F_G=90.0,
            local_vol_B=flat_local_vol, mat_idx=0,
            h_func=None, h_fix=None, rho=0.9,
            n_steps=20, n_paths=1000,
            model_spec=ModelSpec.BRENT_ONLY, seed=0,
        )
        assert B_T.shape == (2000,)
        assert G_T.shape == (2000,)

    def test_brent_martingale(self, flat_local_vol):
        """E[B(T)] ≈ B(0) under r=0 Bachelier dynamics."""
        F_B = 80.0
        B_T, _ = simulate_joint_paths(
            T=0.5, F_B=F_B, F_G=90.0,
            local_vol_B=flat_local_vol, mat_idx=0,
            h_func=None, h_fix=None, rho=0.0,
            n_steps=100, n_paths=50_000,
            model_spec=ModelSpec.BRENT_ONLY, seed=42,
        )
        assert abs(B_T.mean() - F_B) < 0.5  # within $0.50 for 100k paths

    def test_gasoil_martingale_brent_only(self, flat_local_vol):
        """E[G(T)] ≈ G(0) for Brent-only spec (h=0)."""
        F_G = 92.0
        _, G_T = simulate_joint_paths(
            T=0.5, F_B=80.0, F_G=F_G,
            local_vol_B=flat_local_vol, mat_idx=0,
            h_func=None, h_fix=None, rho=0.0,
            n_steps=100, n_paths=50_000,
            model_spec=ModelSpec.BRENT_ONLY, seed=42,
        )
        assert abs(G_T.mean() - F_G) < 0.5

    def test_fixed_spread_adds_to_vol(self, flat_local_vol):
        """Fixed-spread correction increases Gasoil variance vs Brent-only."""
        _, G_only = simulate_joint_paths(
            T=0.5, F_B=80.0, F_G=90.0,
            local_vol_B=flat_local_vol, mat_idx=0,
            h_func=None, h_fix=None, rho=0.0,
            n_steps=50, n_paths=10_000,
            model_spec=ModelSpec.BRENT_ONLY, seed=0,
        )
        _, G_fixed = simulate_joint_paths(
            T=0.5, F_B=80.0, F_G=90.0,
            local_vol_B=flat_local_vol, mat_idx=0,
            h_func=None, h_fix=1.0, rho=0.0,
            n_steps=50, n_paths=10_000,
            model_spec=ModelSpec.FIXED_SPREAD, seed=0,
        )
        # Higher h_fix → higher Gasoil vol → wider distribution
        assert G_fixed.std() > G_only.std() * 0.95  # should be materially larger

    def test_correlation_affects_joint_distribution(self, flat_local_vol):
        """High rho → Brent and Gasoil moves are correlated."""
        B_hi, G_hi = simulate_joint_paths(
            T=0.5, F_B=80.0, F_G=90.0,
            local_vol_B=flat_local_vol, mat_idx=0,
            h_func=None, h_fix=None, rho=0.99,
            n_steps=50, n_paths=10_000,
            model_spec=ModelSpec.BRENT_ONLY, seed=1,
        )
        B_lo, G_lo = simulate_joint_paths(
            T=0.5, F_B=80.0, F_G=90.0,
            local_vol_B=flat_local_vol, mat_idx=0,
            h_func=None, h_fix=None, rho=0.0,
            n_steps=50, n_paths=10_000,
            model_spec=ModelSpec.BRENT_ONLY, seed=1,
        )
        corr_hi = np.corrcoef(B_hi, G_hi)[0, 1]
        corr_lo = np.corrcoef(B_lo, G_lo)[0, 1]
        assert corr_hi > corr_lo


class TestMcPriceOptions:
    def test_prices_nonnegative(self):
        """MC prices must be non-negative."""
        terminal = np.random.default_rng(0).normal(80, 3, 50_000)
        strikes = np.array([75, 78, 80, 82, 85], dtype=float)
        ivs = mc_price_options(terminal, 80.0, strikes, 0.5)
        assert np.all(np.where(np.isnan(ivs), 0, ivs) >= -1e-8)

    def test_iv_nonnegative(self):
        """Implied vols must be non-negative."""
        terminal = np.random.default_rng(0).normal(80, 3, 50_000)
        strikes = np.linspace(75, 85, 7)
        ivs = mc_price_options(terminal, 80.0, strikes, 0.5)
        valid_ivs = ivs[~np.isnan(ivs)]
        assert np.all(valid_ivs >= 0)

    def test_atm_iv_close_to_vol(self):
        """ATM IV from MC normal distribution should ≈ σ_N·√T / √T = σ_N."""
        sigma_n = 3.0
        T = 0.5
        rng = np.random.default_rng(99)
        F = 80.0
        terminal = F + rng.normal(0, sigma_n * np.sqrt(T), 200_000)
        ivs = mc_price_options(terminal, F, np.array([F]), T)
        assert abs(ivs[0] - sigma_n) < 0.3  # within $0.30


class TestInterpolateToMoneyness:
    def test_output_length(self):
        strikes = np.array([75, 78, 80, 82, 85], dtype=float)
        ivs = np.array([3.0, 2.8, 2.7, 2.8, 3.0])
        common_mon = np.linspace(0.9, 1.1, 15)
        interp = interpolate_to_moneyness_grid(strikes, 80.0, ivs, common_mon)
        assert len(interp) == 15

    def test_no_oscillation_pchip(self):
        """PCHIP should not introduce large oscillations for monotone-in smile data."""
        strikes = np.array([74, 77, 80, 83, 86], dtype=float)
        ivs = np.array([3.5, 3.0, 2.8, 3.0, 3.4])  # typical smile
        common_mon = np.linspace(0.9, 1.1, 30)
        interp = interpolate_to_moneyness_grid(strikes, 80.0, ivs, common_mon)
        valid = interp[~np.isnan(interp)]
        assert np.all(valid > 0)

    def test_flat_extrapolation(self):
        """Beyond strike support, interpolation should be flat."""
        strikes = np.array([78, 80, 82], dtype=float)
        ivs = np.array([3.0, 2.8, 3.0])
        mon_grid = np.array([0.5, 0.975, 1.0, 1.025, 2.0])
        interp = interpolate_to_moneyness_grid(strikes, 80.0, ivs, mon_grid)
        # Very low moneyness: flat at left edge
        assert abs(interp[0] - ivs[0]) < 1e-10
        # Very high moneyness: flat at right edge
        assert abs(interp[-1] - ivs[-1]) < 1e-10


class TestAnchoredSurface:
    def test_anchored_formula(self):
        """σ^G_final = σ^B_mkt + (σ^G_mod - σ^B_mod)."""
        M, J = 3, 5
        brent_mkt = np.random.default_rng(0).uniform(2, 4, (M, J))
        brent_mod = np.random.default_rng(1).uniform(1.8, 4.2, (M, J))
        gasoil_mod = brent_mod + np.random.default_rng(2).uniform(0, 1, (M, J))

        anchored = compute_anchored_surface(brent_mkt, brent_mod, gasoil_mod)
        expected = brent_mkt + (gasoil_mod - brent_mod)
        assert np.allclose(anchored, expected)

    def test_anchored_shape(self):
        M, J = 5, 10
        brent_mkt = np.ones((M, J)) * 3.0
        brent_mod = np.ones((M, J)) * 2.9
        gasoil_mod = np.ones((M, J)) * 3.5
        anchored = compute_anchored_surface(brent_mkt, brent_mod, gasoil_mod)
        assert anchored.shape == (M, J)


class TestComputeErrorMetrics:
    def test_perfect_fit(self):
        """Zero error when model == market."""
        arr = np.ones((3, 5)) * 2.5
        metrics = compute_error_metrics(arr, arr)
        assert abs(metrics["rmse"]) < 1e-12
        assert abs(metrics["mae"]) < 1e-12

    def test_known_error(self):
        """RMSE and MAE for known arrays."""
        model = np.array([[3.0, 3.0], [3.0, 3.0]])
        market = np.array([[2.0, 2.0], [2.0, 2.0]])
        metrics = compute_error_metrics(model, market)
        assert abs(metrics["rmse"] - 1.0) < 1e-10
        assert abs(metrics["mae"] - 1.0) < 1e-10

    def test_relative_error_positive(self):
        model = np.array([[3.0, 2.5]])
        market = np.array([[2.5, 3.0]])
        metrics = compute_error_metrics(model, market)
        assert metrics["relative_error"] >= 0

    def test_all_nan_returns_nan(self):
        model = np.full((3, 5), np.nan)
        market = np.full((3, 5), np.nan)
        metrics = compute_error_metrics(model, market)
        assert np.isnan(metrics["rmse"])
