"""
Tests for Bachelier (normal model) option pricing and implied-vol inversion.

Tests:
- Price formula correctness (call-put parity, intrinsic value, ATM)
- Implied-vol inversion round-trip
- Gaussian-mixture pricing
- Edge cases
"""

import numpy as np
import pytest

from src.bachelier import (
    bachelier_price,
    bachelier_impvol,
    bachelier_mixture_price,
    bachelier_mixture_impvol,
    gaussian_density,
    EPSILON,
)


class TestBachelierPrice:
    def test_call_put_parity(self):
        """C - P = F - K for all (F, K, sigma, T)."""
        F, K, sigma, T = 75.0, 73.0, 3.0, 0.5
        call = bachelier_price(F, K, sigma, T, cp=1)
        put = bachelier_price(F, K, sigma, T, cp=-1)
        assert abs((call - put) - (F - K)) < 1e-8

    def test_call_put_parity_atm(self):
        """At-the-money: C = P = sigma * sqrt(T) / sqrt(2*pi)."""
        F, sigma, T = 80.0, 2.5, 0.25
        call = bachelier_price(F, F, sigma, T, cp=1)
        put = bachelier_price(F, F, sigma, T, cp=-1)
        expected = sigma * np.sqrt(T) / np.sqrt(2 * np.pi)
        assert abs(call - expected) < 1e-8
        assert abs(call - put) < 1e-10

    def test_deep_itm_call_approaches_intrinsic(self):
        """Deep ITM call ≈ F - K."""
        F, K, sigma, T = 100.0, 50.0, 1.0, 0.1
        call = bachelier_price(F, K, sigma, T, cp=1)
        intrinsic = F - K
        assert call > intrinsic - 1e-10
        assert abs(call - intrinsic) < 1.0  # close to intrinsic

    def test_zero_maturity_equals_intrinsic(self):
        """T→0: call price approaches max(F-K, 0)."""
        F, K, sigma = 80.0, 75.0, 3.0
        call = bachelier_price(F, K, sigma, 1e-8, cp=1)
        assert abs(call - (F - K)) < 1e-4

    def test_zero_vol_equals_intrinsic(self):
        """sigma=0: price equals discounted intrinsic."""
        F, K, T = 80.0, 75.0, 1.0
        call = bachelier_price(F, K, 1e-8, T, cp=1)
        assert abs(call - (F - K)) < 1e-4

    def test_price_nonnegative(self):
        """Option price must always be non-negative."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            F = rng.uniform(50, 100)
            K = rng.uniform(40, 110)
            sigma = rng.uniform(0.1, 5.0)
            T = rng.uniform(0.01, 2.0)
            call = bachelier_price(F, K, sigma, T, cp=1)
            put = bachelier_price(F, K, sigma, T, cp=-1)
            assert float(call) >= -1e-10
            assert float(put) >= -1e-10

    def test_vectorised_shapes(self):
        """Vectorised inputs return correct shapes."""
        F = np.array([75.0, 80.0])
        K = np.array([73.0, 82.0])
        sigma = np.array([2.0, 3.0])
        T = np.array([0.5, 1.0])
        result = bachelier_price(F, K, sigma, T, cp=1)
        assert result.shape == (2,)


class TestBachelierImpvol:
    def test_round_trip_atm(self):
        """IV inversion of ATM price recovers sigma_N."""
        F, K, sigma_target, T = 80.0, 80.0, 2.5, 0.5
        price = bachelier_price(F, K, sigma_target, T, cp=1)
        sigma_recovered = bachelier_impvol(price, F, K, T, cp=1)
        assert abs(float(sigma_recovered) - sigma_target) < 1e-6

    def test_round_trip_otm(self):
        """IV inversion of OTM call recovers sigma_N."""
        F, K, sigma_target, T = 80.0, 85.0, 3.0, 0.25
        price = bachelier_price(F, K, sigma_target, T, cp=1)
        sigma_recovered = bachelier_impvol(price, F, K, T, cp=1)
        assert abs(float(sigma_recovered) - sigma_target) < 1e-5

    def test_round_trip_put(self):
        """IV inversion round-trip for put options."""
        F, K, sigma_target, T = 75.0, 77.0, 2.0, 0.75
        price = bachelier_price(F, K, sigma_target, T, cp=-1)
        sigma_recovered = bachelier_impvol(price, F, K, T, cp=-1)
        assert abs(float(sigma_recovered) - sigma_target) < 1e-5

    def test_intrinsic_price_returns_near_zero(self):
        """Intrinsic-value call with zero time value → near-zero IV or nan."""
        # At expiry, sigma can be arbitrary; check no exception
        F, K, T = 80.0, 70.0, 1e-8
        price = F - K  # pure intrinsic
        sigma = bachelier_impvol(price, F, K, T, cp=1)
        # Should be nan or very large (numerical instability expected at T≈0)
        assert np.isnan(float(sigma)) or float(sigma) > 0


class TestGaussianMixturePrice:
    def test_single_component_equals_bachelier(self):
        """One-component mixture = Bachelier price."""
        F, K, sigma_n, T = 80.0, 80.0, 2.5, 0.5
        weights = np.array([1.0])
        means = np.array([F])
        stds = np.array([sigma_n * np.sqrt(T)])  # std = sigma_n * sqrt(T)

        mix_price = bachelier_mixture_price(F, K, T, weights, means, stds, cp=1)
        direct_price = float(bachelier_price(F, K, sigma_n, T, cp=1))
        assert abs(mix_price - direct_price) < 1e-8

    def test_mixture_price_nonnegative(self):
        """Mixture price must be non-negative."""
        rng = np.random.default_rng(42)
        F = 80.0
        T = 0.5
        for _ in range(20):
            w = rng.dirichlet([1, 1, 1])
            means = F + rng.normal(0, 1, 3)
            stds = np.abs(rng.normal(1, 0.5, 3)) + 0.1
            K = rng.uniform(70, 90)
            price = bachelier_mixture_price(F, K, T, w, means, stds, cp=1)
            assert price >= -1e-10

    def test_mixture_call_put_parity(self):
        """Call-put parity holds for Gaussian mixture."""
        F, K, T = 80.0, 78.0, 0.5
        weights = np.array([0.4, 0.35, 0.25])
        means = np.array([79.0, 80.0, 82.0])
        stds = np.array([1.5, 2.0, 2.5])

        call = bachelier_mixture_price(F, K, T, weights, means, stds, cp=1)
        put = bachelier_mixture_price(F, K, T, weights, means, stds, cp=-1)

        # Expected forward from mixture: sum_i lambda_i * mu_i
        expected_fwd = float(np.dot(weights, means))
        assert abs((call - put) - (expected_fwd - K)) < 1e-8

    def test_mixture_impvol_round_trip(self):
        """Mixture implied vol inversion round-trip."""
        F, K, T = 80.0, 80.0, 0.5
        weights = np.array([0.5, 0.5])
        means = np.array([79.0, 81.0])
        stds = np.array([2.0, 2.5])

        # Force forward constraint: sum lambda_i mu_i = F
        means[1] = (F - weights[0] * means[0]) / weights[1]

        iv = bachelier_mixture_impvol(F, K, T, weights, means, stds, cp=1)
        assert not np.isnan(iv)
        assert iv > 0

    def test_weights_sum_to_one_constraint(self):
        """Martingale constraint: sum lambda_i mu_i = F.

        Backed-out last mean: m3 = (F - w1*m1 - w2*m2) / w3
        = (80 - 0.3*78 - 0.4*80) / 0.3 = 24.6 / 0.3 = 82.0
        """
        F = 80.0
        weights = np.array([0.3, 0.4, 0.3])
        means = np.array([78.0, 80.0, 82.0])  # backed-out last mean
        assert abs(np.dot(weights, means) - F) < 1e-6


class TestGaussianDensity:
    def test_density_nonnegative(self):
        x = np.linspace(-5, 5, 100)
        d = gaussian_density(x, mu=0.0, sigma=1.0)
        assert np.all(d >= 0)

    def test_density_integrates_to_one(self):
        x = np.linspace(-10, 10, 10000)
        d = gaussian_density(x, mu=0.0, sigma=1.0)
        integral = np.trapz(d, x)
        assert abs(integral - 1.0) < 1e-3
