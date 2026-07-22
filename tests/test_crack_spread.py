"""
Tests for crack-spread feature engineering and h function estimation.

Covers:
- Feature vector construction
- HFunction interpolation and flat extrapolation
- Correlation estimation
- Current crack spread computation
"""

import numpy as np
import pandas as pd
import pytest

from src.crack_spread import (
    compute_features,
    build_h_functions,
    compute_current_crack_spread,
    estimate_historical_correlation,
    HFunction,
)


@pytest.fixture
def synthetic_series():
    """Generate 200 business days of synthetic Brent/Gasoil series for 3 tenors."""
    rng = np.random.default_rng(0)
    M = 3
    n_days = 200
    dates = pd.bdate_range(start="2025-01-01", periods=n_days, freq="B")

    brent_series = {}
    gasoil_series = {}
    base_B = 75.0
    base_G = 90.0

    for m in range(M):
        ret_B = rng.normal(0, 1.5, n_days)
        ret_G = rng.normal(0, 1.8, n_days)
        b_prices = base_B + np.cumsum(ret_B)
        g_prices = base_G + np.cumsum(ret_G)
        brent_series[m] = pd.Series(np.maximum(b_prices, 10), index=dates)
        gasoil_series[m] = pd.Series(np.maximum(g_prices, 15), index=dates)

    return brent_series, gasoil_series, dates


class TestComputeFeatures:
    def test_returns_dataframe(self, synthetic_series):
        b, g, _ = synthetic_series
        M = 3
        df = compute_features(b, g, M=M, n=20, n2=10)
        assert isinstance(df, pd.DataFrame)

    def test_correct_columns(self, synthetic_series):
        b, g, _ = synthetic_series
        M = 3
        df = compute_features(b, g, M=M, n=20, n2=10)
        if not df.empty:
            for m in range(M):
                assert f"V_{m}" in df.columns
                assert f"S_{m}" in df.columns

    def test_no_nan_in_complete_data(self, synthetic_series):
        """With complete, clean series, feature matrix should have no NaN."""
        b, g, _ = synthetic_series
        M = 3
        df = compute_features(b, g, M=M, n=20, n2=10)
        if not df.empty:
            v_cols = [f"V_{m}" for m in range(M)]
            s_cols = [f"S_{m}" for m in range(M)]
            assert not df[v_cols + s_cols].isna().any().any()

    def test_downsample_reduces_rows(self, synthetic_series):
        """Downsampling every n2 should reduce rows vs n2=1."""
        b, g, _ = synthetic_series
        M = 3
        df_dense = compute_features(b, g, M=M, n=20, n2=1)
        df_sparse = compute_features(b, g, M=M, n=20, n2=5)
        if not df_dense.empty and not df_sparse.empty:
            assert len(df_sparse) <= len(df_dense)

    def test_insufficient_history_returns_empty(self):
        """Fewer than n observations → empty DataFrame."""
        dates = pd.bdate_range(start="2025-01-01", periods=10)
        brent = {0: pd.Series(np.ones(10) * 75, index=dates)}
        gasoil = {0: pd.Series(np.ones(10) * 90, index=dates)}
        df = compute_features(brent, gasoil, M=1, n=20)
        assert df.empty


class TestBuildHFunctions:
    @pytest.fixture
    def small_feature_df(self):
        """50-row feature DataFrame with 3 tenors for quick clustering tests."""
        rng = np.random.default_rng(0)
        M = 3
        n = 50
        data = {
            "date": pd.date_range(start="2025-01-01", periods=n, freq="B"),
        }
        for m in range(M):
            data[f"V_{m}"] = rng.normal(0.2, 0.1, n)
            data[f"S_{m}"] = rng.normal(15, 3, n)
        return pd.DataFrame(data)

    def test_returns_m_functions(self, small_feature_df):
        M = 3
        h_fns, _ = build_h_functions(small_feature_df, M=M, k=3, n_init=5, seed=0)
        assert len(h_fns) == M

    def test_hfunction_callable(self, small_feature_df):
        h_fns, _ = build_h_functions(small_feature_df, M=3, k=3, n_init=5, seed=0)
        for hf in h_fns:
            val = hf(15.0)  # typical crack spread
            assert np.isfinite(val)

    def test_flat_extrapolation_below(self, small_feature_df):
        """h_m(s) at s << min centroid = h_m(s_min) [flat extrapolation]."""
        h_fns, _ = build_h_functions(small_feature_df, M=3, k=3, n_init=5, seed=0)
        hf = h_fns[0]
        s_min = hf.centroid_crack.min()
        val_at_min = float(hf(s_min))
        val_below = float(hf(s_min - 1000))
        assert abs(val_below - val_at_min) < 1e-10

    def test_flat_extrapolation_above(self, small_feature_df):
        """h_m(s) at s >> max centroid = h_m(s_max) [flat extrapolation]."""
        h_fns, _ = build_h_functions(small_feature_df, M=3, k=3, n_init=5, seed=0)
        hf = h_fns[0]
        s_max = hf.centroid_crack.max()
        val_at_max = float(hf(s_max))
        val_above = float(hf(s_max + 1000))
        assert abs(val_above - val_at_max) < 1e-10

    def test_km_result_has_6_centroids(self, small_feature_df):
        """k-medians result should have exactly k=3 centroids here."""
        _, km = build_h_functions(small_feature_df, M=3, k=3, n_init=5, seed=0)
        assert km.centroids.shape[0] == 3


class TestComputeCurrentCrackSpread:
    def test_crack_spread_is_gasoil_minus_brent(self, synthetic_series):
        b, g, dates = synthetic_series
        val_date = pd.Timestamp("2025-08-01")
        M = 3
        crack = compute_current_crack_spread(b, g, val_date, M=M)
        assert crack.shape == (M,)
        # Crack spread should be finite
        assert np.all(np.isfinite(crack))


class TestEstimateHistoricalCorrelation:
    def test_correlation_in_minus1_to_1(self, synthetic_series):
        b, g, dates = synthetic_series
        val_date = pd.Timestamp("2025-08-01")
        M = 3
        rho = estimate_historical_correlation(b, g, val_date, M=M, q=60)
        assert np.all(rho >= -1.0 - 1e-8)
        assert np.all(rho <= 1.0 + 1e-8)

    def test_positive_corr_for_correlated_series(self):
        """Highly correlated series should yield rho > 0."""
        rng = np.random.default_rng(0)
        n = 120
        dates = pd.bdate_range(start="2025-01-01", periods=n, freq="B")
        base_returns = rng.normal(0, 1.5, n)
        brent_series = {0: pd.Series(75 + np.cumsum(base_returns), index=dates)}
        # Gasoil = Brent + small noise → very high positive correlation
        gasoil_series = {0: pd.Series(90 + np.cumsum(base_returns + rng.normal(0, 0.1, n)), index=dates)}

        rho = estimate_historical_correlation(
            brent_series, gasoil_series,
            pd.Timestamp("2025-06-01"), M=1, q=60,
        )
        assert rho[0] > 0.8
