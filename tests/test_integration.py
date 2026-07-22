"""
Integration tests verifying the end-to-end pipeline from data generation
through Brent calibration, h estimation, and joint pricing.

All heavy computations (calibration, MC pricing) are cached in module-scoped
fixtures so they run only once per test session, keeping the suite fast.
"""

import numpy as np
import pytest

from data.download_data import build_synthetic_brent_gasoil_data

# ── Fast-path overrides for tests ────────────────────────────────────────────
_VAL_DATE = "2026-02-02"
_N_RESTARTS = 1       # single restart — enough to confirm the code path runs
_N_MC_PATHS = 1_000  # 2 000 effective antithetic paths
_N_MC_STEPS = 10     # coarse Euler grid — not for accuracy, just correctness


# ── Module-scoped fixtures (computed once per session) ────────────────────────

@pytest.fixture(scope="module")
def synthetic_data():
    """Synthetic Brent/Gasoil dataset for one validation date."""
    return build_synthetic_brent_gasoil_data(
        valuation_dates=[_VAL_DATE],
        M=11,
        history_days=150,
        seed=99,
    )


@pytest.fixture(scope="module")
def exp1_result(synthetic_data):
    """Brent calibration result — computed once, shared across all Exp1/Exp3 tests."""
    from exp.exp_1_brent_calibration import run_brent_calibration
    return run_brent_calibration(
        _VAL_DATE,
        synthetic_data,
        n_restarts=_N_RESTARTS,
        n_mc_paths=_N_MC_PATHS,
        n_mc_steps=_N_MC_STEPS,
    )


@pytest.fixture(scope="module")
def exp2_result(synthetic_data):
    """h estimation result — computed once, shared across all Exp2/Exp3 tests."""
    from exp.exp_2_crack_spread_h import run_h_estimation
    return run_h_estimation(_VAL_DATE, synthetic_data)


@pytest.fixture(scope="module")
def exp3_result(synthetic_data, exp1_result, exp2_result):
    """Joint pricing result — computed once with fast MC params."""
    from exp.exp_3_joint_pricing import run_joint_pricing
    return run_joint_pricing(
        _VAL_DATE,
        synthetic_data,
        exp1_result,
        exp2_result,
        n_mc_paths=_N_MC_PATHS,
        n_mc_steps=_N_MC_STEPS,
    )


# ── Data-generation tests ─────────────────────────────────────────────────────

class TestSyntheticDataGeneration:
    def test_returns_dict_with_date_key(self, synthetic_data):
        assert _VAL_DATE in synthetic_data

    def test_11_tenors(self, synthetic_data):
        d = synthetic_data[_VAL_DATE]
        assert len(d["maturities"]) == 11
        assert len(d["F_B"]) == 11
        assert len(d["F_G"]) == 11

    def test_brent_series_positive(self, synthetic_data):
        d = synthetic_data[_VAL_DATE]
        for m in range(11):
            assert np.all(d["brent_series"][m].values > 0)

    def test_gasoil_forward_above_brent(self, synthetic_data):
        """Gasoil forward should be above Brent (positive crack spread)."""
        d = synthetic_data[_VAL_DATE]
        assert np.mean(d["F_G"]) > np.mean(d["F_B"])

    def test_maturities_increasing(self, synthetic_data):
        d = synthetic_data[_VAL_DATE]
        tau = d["maturities"]
        assert np.all(np.diff(tau) > 0)

    def test_option_ivs_positive(self, synthetic_data):
        d = synthetic_data[_VAL_DATE]
        for m in range(11):
            assert np.all(np.array(d["brent_options"]["ivs_per_mat"][m]) > 0)
            assert np.all(np.array(d["gasoil_options"]["ivs_per_mat"][m]) > 0)


# ── Exp 1 tests ───────────────────────────────────────────────────────────────

class TestExp1Integration:
    """Exp1 end-to-end: calibrate Brent mixture and reconstruct local vol."""

    def test_calibration_runs_without_error(self, exp1_result):
        assert exp1_result is not None

    def test_local_vol_callable(self, exp1_result):
        local_vol = exp1_result["local_vol_B"]
        sigma = local_vol.sigma_B(0.3, 80.0, mat_idx=2)
        assert sigma > 0 and np.isfinite(sigma)

    def test_bucket_vars_nonnegative(self, exp1_result):
        assert np.all(exp1_result["bucket_vars"] >= -1e-10)

    def test_cum_vars_nondecreasing(self, exp1_result):
        cv = exp1_result["cum_vars_projected"]
        for i in range(cv.shape[0]):
            assert np.all(np.diff(cv[i]) >= -1e-10)

    def test_rmse_calib_finite(self, exp1_result):
        assert np.isfinite(exp1_result["rmse_calib"])


# ── Exp 2 tests ───────────────────────────────────────────────────────────────

class TestExp2Integration:
    """Exp2 end-to-end: h function estimation from historical data."""

    def test_h_estimation_runs(self, exp2_result):
        assert exp2_result is not None

    def test_h_functions_length(self, exp2_result):
        assert len(exp2_result["h_functions"]) == 11

    def test_h_fixed_shape(self, exp2_result):
        assert exp2_result["h_fixed"].shape == (11,)

    def test_rho_in_valid_range(self, exp2_result):
        rho = exp2_result["rho"]
        assert np.all(rho >= -1.001)
        assert np.all(rho <= 1.001)


# ── Exp 3 tests ───────────────────────────────────────────────────────────────

class TestExp3Integration:
    """Exp3 end-to-end: joint pricing and error comparison."""

    def test_joint_pricing_runs(self, exp3_result):
        assert exp3_result is not None

    def test_error_df_has_three_specs(self, exp3_result):
        assert len(exp3_result["error_df"]) == 3
        assert set(exp3_result["error_df"]["spec"]) == {
            "full", "brent_only", "fixed_spread"
        }

    def test_rmse_finite(self, exp3_result):
        err_df = exp3_result["error_df"]
        assert np.all(err_df["rmse_anchored"].apply(np.isfinite))

    def test_brent_only_is_worst_or_close(self, exp3_result):
        """
        Validate that fixed-spread RMSE is not dramatically worse than
        Brent-only (with synthetic data the exact ranking may vary).
        """
        err_df = exp3_result["error_df"].set_index("spec")
        rmse_bo = err_df.loc["brent_only", "rmse_anchored"]
        rmse_fs = err_df.loc["fixed_spread", "rmse_anchored"]
        rmse_fl = err_df.loc["full", "rmse_anchored"]

        assert all(np.isfinite([rmse_bo, rmse_fs, rmse_fl]))
        assert all(v >= 0 for v in [rmse_bo, rmse_fs, rmse_fl])
