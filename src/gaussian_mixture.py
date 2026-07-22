"""
Gaussian-mixture terminal marginal calibration for Brent futures.

Methodology: Section 2 of the paper — maturity-wise Gaussian-mixture marginals
under the Bachelier framework.

Custom implementation — Context7 found no library equivalent for Gaussian-mixture
Bachelier calibration with futures mean constraints (paper Eq. 2.x).

Parameterization:
  - λᵢ = softmax(aᵢ)                    (admissibility constraint, sums to 1)
  - μᵢ = direct (last backed from Σλᵢμᵢ = F)
  - Vᵢ = exp(vᵢ)                         (positivity of variance)
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize, OptimizeResult
from scipy.optimize import isotonic_regression  # Context7 confirmed: PAVA O(N)

from src.bachelier import (
    bachelier_mixture_impvol,
    bachelier_mixture_price,
    EPSILON,
)


class MixtureParams(NamedTuple):
    """Calibrated Gaussian-mixture parameters for a single maturity."""
    maturity: float          # T_m
    forward: float           # F_m = B(0, T_m)
    weights: np.ndarray      # λᵢ  shape (N,)
    means: np.ndarray        # μᵢ  shape (N,)  (means of terminal distribution)
    stds: np.ndarray         # σᵢ  shape (N,)  (std of terminal distribution, NOT vol)
    cum_variances: np.ndarray # Vᵢ = σᵢ²     shape (N,)
    rmse_iv: float           # calibration RMSE in normal-vol units


def _softmax(a: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    a = a - np.max(a)
    e = np.exp(a)
    return e / e.sum()


def _pack_params(
    weights: np.ndarray,
    means: np.ndarray,
    log_vars: np.ndarray,
    N: int,
) -> np.ndarray:
    """
    Pack (unconstrained) parameters into a flat vector.
    weights[N-1] and means[N-1] are backed-out, so we optimise N-1 of each.
    log_vars are unconstrained equivalents of log(V_i).
    """
    return np.concatenate([
        weights[:N - 1],    # N-1 softmax logits
        means[:N - 1],      # N-1 free means
        log_vars,           # N log-variances
    ])


def _unpack_params(
    theta: np.ndarray,
    N: int,
    F: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unpack flat vector into admissible (weights, means, stds).

    Returns
    -------
    weights   shape (N,)
    means     shape (N,)
    stds      shape (N,)  (standard deviation = sqrt(V_i))
    """
    raw_w = np.zeros(N)
    raw_w[: N - 1] = theta[: N - 1]
    # raw_w[N-1] = 0.0  (reference logit)
    weights = _softmax(raw_w)

    # Free means for components 0..N-2; back out last mean from martingale constraint
    free_means = theta[N - 1: 2 * N - 2]
    means = np.zeros(N)
    means[: N - 1] = free_means
    means[N - 1] = (F - np.dot(weights[: N - 1], free_means)) / weights[N - 1]

    log_vars = theta[2 * N - 2:]
    stds = np.exp(0.5 * log_vars)  # sqrt(exp(v)) = exp(v/2)

    return weights, means, stds


def _iv_residuals(
    theta: np.ndarray,
    N: int,
    F: float,
    T: float,
    strikes: np.ndarray,
    market_ivs: np.ndarray,
    call_flags: np.ndarray,
) -> np.ndarray:
    """Implied-vol residuals vector for scipy.optimize."""
    weights, means, stds = _unpack_params(theta, N, F)

    if np.any(stds < EPSILON):
        return np.full(len(strikes), 1e3)

    residuals = []
    for K, iv_mkt, cp in zip(strikes, market_ivs, call_flags):
        iv_model = bachelier_mixture_impvol(F, float(K), T, weights, means, stds, int(cp))
        if np.isnan(iv_model):
            iv_model = iv_mkt + 1.0  # large penalty
        residuals.append(iv_model - iv_mkt)

    return np.array(residuals)


def calibrate_mixture_maturity(
    F: float,
    T: float,
    strikes: np.ndarray,
    market_ivs: np.ndarray,
    call_flags: np.ndarray,
    N: int = 3,
    n_restarts: int = 8,
    seed: int = 42,
) -> MixtureParams:
    """
    Calibrate a Gaussian-mixture terminal distribution for one maturity.

    Minimises sum((σ_model - σ_mkt)²) over mixture parameters.
    Uses softmax/log-variance parameterisation for admissibility.

    Parameters
    ----------
    F          : futures price B(0, T_m)
    T          : time to maturity (year fraction)
    strikes    : array of strikes K_{m,j}
    market_ivs : array of market normal IVs σ_IV_mkt (price units)
    call_flags : array of +1 (call) / -1 (put) for each quote
    N          : number of Gaussian components
    n_restarts : number of random restarts for optimisation
    seed       : random seed

    Returns
    -------
    MixtureParams with calibrated parameters
    """
    rng = np.random.default_rng(seed)
    atm_iv = float(np.median(market_ivs))
    atm_std = atm_iv * np.sqrt(T)  # approximate terminal std

    best_result: OptimizeResult | None = None
    best_cost = np.inf

    for restart in range(n_restarts):
        # Random initial means around F
        init_means = F + rng.normal(0, atm_std * 0.3, N)
        # Uniform initial weights
        init_w = np.zeros(N)  # all zero logits → equal weights
        # Initial log-variances from ATM vol
        init_log_v = np.full(N, np.log(atm_std**2 + EPSILON))

        theta0 = _pack_params(init_w, init_means, init_log_v, N)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                fun=lambda th: 0.5 * np.sum(_iv_residuals(th, N, F, T, strikes, market_ivs, call_flags) ** 2),
                x0=theta0,
                method="Nelder-Mead",
                options={"maxiter": 5000, "xatol": 1e-7, "fatol": 1e-10},
            )

        if result.fun < best_cost:
            best_cost = result.fun
            best_result = result

    # Try L-BFGS-B from best Nelder-Mead result
    if best_result is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            refine = minimize(
                fun=lambda th: 0.5 * np.sum(_iv_residuals(th, N, F, T, strikes, market_ivs, call_flags) ** 2),
                x0=best_result.x,
                method="L-BFGS-B",
                options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-8},
            )
        if refine.fun < best_cost:
            best_result = refine

    theta_opt = best_result.x
    weights, means, stds = _unpack_params(theta_opt, N, F)

    # RMSE in IV space
    residuals = _iv_residuals(theta_opt, N, F, T, strikes, market_ivs, call_flags)
    rmse_iv = float(np.sqrt(np.mean(residuals**2)))

    cum_variances = stds**2  # V_i = σᵢ²

    return MixtureParams(
        maturity=T,
        forward=F,
        weights=weights,
        means=means,
        stds=stds,
        cum_variances=cum_variances,
        rmse_iv=rmse_iv,
    )


def calibrate_mixture_surface(
    forwards: np.ndarray,
    maturities: np.ndarray,
    strikes_per_mat: list[np.ndarray],
    ivs_per_mat: list[np.ndarray],
    cp_per_mat: list[np.ndarray],
    N: int = 3,
    n_restarts: int = 8,
    seed: int = 42,
) -> list[MixtureParams]:
    """
    Calibrate Gaussian-mixture marginals for all M maturities.

    Parameters
    ----------
    forwards          : shape (M,) Brent futures B(0,T_m)
    maturities        : shape (M,) year fractions T_m
    strikes_per_mat   : list of M strike arrays
    ivs_per_mat       : list of M market-IV arrays (normal vol, price units)
    cp_per_mat        : list of M call-flag arrays (+1/-1)
    N                 : number of mixture components
    n_restarts        : restarts per maturity calibration
    seed              : base random seed

    Returns
    -------
    List of MixtureParams, one per maturity, sorted by ascending maturity
    """
    M = len(maturities)
    assert M == len(forwards) == len(strikes_per_mat) == len(ivs_per_mat) == len(cp_per_mat)

    order = np.argsort(maturities)
    results: list[MixtureParams] = []

    for rank, m in enumerate(order):
        params = calibrate_mixture_maturity(
            F=float(forwards[m]),
            T=float(maturities[m]),
            strikes=strikes_per_mat[m],
            market_ivs=ivs_per_mat[m],
            call_flags=cp_per_mat[m],
            N=N,
            n_restarts=n_restarts,
            seed=seed + rank,
        )
        results.append(params)

    return results


def apply_monotone_projection(
    cum_vars_per_component: np.ndarray,
    maturities: np.ndarray,
) -> np.ndarray:
    """
    Enforce non-decreasing cumulative variances across maturities for each component.

    Uses scipy.optimize.isotonic_regression (PAVA, O(N)) — Context7 confirmed.

    Parameters
    ----------
    cum_vars_per_component : shape (N, M) — component i, maturity m
    maturities             : shape (M,)  year fractions (sorted ascending)

    Returns
    -------
    projected : shape (N, M) non-decreasing along axis=1
    """
    N, M = cum_vars_per_component.shape
    projected = np.empty_like(cum_vars_per_component)

    for i in range(N):
        raw_seq = cum_vars_per_component[i]
        # scipy.optimize.isotonic_regression returns OptimizeResult with .x
        result = isotonic_regression(raw_seq, increasing=True)
        projected[i] = result.x

    return projected


def compute_bucket_vols(
    cum_vars_projected: np.ndarray,
    maturities: np.ndarray,
) -> np.ndarray:
    """
    Compute piecewise-constant component bucket volatilities γ_{i,m}.

    γ_{i,1}² = V_{T1}^{i,1} / T_1
    γ_{i,m}² = (V_{Tm}^{i,m} - V_{T_{m-1}}^{i,m-1}) / (T_m - T_{m-1})  for m≥2

    Parameters
    ----------
    cum_vars_projected : shape (N, M) non-decreasing cumulative variances
    maturities         : shape (M,) ascending year fractions

    Returns
    -------
    bucket_vars : shape (N, M) piecewise-constant bucket variances γ²_{i,m}
                  (floored at 0 to avoid numerical negatives)
    """
    N, M = cum_vars_projected.shape
    bucket_vars = np.empty((N, M))

    for i in range(N):
        bucket_vars[i, 0] = cum_vars_projected[i, 0] / maturities[0]
        for m in range(1, M):
            dt = maturities[m] - maturities[m - 1]
            dv = cum_vars_projected[i, m] - cum_vars_projected[i, m - 1]
            bucket_vars[i, m] = max(dv / dt, 0.0)  # floor at 0

    return bucket_vars
