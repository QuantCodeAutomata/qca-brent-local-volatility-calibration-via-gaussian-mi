"""
Brent local-volatility function reconstruction from calibrated Gaussian-mixture parameters.

Implements Proposition 2.1 of the paper under the simplifying assumption μᵢ(t) = 0
for all components i and all times t (documented implementation assumption).

With μᵢ(t) = 0, the second term of Proposition 2.1 vanishes and the formula reduces to:

    σ_B²(t, y) = Σᵢ wᵢ(t, y) · γᵢ²(t)

where the density weights are:
    wᵢ(t, y) = λᵢ · p_t^i(y) / Σⱼ λⱼ · p_t^j(y)

and each p_t^i is Gaussian:
    p_t^i(y) ~ N(μᵢ^t, Vᵢ(t))
    μᵢ^t = F   (since μᵢ(s) = 0 ⟹ mean stays at forward)
    Vᵢ(t) = ∫₀ᵗ γᵢ²(s) ds  (cumulative variance up to t)

Custom implementation — Context7 found no library equivalent (paper Proposition 2.1).
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import norm as spnorm

from src.gaussian_mixture import MixtureParams
from src.bachelier import EPSILON


@dataclass
class BrentLocalVolModel:
    """
    Callable Brent local-volatility model σ_B(t, y).

    Built from calibrated Gaussian-mixture parameters and piecewise-constant
    bucket volatilities.

    Attributes
    ----------
    maturities     : sorted array of quoted maturities T_m
    forwards       : array of futures prices B(0, T_m)
    weights        : shape (N, M) — λᵢ per maturity (may vary, but here fixed per component)
    component_fwds : shape (N,) — component means at t=0 (= F since μᵢ=0)
    bucket_vars    : shape (N, M) — γ²_{i,m}, bucket variance for component i on [T_{m-1}, T_m)
    cum_vars       : shape (N, M) — Vᵢ(T_m) projected cumulative variances
    N              : number of Gaussian components
    M              : number of quoted maturities
    """

    maturities: np.ndarray   # shape (M,)
    forwards: np.ndarray     # shape (M,)
    weights: np.ndarray      # shape (N, M) — λᵢ (from maturity-wise calibration)
    cum_vars: np.ndarray     # shape (N, M) — Vᵢ(T_m) projected
    bucket_vars: np.ndarray  # shape (N, M) — γ²_{i,m}
    N: int
    M: int

    def _bucket_index(self, t: float) -> int:
        """Return the maturity bucket index m such that T_{m-1} ≤ t < T_m."""
        idx = bisect.bisect_right(self.maturities, t)
        return min(idx, self.M - 1)

    def _cum_var_at_t(self, t: float, i: int) -> float:
        """
        Compute cumulative variance Vᵢ(t) = ∫₀ᵗ γᵢ²(s) ds
        using piecewise-constant bucket vols.

        For t ≤ T_1:  Vᵢ(t) = γ²_{i,1} · t
        For T_{m-1} ≤ t ≤ T_m:
            Vᵢ(t) = Vᵢ(T_{m-1}) + γ²_{i,m} · (t - T_{m-1})
        """
        if t <= 0.0:
            return 0.0
        if t <= self.maturities[0]:
            return float(self.bucket_vars[i, 0]) * t

        m = self._bucket_index(t)
        v = 0.0
        for k in range(m):
            if k == 0:
                v += float(self.bucket_vars[i, 0]) * float(self.maturities[0])
            else:
                dt = float(self.maturities[k]) - float(self.maturities[k - 1])
                v += float(self.bucket_vars[i, k]) * dt

        if m < self.M:
            T_prev = self.maturities[m - 1] if m > 0 else 0.0
            v += float(self.bucket_vars[i, m]) * (t - float(T_prev))

        return max(v, 0.0)

    def _component_mean_at_t(self, t: float, mat_idx: int) -> float:
        """
        Component mean at time t.
        Under μᵢ(t) = 0 assumption, mean stays at forward B(0, T_m).
        We use the maturity-matched forward as reference (approximation for dynamic paths).
        """
        return float(self.forwards[mat_idx])

    def _weights_at_mat(self, mat_idx: int) -> np.ndarray:
        """Return mixture weights λᵢ for the given maturity bucket."""
        return self.weights[:, mat_idx]

    def sigma_B(self, t: float, y: float, mat_idx: int = -1) -> float:
        """
        Local-volatility σ_B(t, y) from Proposition 2.1 with μᵢ=0.

        σ_B²(t, y) = Σᵢ wᵢ(t, y) · γᵢ²(t)

        where wᵢ(t, y) = λᵢ · φ(y; μᵢ(t), Vᵢ(t)) / Σⱼ λⱼ · φ(y; μⱼ(t), Vⱼ(t))

        Parameters
        ----------
        t       : current time
        y       : current futures level B(t, T_m)
        mat_idx : index into self.maturities (for weight and forward lookup);
                  if -1, use bucket from t

        Returns
        -------
        σ_B(t, y) ≥ 0
        """
        if t <= 0.0:
            t = 1e-6

        m_idx = mat_idx if mat_idx >= 0 else self._bucket_index(t)
        m_idx = min(m_idx, self.M - 1)

        lam = self._weights_at_mat(m_idx)

        # Compute Vᵢ(t) and current bucket variance γᵢ²(t)
        V_t = np.array([self._cum_var_at_t(t, i) for i in range(self.N)])
        gamma_sq_t = np.array([
            float(self.bucket_vars[i, self._bucket_index(t)])
            for i in range(self.N)
        ])

        # Component means: B(0, T_m) under μᵢ=0
        mu_t = np.full(self.N, float(self.forwards[m_idx]))

        # Component densities pᵢ(y; t)
        densities = np.array([
            spnorm.pdf(y, loc=float(mu_t[i]), scale=max(np.sqrt(V_t[i]), EPSILON))
            for i in range(self.N)
        ])

        # Weighted densities
        w_unnorm = lam * densities
        denom = w_unnorm.sum()

        if denom < EPSILON:
            # Fallback: equal-weight average of bucket vols
            sigma_sq = float(np.dot(lam, gamma_sq_t))
        else:
            w = w_unnorm / denom
            sigma_sq = float(np.dot(w, gamma_sq_t))

        return max(np.sqrt(sigma_sq), EPSILON)

    def sigma_B_vectorised(
        self,
        t: float,
        y_arr: np.ndarray,
        mat_idx: int,
    ) -> np.ndarray:
        """
        Vectorised version of sigma_B over an array of y values.

        Parameters
        ----------
        t       : current time
        y_arr   : array of futures levels
        mat_idx : maturity index

        Returns
        -------
        Array of local volatilities, same shape as y_arr
        """
        y_arr = np.asarray(y_arr, dtype=float)
        if t <= 0.0:
            t = 1e-6

        m_idx = min(mat_idx, self.M - 1)
        lam = self._weights_at_mat(m_idx)

        V_t = np.array([self._cum_var_at_t(t, i) for i in range(self.N)])
        gamma_sq_t = np.array([
            float(self.bucket_vars[i, self._bucket_index(t)])
            for i in range(self.N)
        ])
        mu_t = np.full(self.N, float(self.forwards[m_idx]))

        # densities: shape (N, len(y_arr))
        densities = np.array([
            spnorm.pdf(y_arr, loc=float(mu_t[i]), scale=max(np.sqrt(V_t[i]), EPSILON))
            for i in range(self.N)
        ])

        w_unnorm = lam[:, None] * densities  # (N, len_y)
        denom = w_unnorm.sum(axis=0)          # (len_y,)

        safe_denom = np.where(denom < EPSILON, 1.0, denom)
        w = w_unnorm / safe_denom[None, :]   # (N, len_y)

        # Fallback mask where denom is too small
        fallback_sigma_sq = float(np.dot(lam, gamma_sq_t))
        sigma_sq = np.where(
            denom < EPSILON,
            fallback_sigma_sq,
            np.dot(gamma_sq_t, w),            # (len_y,)
        )

        return np.maximum(np.sqrt(sigma_sq), EPSILON)


def build_local_vol_model(
    mixture_params_list: list[MixtureParams],
    bucket_vars: np.ndarray,
    cum_vars_projected: np.ndarray,
) -> BrentLocalVolModel:
    """
    Construct a BrentLocalVolModel from calibration outputs.

    Parameters
    ----------
    mixture_params_list  : list of MixtureParams, one per maturity (sorted ascending)
    bucket_vars          : shape (N, M) piecewise bucket variances
    cum_vars_projected   : shape (N, M) projected cumulative variances

    Returns
    -------
    BrentLocalVolModel callable local-vol function
    """
    M = len(mixture_params_list)
    N = len(mixture_params_list[0].weights)

    maturities = np.array([p.maturity for p in mixture_params_list])
    forwards = np.array([p.forward for p in mixture_params_list])
    weights = np.column_stack([p.weights for p in mixture_params_list])  # (N, M)

    return BrentLocalVolModel(
        maturities=maturities,
        forwards=forwards,
        weights=weights,
        cum_vars=cum_vars_projected,
        bucket_vars=bucket_vars,
        N=N,
        M=M,
    )
