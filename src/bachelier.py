"""
Bachelier (normal / arithmetic Brownian motion) option pricing and implied-vol inversion.

Uses pyfeng.Norm — Context7 confirmed: Choi2009 closed-form IV inversion, vectorised pricing.
Falls back to direct formula + scipy.brentq for edge cases.
"""

from __future__ import annotations

import numpy as np
import pyfeng
from scipy.optimize import brentq
from scipy.stats import norm as spnorm


# ── Shared pyfeng.Norm instance (stateless helper) ──────────────────────────
_NORM_MODEL = pyfeng.Norm(sigma=1.0)

EPSILON = 1e-10  # numerical floor


def bachelier_price(
    F: float | np.ndarray,
    K: float | np.ndarray,
    sigma_n: float | np.ndarray,
    T: float | np.ndarray,
    cp: int = 1,
) -> float | np.ndarray:
    """
    Bachelier (normal model) option price.

    Parameters
    ----------
    F      : forward / futures price
    K      : strike
    sigma_n: normal implied volatility (in price units, not %)
    T      : time to maturity (year fraction)
    cp     : +1 for call, -1 for put

    Returns
    -------
    Bachelier option price (same units as F and K)

    Notes
    -----
    Formula: C = (F-K)·Φ(d) + σ_N·√T·φ(d),  d = (F-K)/(σ_N·√T)
    Uses pyfeng.Norm.price_formula — Context7 confirmed.
    """
    sigma_n = np.asarray(sigma_n, dtype=float)
    T = np.asarray(T, dtype=float)
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)

    # pyfeng convention: is_fwd=True means first arg is forward
    return pyfeng.Norm.price_formula(
        strike=K,
        spot=F,
        sigma=sigma_n,
        texp=T,
        cp=cp,
        is_fwd=True,
    )


def bachelier_impvol(
    price: float | np.ndarray,
    F: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    cp: int = 1,
) -> float | np.ndarray:
    """
    Invert the Bachelier formula to recover normal implied volatility.

    Uses pyfeng.Norm.impvol (Choi2009 closed-form approximation) — Context7 confirmed.
    Falls back to scalar scipy.brentq on failure.

    Parameters
    ----------
    price  : market option price
    F      : forward / futures price
    K      : strike
    T      : time to maturity (year fraction)
    cp     : +1 for call, -1 for put

    Returns
    -------
    Normal implied volatility σ_N in price units.
    Returns np.nan if inversion fails.
    """
    price = np.asarray(price, dtype=float)
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)

    model = pyfeng.Norm(sigma=None)
    try:
        iv = model.impvol(price, strike=K, spot=F, texp=T, cp=cp, is_fwd=True)
        return iv
    except Exception:
        pass

    # Scalar fallback via brentq
    scalar = price.ndim == 0
    price_1d = np.atleast_1d(price)
    F_1d = np.broadcast_to(np.atleast_1d(F), price_1d.shape)
    K_1d = np.broadcast_to(np.atleast_1d(K), price_1d.shape)
    T_1d = np.broadcast_to(np.atleast_1d(T), price_1d.shape)

    out = np.empty(price_1d.shape)
    for idx in np.ndindex(price_1d.shape):
        try:
            out[idx] = _brentq_impvol(
                float(price_1d[idx]),
                float(F_1d[idx]),
                float(K_1d[idx]),
                float(T_1d[idx]),
                cp,
            )
        except Exception:
            out[idx] = np.nan
    return float(out[0]) if scalar else out


def _brentq_impvol(
    price: float, F: float, K: float, T: float, cp: int
) -> float:
    """Scalar Brent-method implied-vol solver for Bachelier formula."""
    intrinsic = max(cp * (F - K), 0.0)
    if price < intrinsic - EPSILON:
        return np.nan
    if T < EPSILON:
        return np.nan

    def objective(sigma: float) -> float:
        return float(bachelier_price(F, K, sigma, T, cp)) - price

    sigma_hi = max(10.0 * abs(F - K + 1.0), 10.0)
    try:
        return brentq(objective, EPSILON, sigma_hi, xtol=1e-10, rtol=1e-10)
    except ValueError:
        return np.nan


def bachelier_mixture_price(
    F: float,
    K: float,
    T: float,
    weights: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    cp: int = 1,
) -> float:
    """
    Gaussian-mixture Bachelier option price.

    Price = Σᵢ λᵢ · Bachelier(F=μᵢ, K, σᵢ/√T·√T, T, cp)
    where component i has distribution N(μᵢ, σᵢ²).

    Because each component is Gaussian N(μᵢ, σᵢ²), the Bachelier price
    for that component is exactly: E[max(cp·(X-K), 0)] with X ~ N(μᵢ, σᵢ²).
    This equals bachelier_price(F=μᵢ, K, sigma_n=σᵢ/√T, T, cp).

    Parameters
    ----------
    F       : overall futures price (for reference; not used in price, only for constraint)
    K       : strike
    T       : time to maturity
    weights : mixture weights λᵢ (summing to 1)
    means   : component means μᵢ in price units
    stds    : component standard deviations σᵢ (sqrt(V_i), NOT V_i/T)
    cp      : +1 call, -1 put

    Returns
    -------
    Gaussian-mixture option price
    """
    weights = np.asarray(weights, dtype=float)
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)

    if T < EPSILON:
        # At expiry
        return float(np.sum(weights * np.maximum(cp * (means - K), 0.0)))

    # Component normal-vol: σ_N,i = std_i / √T (so that Var = σ_N²·T = std²)
    sigma_n_i = stds / np.sqrt(T)

    prices_i = np.array([
        float(bachelier_price(means[i], K, sigma_n_i[i], T, cp))
        for i in range(len(weights))
    ])
    return float(np.dot(weights, prices_i))


def bachelier_mixture_impvol(
    F: float,
    K: float,
    T: float,
    weights: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    cp: int = 1,
) -> float:
    """
    Normal implied vol of a Gaussian-mixture Bachelier option.

    Computes the mixture price then inverts the Bachelier formula.

    Returns np.nan if inversion fails.
    """
    mix_price = bachelier_mixture_price(F, K, T, weights, means, stds, cp)
    intrinsic = max(cp * (F - K), 0.0)

    if mix_price < intrinsic - EPSILON or T < EPSILON:
        return np.nan

    return bachelier_impvol(mix_price, F, K, T, cp)


def gaussian_density(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Standard Gaussian PDF N(x; mu, sigma^2)."""
    sigma = max(sigma, EPSILON)
    return spnorm.pdf(x, loc=mu, scale=sigma)


def gaussian_cdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Standard Gaussian CDF Φ((x-mu)/sigma)."""
    sigma = max(sigma, EPSILON)
    return spnorm.cdf(x, loc=mu, scale=sigma)


def gaussian_survival(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Complementary CDF P(X > x) for X ~ N(mu, sigma^2)."""
    sigma = max(sigma, EPSILON)
    return spnorm.sf(x, loc=mu, scale=sigma)
