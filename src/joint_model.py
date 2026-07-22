"""
Joint Brent-Gasoil Euler Monte Carlo pricing under correlated Bachelier dynamics.

Model specifications:
    a. Full model:  σ^G(t,T_m) = σ^B(t, B(t,T_m)) + h_m(G(t,T_m) - B(t,T_m))
    b. Brent-only:  σ^G(t,T_m) = σ^B(t, B(t,T_m))   [h=0]
    c. Fixed-spread: σ^G(t,T_m) = σ^B(t, B(t,T_m)) + h_m^fix  [frozen at s0 crack spread]

Correlated Brownian increments:
    ΔW^B = √Δt · Z^B
    ΔW^G = √Δt · [ρ̂ · Z^B + √(1-ρ̂²) · Z^G]

Custom implementation — Context7 found no library equivalent for the exact
joint local-vol MC with crack-spread-dependent correction (paper Section 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional

import numpy as np

from src.bachelier import bachelier_impvol, EPSILON
from src.local_vol import BrentLocalVolModel
from src.crack_spread import HFunction


SIGMA_FLOOR = 1e-4   # positivity floor for total vol (paper does not address, implementation safeguard)


class ModelSpec(str, Enum):
    FULL = "full"
    BRENT_ONLY = "brent_only"
    FIXED_SPREAD = "fixed_spread"


@dataclass
class SmileResult:
    """IV smile for one maturity, one model spec."""
    maturity: float
    forward_brent: float
    forward_gasoil: float
    strikes: np.ndarray      # in price units
    moneyness: np.ndarray    # K / F_G (relative)
    iv_brent_model: np.ndarray
    iv_gasoil_model: np.ndarray
    iv_gasoil_anchored: np.ndarray  # anchored: σ_B_mkt + (σ_G_model - σ_B_model)
    iv_brent_market: Optional[np.ndarray] = None
    iv_gasoil_market: Optional[np.ndarray] = None


def simulate_joint_paths(
    T: float,
    F_B: float,
    F_G: float,
    local_vol_B: BrentLocalVolModel,
    mat_idx: int,
    h_func: Optional[HFunction],
    h_fix: Optional[float],
    rho: float,
    n_steps: int,
    n_paths: int,
    model_spec: ModelSpec,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate joint (B, G) futures paths under Euler discretisation.

    Parameters
    ----------
    T            : maturity (year fraction)
    F_B          : initial Brent futures price B(0,T_m)
    F_G          : initial Gasoil futures price G(0,T_m)
    local_vol_B  : Brent local-vol callable σ_B(t, y, mat_idx)
    mat_idx      : maturity index into local_vol_B
    h_func       : HFunction h_m(s) [for full model], or None
    h_fix        : frozen h_m^fix value [for fixed-spread model], or None
    rho          : correlation ρ̂_{T_m}^{G,B}
    n_steps      : time discretisation steps
    n_paths      : number of MC paths (antithetic variates doubles this internally)
    model_spec   : FULL / BRENT_ONLY / FIXED_SPREAD
    seed         : random seed

    Returns
    -------
    B_T : shape (2*n_paths,) — Brent terminal values B(T,T)
    G_T : shape (2*n_paths,) — Gasoil terminal values G(T,T)
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    rho_comp = np.sqrt(max(1.0 - rho**2, 0.0))

    n_half = n_paths  # antithetic variates: simulate n_half paths, mirror them

    B = np.full(n_half, F_B)
    G = np.full(n_half, F_G)

    B_anti = np.full(n_half, F_B)
    G_anti = np.full(n_half, F_G)

    t = 0.0
    for step in range(n_steps):
        Z_B = rng.standard_normal(n_half)
        Z_G = rng.standard_normal(n_half)

        dW_B = sqrt_dt * Z_B
        dW_G = sqrt_dt * (rho * Z_B + rho_comp * Z_G)

        # Antithetic
        dW_B_anti = -dW_B
        dW_G_anti = -dW_G

        # Brent local vol (vectorised over paths)
        sigma_B = local_vol_B.sigma_B_vectorised(t, B, mat_idx)
        sigma_B_anti = local_vol_B.sigma_B_vectorised(t, B_anti, mat_idx)

        # Gasoil vol correction
        if model_spec == ModelSpec.BRENT_ONLY:
            corr = np.zeros(n_half)
            corr_anti = np.zeros(n_half)
        elif model_spec == ModelSpec.FIXED_SPREAD:
            h_val = h_fix if h_fix is not None else 0.0
            corr = np.full(n_half, h_val)
            corr_anti = np.full(n_half, h_val)
        else:  # FULL: pathwise crack spread
            crack = G - B
            crack_anti = G_anti - B_anti
            corr = np.array([float(h_func(float(c))) for c in crack])
            corr_anti = np.array([float(h_func(float(c))) for c in crack_anti])

        sigma_G = np.maximum(sigma_B + corr, SIGMA_FLOOR)
        sigma_G_anti = np.maximum(sigma_B_anti + corr_anti, SIGMA_FLOOR)

        B += sigma_B * dW_B
        G += sigma_G * dW_G

        B_anti += sigma_B_anti * dW_B_anti
        G_anti += sigma_G_anti * dW_G_anti

        t += dt

    B_T = np.concatenate([B, B_anti])
    G_T = np.concatenate([G, G_anti])
    return B_T, G_T


def mc_price_options(
    terminal_vals: np.ndarray,
    F: float,
    strikes: np.ndarray,
    T: float,
    cp: int = 1,
) -> np.ndarray:
    """
    Monte Carlo option prices and invert to normal implied vols.

    Parameters
    ----------
    terminal_vals : shape (n_paths,) terminal asset values
    F             : initial futures price
    strikes       : array of strikes
    T             : time to maturity
    cp            : +1 call, -1 put

    Returns
    -------
    ivs : shape (len(strikes),) normal implied vols
    """
    ivs = np.empty(len(strikes))
    for j, K in enumerate(strikes):
        payoffs = np.maximum(cp * (terminal_vals - K), 0.0)
        price = float(payoffs.mean())
        ivs[j] = bachelier_impvol(price, F, float(K), T, cp)

    return ivs


def interpolate_to_moneyness_grid(
    strikes: np.ndarray,
    F: float,
    ivs: np.ndarray,
    common_moneyness: np.ndarray,
) -> np.ndarray:
    """
    Interpolate an IV smile onto a common moneyness grid k = K/F.

    Uses PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) as specified
    by the paper. Flat extrapolation beyond the strike range.

    Parameters
    ----------
    strikes          : observed strikes
    F                : forward price
    ivs              : observed IVs (same length as strikes)
    common_moneyness : target moneyness grid k = K/F

    Returns
    -------
    ivs_interp : IVs on common moneyness grid
    """
    from scipy.interpolate import PchipInterpolator  # Context7: no-oscillation monotone

    observed_moneyness = strikes / F
    order = np.argsort(observed_moneyness)
    mon = observed_moneyness[order]
    iv = ivs[order]

    # Remove NaN
    valid = ~np.isnan(iv)
    mon = mon[valid]
    iv = iv[valid]

    if len(mon) < 2:
        return np.full(len(common_moneyness), np.nanmean(iv) if len(iv) > 0 else np.nan)

    pchip = PchipInterpolator(mon, iv, extrapolate=False)
    iv_interp = pchip(common_moneyness)

    # Flat extrapolation for out-of-support moneyness
    iv_interp = np.where(common_moneyness < mon[0], iv[0], iv_interp)
    iv_interp = np.where(common_moneyness > mon[-1], iv[-1], iv_interp)

    return iv_interp


def price_full_surface(
    local_vol_B: BrentLocalVolModel,
    h_functions: List[HFunction],
    h_fixed: np.ndarray,
    rho_per_tenor: np.ndarray,
    F_B: np.ndarray,
    F_G: np.ndarray,
    maturities: np.ndarray,
    common_moneyness: np.ndarray,
    n_steps: int = 50,
    n_paths: int = 50_000,
    seed: int = 0,
) -> dict[ModelSpec, np.ndarray]:
    """
    Price the full Gasoil implied-vol surface for all three model specifications.

    Parameters
    ----------
    local_vol_B      : Brent local-vol model
    h_functions      : list of M HFunction objects
    h_fixed          : shape (M,) frozen valuation-date h corrections
    rho_per_tenor    : shape (M,) correlation estimates
    F_B              : shape (M,) Brent forward prices
    F_G              : shape (M,) Gasoil forward prices
    maturities       : shape (M,) year fractions T_m
    common_moneyness : shape (J,) moneyness grid k = K/F
    n_steps          : Euler time steps per maturity
    n_paths          : MC paths (antithetic doubles this internally)
    seed             : base random seed

    Returns
    -------
    Dict mapping ModelSpec → np.ndarray of shape (M, J) Gasoil IVs
    Also includes key "brent_model" → shape (M, J) Brent model IVs
    """
    M = len(maturities)
    J = len(common_moneyness)

    result: dict[ModelSpec | str, np.ndarray] = {
        ModelSpec.FULL: np.full((M, J), np.nan),
        ModelSpec.BRENT_ONLY: np.full((M, J), np.nan),
        ModelSpec.FIXED_SPREAD: np.full((M, J), np.nan),
        "brent_model": np.full((M, J), np.nan),
    }

    for m, (T, F_b, F_g, rho) in enumerate(zip(maturities, F_B, F_G, rho_per_tenor)):
        if T < EPSILON:
            continue

        # Strike grid from common moneyness
        strikes_G = common_moneyness * F_g
        strikes_B = common_moneyness * F_b

        for spec in [ModelSpec.FULL, ModelSpec.BRENT_ONLY, ModelSpec.FIXED_SPREAD]:
            h_fn = h_functions[m] if spec == ModelSpec.FULL else None
            h_fix = float(h_fixed[m]) if spec == ModelSpec.FIXED_SPREAD else None

            B_T, G_T = simulate_joint_paths(
                T=T,
                F_B=F_b,
                F_G=F_g,
                local_vol_B=local_vol_B,
                mat_idx=m,
                h_func=h_fn,
                h_fix=h_fix,
                rho=rho,
                n_steps=n_steps,
                n_paths=n_paths,
                model_spec=spec,
                seed=seed + m * 10 + list(ModelSpec).index(spec),
            )

            # Brent model IVs (from first spec run only)
            if spec == ModelSpec.FULL:
                b_ivs = mc_price_options(B_T, F_b, strikes_B, T, cp=1)
                result["brent_model"][m] = b_ivs

            g_ivs = mc_price_options(G_T, F_g, strikes_G, T, cp=1)
            result[spec][m] = g_ivs

    return result


def compute_anchored_surface(
    brent_market_ivs: np.ndarray,
    brent_model_ivs: np.ndarray,
    gasoil_model_ivs: np.ndarray,
) -> np.ndarray:
    """
    Compute anchored final Gasoil IV surface.

    σ_IV^{G,final}(k,T_m) = σ_IV^{B,mkt}(k,T_m) + [σ_IV^{G,mod}(k,T_m) - σ_IV^{B,mod}(k,T_m)]

    Parameters
    ----------
    brent_market_ivs  : shape (M, J) — liquid Brent market IVs
    brent_model_ivs   : shape (M, J) — Brent model IVs
    gasoil_model_ivs  : shape (M, J) — Gasoil model IVs (any specification)

    Returns
    -------
    anchored : shape (M, J) final Gasoil IV surface
    """
    delta_iv = gasoil_model_ivs - brent_model_ivs
    return brent_market_ivs + delta_iv


def compute_error_metrics(
    model_ivs: np.ndarray,
    market_ivs: np.ndarray,
) -> dict[str, float]:
    """
    Compute RMSE, MAE, and relative error between model and market IVs.

    Parameters
    ----------
    model_ivs  : shape (M, J)
    market_ivs : shape (M, J)

    Returns
    -------
    dict with keys: 'rmse', 'mae', 'relative_error', 'n_obs'
    """
    diff = model_ivs - market_ivs
    valid = ~(np.isnan(diff) | np.isnan(market_ivs))

    if valid.sum() == 0:
        return {"rmse": np.nan, "mae": np.nan, "relative_error": np.nan, "n_obs": 0}

    d = diff[valid]
    m = market_ivs[valid]

    rmse = float(np.sqrt(np.mean(d**2)))
    mae = float(np.mean(np.abs(d)))
    rel_err = float(np.mean(np.abs(d) / np.maximum(np.abs(m), EPSILON)))

    return {"rmse": rmse, "mae": mae, "relative_error": rel_err, "n_obs": int(valid.sum())}
