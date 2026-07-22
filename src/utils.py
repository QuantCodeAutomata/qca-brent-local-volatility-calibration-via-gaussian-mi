"""
Utility functions for the Brent/Gasoil local-volatility framework.

Includes:
- Year-fraction computation
- IV surface alignment and moneyness grid construction
- Roll-gap adjustments
- Plotting helpers
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def year_fraction(date1: pd.Timestamp, date2: pd.Timestamp, convention: str = "ACT365") -> float:
    """
    Compute time to maturity as a year fraction.

    Parameters
    ----------
    date1      : reference / valuation date
    date2      : maturity date
    convention : day count convention ('ACT365', 'ACT360', 'ACT252')

    Returns
    -------
    tau : year fraction τ = (date2 - date1).days / day_count
    """
    days = (date2 - date1).days
    if days <= 0:
        return 1e-6  # near-zero but non-zero for numerical stability
    denom = {"ACT365": 365.0, "ACT360": 360.0, "ACT252": 252.0}[convention]
    return days / denom


def build_moneyness_grid(
    n_points: int = 21,
    min_moneyness: float = 0.85,
    max_moneyness: float = 1.15,
) -> np.ndarray:
    """
    Construct a common moneyness grid k = K/F.

    Parameters
    ----------
    n_points      : number of grid points
    min_moneyness : minimum k (default 0.85 → 15% OTM put)
    max_moneyness : maximum k (default 1.15 → 15% OTM call)

    Returns
    -------
    grid : shape (n_points,) moneyness values
    """
    return np.linspace(min_moneyness, max_moneyness, n_points)


def compute_rmse_mae(
    model: np.ndarray,
    market: np.ndarray,
) -> tuple[float, float]:
    """Compute RMSE and MAE between two arrays (ignoring NaN)."""
    diff = model - market
    valid = ~np.isnan(diff)
    if valid.sum() == 0:
        return np.nan, np.nan
    d = diff[valid]
    return float(np.sqrt(np.mean(d**2))), float(np.mean(np.abs(d)))


def plot_smile_comparison(
    maturities: np.ndarray,
    moneyness: np.ndarray,
    brent_mkt: np.ndarray,
    gasoil_mkt: np.ndarray,
    model_ivs_dict: dict,
    date_str: str,
    save_path: Optional[Path] = None,
    max_mats: int = 6,
) -> plt.Figure:
    """
    Plot smile comparison: market Brent vs market Gasoil vs model Gasoil surfaces.

    Parameters
    ----------
    maturities      : shape (M,) year fractions
    moneyness       : shape (J,) common moneyness grid
    brent_mkt       : shape (M, J) Brent market IVs
    gasoil_mkt      : shape (M, J) Gasoil market IVs
    model_ivs_dict  : dict of {label: np.ndarray (M, J)} model IV surfaces
    date_str        : valuation date string for title
    save_path       : optional file path to save figure
    max_mats        : maximum number of maturities to plot

    Returns
    -------
    matplotlib Figure
    """
    M = len(maturities)
    n_plot = min(M, max_mats)
    indices = np.linspace(0, M - 1, n_plot, dtype=int)

    ncols = 3
    nrows = (n_plot + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.array(axes).flatten()

    colors = {"full": "tab:blue", "brent_only": "tab:orange", "fixed_spread": "tab:green",
              "brent_model": "tab:gray", "anchored": "tab:purple"}
    markers = {"full": "o", "brent_only": "s", "fixed_spread": "^",
               "brent_model": "x", "anchored": "D"}

    for plot_idx, m in enumerate(indices):
        ax = axes[plot_idx]
        T = maturities[m]

        if brent_mkt is not None and not np.all(np.isnan(brent_mkt[m])):
            ax.plot(moneyness, brent_mkt[m], "k--", label="Brent mkt", lw=2)
        if gasoil_mkt is not None and not np.all(np.isnan(gasoil_mkt[m])):
            ax.plot(moneyness, gasoil_mkt[m], "k-", label="Gasoil mkt", lw=2)

        for label, iv_surf in model_ivs_dict.items():
            if not np.all(np.isnan(iv_surf[m])):
                ax.plot(
                    moneyness, iv_surf[m],
                    color=colors.get(label, "tab:red"),
                    marker=markers.get(label, "o"),
                    markevery=3,
                    label=label,
                    lw=1.5,
                )

        ax.set_title(f"T={T:.2f}y")
        ax.set_xlabel("Moneyness k=K/F")
        ax.set_ylabel("Normal IV")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for ax in axes[n_plot:]:
        ax.set_visible(False)

    fig.suptitle(f"Smile Comparison — {date_str}", fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_h_functions(
    h_functions: list,
    tenor_labels: list[str],
    date_str: str,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot h_m(s) interpolation curves for all tenors.

    Parameters
    ----------
    h_functions  : list of HFunction objects
    tenor_labels : list of tenor label strings
    date_str     : valuation date string
    save_path    : optional save path

    Returns
    -------
    matplotlib Figure
    """
    M = len(h_functions)
    ncols = 4
    nrows = (M + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = np.array(axes).flatten()

    for m, hf in enumerate(h_functions):
        ax = axes[m]
        s_range = np.linspace(
            hf.centroid_crack.min() - 0.1 * abs(hf.centroid_crack.min()),
            hf.centroid_crack.max() + 0.1 * abs(hf.centroid_crack.max()),
            200,
        )
        h_vals = np.array([float(hf(s)) for s in s_range])

        ax.plot(s_range, h_vals, "b-", lw=2)
        ax.scatter(hf.centroid_crack, hf.centroid_vol_spread, c="red", zorder=5,
                   label="Centroids", s=60)
        ax.set_title(f"h_{m}(s) — {tenor_labels[m] if m < len(tenor_labels) else m}")
        ax.set_xlabel("Crack spread S(t,T_m)")
        ax.set_ylabel("Vol correction h")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for ax in axes[M:]:
        ax.set_visible(False)

    fig.suptitle(f"Volatility Correction h_m(s) — {date_str}", fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_brent_calibration(
    maturities: np.ndarray,
    strikes_per_mat: list,
    market_ivs_per_mat: list,
    model_ivs_per_mat: list,
    date_str: str,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot Brent calibration fit: market vs model IV smiles per maturity.

    Parameters
    ----------
    maturities          : shape (M,)
    strikes_per_mat     : list of M arrays of strikes
    market_ivs_per_mat  : list of M arrays of market normal IVs
    model_ivs_per_mat   : list of M arrays of model normal IVs
    date_str            : valuation date string
    save_path           : optional save path

    Returns
    -------
    matplotlib Figure
    """
    M = len(maturities)
    ncols = 3
    nrows = (M + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.array(axes).flatten()

    for m in range(M):
        ax = axes[m]
        K = strikes_per_mat[m]
        iv_mkt = market_ivs_per_mat[m]
        iv_mod = model_ivs_per_mat[m]

        ax.plot(K, iv_mkt, "ko-", label="Market", lw=2, ms=5)
        ax.plot(K, iv_mod, "r^--", label="Model", lw=2, ms=5)
        ax.set_title(f"T={maturities[m]:.2f}y")
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Normal IV")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for ax in axes[M:]:
        ax.set_visible(False)

    fig.suptitle(f"Brent Calibration Fit — {date_str}", fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def save_results_md(content: str, mode: str = "w") -> None:
    """Write or append to results/RESULTS.md."""
    path = RESULTS_DIR / "RESULTS.md"
    with open(path, mode) as f:
        f.write(content)


def make_date_str(date) -> str:
    """Convert various date formats to YYYY-MM-DD string."""
    return pd.Timestamp(date).strftime("%Y-%m-%d")
