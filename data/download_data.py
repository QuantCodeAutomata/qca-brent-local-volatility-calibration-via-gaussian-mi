"""
Data loader for Brent and Gasoil futures and options data using the massive (Polygon.io) API.

Uses the massive RESTClient to download:
  - Daily aggregate bars for Brent (BZ) and Gasoil (QS) futures
  - Nearest-month continuation contracts as proxies for roll-adjusted series
  - Synthetic option IV surfaces from historical volatility

API authentication: set MASSIVE_TOKEN environment variable.

Note: ICE Brent and Gasoil options with complete IV surfaces are not universally
available via Polygon. This module constructs realistic synthetic data for
self-contained experiment replication, clearly marked as synthetic.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from massive import RESTClient

DATA_DIR = Path(__file__).parent
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_client() -> RESTClient:
    """Return authenticated massive RESTClient using MASSIVE_TOKEN env variable."""
    api_key = os.getenv("MASSIVE_TOKEN")
    if not api_key:
        raise EnvironmentError("MASSIVE_TOKEN environment variable not set")
    return RESTClient(api_key=api_key)


def download_futures_history(
    ticker: str,
    from_date: str,
    to_date: str,
    client: Optional[RESTClient] = None,
) -> pd.DataFrame:
    """
    Download daily futures aggregate bars for a ticker.

    Parameters
    ----------
    ticker    : futures ticker, e.g. 'BZ=F' for Brent, 'RB=F' for RBOB
    from_date : start date 'YYYY-MM-DD'
    to_date   : end date 'YYYY-MM-DD'
    client    : RESTClient instance (created if None)

    Returns
    -------
    DataFrame with columns: date, open, high, low, close, volume, vwap
    """
    if client is None:
        client = get_client()

    aggs = []
    for a in client.list_aggs(
        ticker=ticker,
        multiplier=1,
        timespan="day",
        from_=from_date,
        to=to_date,
        limit=50000,
        adjusted=True,
    ):
        aggs.append({
            "date": pd.Timestamp(a.timestamp, unit="ms"),
            "open": a.open,
            "high": a.high,
            "low": a.low,
            "close": a.close,
            "volume": a.volume,
            "vwap": getattr(a, "vwap", None),
        })

    if not aggs:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "vwap"])

    df = pd.DataFrame(aggs).sort_values("date").reset_index(drop=True)
    return df


def build_synthetic_brent_gasoil_data(
    valuation_dates: List[str],
    M: int = 11,
    history_days: int = 300,
    seed: int = 42,
) -> dict:
    """
    Build synthetic but economically realistic Brent and Gasoil futures data
    for use when real ICE data is unavailable.

    Generates:
    - Roll-adjusted historical series for M tenors
    - Synthetic option IV surfaces calibrated to realistic smile shapes
    - Realistic crack spread levels and correlations

    Parameters
    ----------
    valuation_dates : list of date strings 'YYYY-MM-DD'
    M               : number of tenors (paper: 11)
    history_days    : calendar days of history to generate
    seed            : random seed

    Returns
    -------
    dict with keys: 'brent_futures', 'gasoil_futures', 'brent_options', 'gasoil_options',
                    'maturities', 'valuation_dates'
    """
    rng = np.random.default_rng(seed)

    # Tenor structure: approximately monthly buckets out to ~12 months
    # Year fractions (T_m) for M=11 tenors
    tenor_months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert len(tenor_months) == M

    data_by_date = {}

    for val_date_str in valuation_dates:
        val_date = pd.Timestamp(val_date_str)
        start_date = val_date - pd.Timedelta(days=history_days)

        bdate_range = pd.bdate_range(start=start_date, end=val_date, freq="B")

        # Base Brent price around $70-80 / barrel
        brent_base = 75.0 + rng.normal(0, 2.0)
        # Gasoil ≈ Brent + crack spread ~$15-25/bbl
        crack_base = 18.0 + rng.normal(0, 3.0)
        gasoil_base = brent_base + crack_base

        # Brent normal vol ≈ $1.5-2.5 / day
        brent_daily_vol = 1.8

        # Simulate Brent price path (arithmetic BM)
        n_days = len(bdate_range)
        brent_returns = rng.normal(0, brent_daily_vol, n_days)
        brent_prices_path = brent_base + np.cumsum(brent_returns)
        brent_prices_path = np.maximum(brent_prices_path, 20.0)  # floor

        # Gasoil correlated with Brent, ρ ≈ 0.92
        rho_bg = 0.92
        gasoil_returns = (
            rho_bg * brent_returns
            + np.sqrt(1 - rho_bg**2) * rng.normal(0, brent_daily_vol * 1.1, n_days)
        )
        gasoil_prices_path = gasoil_base + np.cumsum(gasoil_returns)
        gasoil_prices_path = np.maximum(gasoil_prices_path, 30.0)

        # Build tenor-specific series (different term-structure slope)
        brent_series: Dict[int, pd.Series] = {}
        gasoil_series: Dict[int, pd.Series] = {}

        for m, months in enumerate(tenor_months):
            # Slight contango/backwardation scaling per tenor
            tau_m = months / 12.0
            term_slope_B = rng.normal(-0.5, 0.2) * tau_m  # slight backwardation
            term_slope_G = rng.normal(-0.3, 0.15) * tau_m

            brent_m = brent_prices_path + term_slope_B + rng.normal(0, 0.1, n_days)
            gasoil_m = gasoil_prices_path + term_slope_G + rng.normal(0, 0.15, n_days)

            brent_series[m] = pd.Series(brent_m, index=bdate_range)
            gasoil_series[m] = pd.Series(gasoil_m, index=bdate_range)

        # Valuation-date forwards
        F_B = np.array([float(brent_series[m].iloc[-1]) for m in range(M)])
        F_G = np.array([float(gasoil_series[m].iloc[-1]) for m in range(M)])

        # Maturities as year fractions
        tau_arr = np.array([months / 12.0 for months in tenor_months])

        # Build synthetic option IV surface
        # Normal vol ≈ ATM_IV = σ_B * √T_m  (term vol)
        # Smile: quadratic in moneyness with slight negative skew for Brent
        brent_options = _build_synthetic_iv_surface(
            F=F_B,
            tau=tau_arr,
            atm_normal_vol=brent_daily_vol * np.sqrt(252),  # annualised
            skew=-0.5,
            convexity=2.0,
            rng=rng,
            commodity="brent",
        )

        gasoil_options = _build_synthetic_iv_surface(
            F=F_G,
            tau=tau_arr,
            atm_normal_vol=brent_daily_vol * 1.1 * np.sqrt(252),
            skew=-0.3,
            convexity=2.5,
            rng=rng,
            commodity="gasoil",
        )

        data_by_date[val_date_str] = {
            "valuation_date": val_date_str,
            "brent_series": brent_series,
            "gasoil_series": gasoil_series,
            "F_B": F_B,
            "F_G": F_G,
            "maturities": tau_arr,
            "tenor_months": tenor_months,
            "brent_options": brent_options,
            "gasoil_options": gasoil_options,
        }

    return data_by_date


def _build_synthetic_iv_surface(
    F: np.ndarray,
    tau: np.ndarray,
    atm_normal_vol: float,
    skew: float,
    convexity: float,
    rng: np.random.Generator,
    commodity: str = "brent",
    n_strikes: int = 9,
) -> dict:
    """
    Build a synthetic option implied-volatility surface.

    IV(K, T_m) = atm_iv_m * (1 + skew * x + convexity * x^2 + noise)
    where x = (K - F_m) / (atm_iv_m * sqrt(T_m))  (normalised moneyness)

    Parameters
    ----------
    F            : shape (M,) forward prices
    tau          : shape (M,) year fractions
    atm_normal_vol: annualised ATM normal vol ($/bbl/year)
    skew          : smile skew parameter (negative = put skew)
    convexity     : smile convexity (positive = smile curvature)
    rng           : random number generator
    commodity     : label string

    Returns
    -------
    dict with keys: 'strikes_per_mat', 'ivs_per_mat', 'cp_per_mat', 'F_per_mat', 'tau_per_mat'
    """
    M = len(F)
    strikes_per_mat = []
    ivs_per_mat = []
    cp_per_mat = []

    for m in range(M):
        T = tau[m]
        F_m = F[m]

        # ATM normal vol for this maturity (term vol scale)
        atm_iv_m = atm_normal_vol * np.sqrt(T)  # absolute normal vol ($/bbl)
        # Small random perturbation per maturity
        atm_iv_m *= (1 + rng.normal(0, 0.02))

        # Strike grid: ±2σ in normalised units
        x_grid = np.linspace(-2.0, 2.0, n_strikes)
        K_grid = F_m + x_grid * atm_iv_m

        ivs_m = []
        cp_m = []
        for x, K in zip(x_grid, K_grid):
            # Quadratic smile
            iv = atm_iv_m * (1 + skew * x + convexity * x**2 * 0.02)
            iv = max(iv, 0.05 * atm_iv_m)  # floor
            iv += rng.normal(0, 0.005 * atm_iv_m)  # bid-ask noise

            ivs_m.append(max(iv, 1e-4))
            cp_m.append(1 if K >= F_m else -1)

        strikes_per_mat.append(np.array(K_grid))
        ivs_per_mat.append(np.array(ivs_m))
        cp_per_mat.append(np.array(cp_m))

    return {
        "strikes_per_mat": strikes_per_mat,
        "ivs_per_mat": ivs_per_mat,
        "cp_per_mat": cp_per_mat,
        "F_per_mat": F,
        "tau_per_mat": tau,
    }


def try_download_real_data(
    ticker: str,
    from_date: str,
    to_date: str,
) -> Optional[pd.DataFrame]:
    """
    Attempt to download real futures data; return None on failure.

    Parameters
    ----------
    ticker    : e.g. 'BZ=F'
    from_date : 'YYYY-MM-DD'
    to_date   : 'YYYY-MM-DD'

    Returns
    -------
    DataFrame or None
    """
    try:
        client = get_client()
        df = download_futures_history(ticker, from_date, to_date, client)
        if len(df) > 0:
            return df
    except Exception:
        pass
    return None


def save_data(data: dict, filename: str) -> None:
    """Save data dict as JSON to data directory."""
    path = DATA_DIR / filename
    # Convert numpy arrays to lists for JSON serialisation
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    with open(path, "w") as f:
        json.dump(convert(data), f, indent=2)


def load_data(filename: str) -> dict:
    """Load data dict from JSON in data directory."""
    path = DATA_DIR / filename
    with open(path) as f:
        return json.load(f)
