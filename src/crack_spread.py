"""
Gasoil-specific volatility correction h estimation from historical Brent/Gasoil futures data.

Implements the paper's Exp2 methodology:
1. Roll-adjusted tenor-specific futures histories (López de Prado gap adjustment)
2. Arithmetic-return rolling volatilities and average crack spreads
3. 2M-dimensional feature vectors, downsampled every n2 observations
4. Cityblock k-medians clustering with k=6
5. Tenor-specific linear interpolation h_m(s) → flat extrapolation beyond support

Custom implementation — Context7 found no library equivalent for the exact
feature construction + k-medians pipeline (paper Section 3 / Exp2).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from src.k_medians import k_medians, KMediansResult


# ── Roll-adjusted futures series ────────────────────────────────────────────

def build_roll_adjusted_series(
    prices: pd.DataFrame,
    roll_days_before_expiry: int = 2,
) -> pd.Series:
    """
    Build a roll-adjusted continuous futures series using López de Prado gap adjustment.

    At each roll event (contract change), the gap g = Price_new - Price_old is computed
    and subtracted from all historical values to remove the artificial jump.

    Parameters
    ----------
    prices                  : DataFrame with columns ['date', 'price', 'expiry'].
                              Sorted ascending by date. Multiple contracts
                              may be present; caller selects the 'active' one.
    roll_days_before_expiry : number of business days before expiry to roll

    Returns
    -------
    roll_adjusted : pd.Series indexed by date with gap-adjusted prices
    """
    prices = prices.sort_values("date").reset_index(drop=True)

    adjusted = prices["price"].copy()
    cumulative_gap = 0.0

    prev_contract_id = prices["expiry"].iloc[0] if "expiry" in prices.columns else None

    for i in range(1, len(prices)):
        cur_id = prices["expiry"].iloc[i] if "expiry" in prices.columns else None
        if prev_contract_id is not None and cur_id != prev_contract_id:
            # Roll event detected
            gap = prices["price"].iloc[i] - prices["price"].iloc[i - 1]
            cumulative_gap += gap
            # Subtract gap from all historical (already adjusted) values
            adjusted.iloc[:i] -= gap

        prev_contract_id = cur_id

    return pd.Series(adjusted.values, index=pd.to_datetime(prices["date"]))


def select_nearby_contract(
    all_prices: pd.DataFrame,
    target_maturity_months: int,
) -> pd.DataFrame:
    """
    Select futures contracts approximately `target_maturity_months` ahead.

    Simplified roll rule: on each trading day, select the front contract
    (or Nth contract) with expiry closest to target months from now.

    Parameters
    ----------
    all_prices              : DataFrame with columns ['date', 'price', 'expiry', 'symbol']
    target_maturity_months  : approximate months to target maturity

    Returns
    -------
    Selected contract series DataFrame with columns ['date', 'price', 'expiry']
    """
    all_prices = all_prices.copy()
    all_prices["date"] = pd.to_datetime(all_prices["date"])
    all_prices["expiry"] = pd.to_datetime(all_prices["expiry"])

    result_rows = []
    for date, group in all_prices.groupby("date"):
        group = group[group["expiry"] > date]
        if group.empty:
            continue
        target = date + pd.DateOffset(months=target_maturity_months)
        idx = (group["expiry"] - target).abs().idxmin()
        row = group.loc[idx]
        result_rows.append({"date": date, "price": row["price"], "expiry": row["expiry"]})

    return pd.DataFrame(result_rows)


# ── Feature engineering ──────────────────────────────────────────────────────

def compute_features(
    brent_series: Dict[int, pd.Series],
    gasoil_series: Dict[int, pd.Series],
    M: int = 11,
    n: int = 30,
    n2: int | None = None,
) -> pd.DataFrame:
    """
    Compute 2M-dimensional feature vectors x_t = (V(T_1),...,V(T_M), S̄(T_1),...,S̄(T_M)).

    V(t, T_m) = hat_sigma_G(t, T_m) - hat_sigma_B(t, T_m)   (vol spread)
    S̄(t, T_m) = rolling mean of crack spread over n days

    Uses arithmetic returns (price differences) to match Bachelier framework.

    Parameters
    ----------
    brent_series  : dict {tenor_idx: pd.Series of roll-adjusted Brent prices}
    gasoil_series : dict {tenor_idx: pd.Series of roll-adjusted Gasoil prices}
    M             : number of tenors
    n             : rolling window in trading days
    n2            : downsample interval (default round(n/2))

    Returns
    -------
    DataFrame of shape (n_obs, 2M) with feature vectors, after downsampling.
    Also adds a 'date' column.
    """
    if n2 is None:
        n2 = max(1, round(n / 2))

    # Align dates
    all_dates = None
    for m in range(M):
        dates = brent_series[m].index.intersection(gasoil_series[m].index)
        if all_dates is None:
            all_dates = dates
        else:
            all_dates = all_dates.intersection(dates)

    if all_dates is None or len(all_dates) < n + 1:
        return pd.DataFrame()

    all_dates = all_dates.sort_values()

    V_cols = {m: [] for m in range(M)}
    S_bar_cols = {m: [] for m in range(M)}
    valid_dates = []

    for idx in range(n, len(all_dates)):
        window_dates = all_dates[idx - n: idx]
        cur_date = all_dates[idx]
        complete = True

        v_row = np.zeros(M)
        s_bar_row = np.zeros(M)

        for m in range(M):
            brent_w = brent_series[m].reindex(window_dates)
            gasoil_w = gasoil_series[m].reindex(window_dates)

            if brent_w.isna().any() or gasoil_w.isna().any():
                complete = False
                break

            # Arithmetic returns (price differences)
            r_B = brent_w.diff().dropna()
            r_G = gasoil_w.diff().dropna()

            if len(r_B) < 2 or len(r_G) < 2:
                complete = False
                break

            sigma_B = float(r_B.std(ddof=1))
            sigma_G = float(r_G.std(ddof=1))

            # Crack spread levels
            crack_spread = gasoil_series[m].reindex(window_dates) - brent_series[m].reindex(window_dates)
            crack_spread = crack_spread.dropna()

            v_row[m] = sigma_G - sigma_B
            s_bar_row[m] = float(crack_spread.mean())

        if complete:
            V_cols_row = v_row
            S_bar_row = s_bar_row
            valid_dates.append(cur_date)

            for m in range(M):
                V_cols[m].append(V_cols_row[m])
                S_bar_cols[m].append(S_bar_row[m])

    if not valid_dates:
        return pd.DataFrame()

    feature_dict = {"date": valid_dates}
    for m in range(M):
        feature_dict[f"V_{m}"] = V_cols[m]
    for m in range(M):
        feature_dict[f"S_{m}"] = S_bar_cols[m]

    features_df = pd.DataFrame(feature_dict)

    # Downsample every n2 observations
    downsampled = features_df.iloc[::n2].reset_index(drop=True)
    return downsampled


# ── h function estimation ─────────────────────────────────────────────────────

@dataclass
class HFunction:
    """
    Tenor-specific crack-spread-to-vol-correction interpolation function h_m(s).

    Built from k-medians cluster centroids.
    Uses flat extrapolation beyond centroid support.
    """
    tenor_idx: int
    centroid_crack: np.ndarray   # c_j^S(T_m), shape (k,)
    centroid_vol_spread: np.ndarray  # c_j^V(T_m), shape (k,)
    _interp: object              # scipy interp1d object

    def __call__(self, s: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluate h_m(s): crack spread → volatility correction.

        Flat extrapolation beyond centroid support.
        """
        s = np.asarray(s, dtype=float)
        return float(self._interp(np.clip(s, self.centroid_crack.min(), self.centroid_crack.max())))


def build_h_functions(
    features_df: pd.DataFrame,
    M: int = 11,
    k: int = 6,
    n_init: int = 50,
    seed: int = 42,
) -> tuple[List[HFunction], KMediansResult]:
    """
    Estimate tenor-specific h functions from feature vectors.

    Steps:
    1. Stack feature vectors into design matrix X (n_obs, 2M)
    2. Run k-medians with cityblock distance (k=6)
    3. Extract centroid crack/vol-spread pairs per tenor
    4. Build linear interpolation with flat extrapolation

    Parameters
    ----------
    features_df : output of compute_features(), has columns V_0..V_{M-1}, S_0..S_{M-1}
    M           : number of tenors
    k           : number of k-medians clusters (paper: k=6)
    n_init      : k-medians restarts
    seed        : random seed

    Returns
    -------
    h_functions  : list of M HFunction objects
    km_result    : KMediansResult with centroids and labels
    """
    v_cols = [f"V_{m}" for m in range(M)]
    s_cols = [f"S_{m}" for m in range(M)]

    X = features_df[v_cols + s_cols].values.astype(float)

    # Remove rows with any NaN
    valid = ~np.isnan(X).any(axis=1)
    X = X[valid]

    if len(X) < k:
        raise ValueError(f"Too few observations ({len(X)}) for k={k} clustering")

    # Custom k-medians with cityblock distance (Context7: no library equivalent)
    km_result = k_medians(X, k=k, n_init=n_init, seed=seed)
    centroids = km_result.centroids  # shape (k, 2M)

    h_functions = []
    for m in range(M):
        crack_nodes = centroids[:, M + m]     # c_j^S(T_m)
        vol_nodes = centroids[:, m]           # c_j^V(T_m)

        # Handle duplicate crack nodes: average vol-spread
        unique_crack, inv = np.unique(crack_nodes, return_inverse=True)
        avg_vol = np.array([
            np.mean(vol_nodes[inv == i]) for i in range(len(unique_crack))
        ])

        # Sort by crack level
        order = np.argsort(unique_crack)
        crack_sorted = unique_crack[order]
        vol_sorted = avg_vol[order]

        if len(crack_sorted) == 1:
            # Constant function if only one unique crack node
            interp_fn = interp1d(
                [crack_sorted[0] - 1, crack_sorted[0], crack_sorted[0] + 1],
                [vol_sorted[0]] * 3,
                kind="linear",
                bounds_error=False,
                fill_value=(vol_sorted[0], vol_sorted[0]),
            )
        else:
            interp_fn = interp1d(
                crack_sorted,
                vol_sorted,
                kind="linear",
                bounds_error=False,
                fill_value=(vol_sorted[0], vol_sorted[-1]),  # flat extrapolation
            )

        h_functions.append(HFunction(
            tenor_idx=m,
            centroid_crack=crack_sorted,
            centroid_vol_spread=vol_sorted,
            _interp=interp_fn,
        ))

    return h_functions, km_result


def compute_current_crack_spread(
    brent_series: Dict[int, pd.Series],
    gasoil_series: Dict[int, pd.Series],
    valuation_date: pd.Timestamp,
    M: int = 11,
) -> np.ndarray:
    """
    Compute current (valuation-date) crack spreads S(s0, T_m).

    Parameters
    ----------
    brent_series    : dict {tenor_idx: roll-adjusted Brent price series}
    gasoil_series   : dict {tenor_idx: roll-adjusted Gasoil price series}
    valuation_date  : valuation date s0
    M               : number of tenors

    Returns
    -------
    crack : shape (M,) crack spreads S(s0, T_m) = G_m(s0) - B_m(s0)
    """
    valuation_date = pd.Timestamp(valuation_date)
    crack = np.zeros(M)
    for m in range(M):
        b = brent_series[m]
        g = gasoil_series[m]
        # Find most recent date at or before valuation_date
        b_dates = b.index[b.index <= valuation_date]
        g_dates = g.index[g.index <= valuation_date]
        if len(b_dates) == 0 or len(g_dates) == 0:
            crack[m] = 0.0
        else:
            b_val = float(b.loc[b_dates[-1]])
            g_val = float(g.loc[g_dates[-1]])
            crack[m] = g_val - b_val
    return crack


def estimate_historical_correlation(
    brent_series: Dict[int, pd.Series],
    gasoil_series: Dict[int, pd.Series],
    valuation_date: pd.Timestamp,
    M: int = 11,
    q: int = 90,
) -> np.ndarray:
    """
    Estimate tenor-specific historical correlations ρ̂_{T_m}^{G,B} over last q days.

    Uses arithmetic returns, consistent with Bachelier framework.

    Parameters
    ----------
    brent_series   : dict of roll-adjusted series
    gasoil_series  : dict of roll-adjusted series
    valuation_date : valuation date s0
    M              : number of tenors
    q              : lookback window in business days (paper: q=90)

    Returns
    -------
    rho : shape (M,) correlation estimates per tenor
    """
    valuation_date = pd.Timestamp(valuation_date)
    rho = np.zeros(M)

    for m in range(M):
        b = brent_series[m]
        g = gasoil_series[m]

        # Select data up to valuation date
        b_hist = b[b.index <= valuation_date].tail(q + 1)
        g_hist = g[g.index <= valuation_date].tail(q + 1)

        common_dates = b_hist.index.intersection(g_hist.index)
        if len(common_dates) < 3:
            rho[m] = 0.9  # default high correlation
            continue

        b_w = b_hist.loc[common_dates]
        g_w = g_hist.loc[common_dates]

        r_B = b_w.diff().dropna()
        r_G = g_w.diff().dropna()

        if len(r_B) < 2:
            rho[m] = 0.9
            continue

        corr_mat = np.corrcoef(r_B.values, r_G.values)
        rho[m] = np.clip(corr_mat[0, 1], -0.999, 0.999)

    return rho
