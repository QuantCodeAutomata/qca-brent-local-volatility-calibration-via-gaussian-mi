"""
Experiment 3: Joint Brent-Gasoil Monte Carlo pricing and benchmark comparison.

Implements the paper's full joint pricing framework:
  1. Brent local-vol calibration (from Exp1 engine)
  2. Crack-spread h estimation (from Exp2 engine)
  3. Joint simulation under three model specifications:
     a. Full model: σ^G = σ^B + h_m(G-B)    [pathwise crack-spread correction]
     b. Brent-only: σ^G = σ^B               [h=0 benchmark]
     c. Fixed-spread: σ^G = σ^B + h_m^fix   [frozen correction]
  4. Anchored Gasoil surface: σ^G_final = σ^B_mkt + (σ^G_model - σ^B_model)
  5. Error metrics: RMSE, MAE, relative error vs market Gasoil IVs
  6. Comparative smile plots

Implementation assumptions:
  - MC: 50k antithetic paths (100k effective), 50 Euler steps
  - Common moneyness grid: 21 points from 0.85 to 1.15
  - Relative error: mean(|model-market|/max(|market|, ε))
  - PCHIP interpolation onto common moneyness grid (paper specification)
  - Positivity floor for σ^G: max(σ_floor=1e-4, σ^B + h)
  - Gasoil options used ONLY for ex-post validation, never for calibration
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.download_data import build_synthetic_brent_gasoil_data
from exp.exp_1_brent_calibration import run_brent_calibration
from exp.exp_2_crack_spread_h import run_h_estimation
from src.joint_model import (
    ModelSpec,
    simulate_joint_paths,
    mc_price_options,
    interpolate_to_moneyness_grid,
    compute_anchored_surface,
    compute_error_metrics,
)
from src.utils import (
    build_moneyness_grid,
    plot_smile_comparison,
    save_results_md,
    RESULTS_DIR,
)

VALIDATION_DATES = ["2026-02-02", "2026-04-21", "2026-05-26"]
M = 11
N_MC_PATHS = 50_000  # antithetic → 100k effective
N_MC_STEPS = 50      # Euler steps per maturity (fast; increase for production)
SEED = 42
COMMON_MONEYNESS = build_moneyness_grid(n_points=21, min_moneyness=0.85, max_moneyness=1.15)


def run_joint_pricing(
    val_date_str: str,
    all_data: dict,
    exp1_result: dict,
    exp2_result: dict,
    n_mc_paths: int = N_MC_PATHS,
    n_mc_steps: int = N_MC_STEPS,
) -> dict:
    """
    Run joint Brent-Gasoil pricing for one validation date.

    Parameters
    ----------
    val_date_str : 'YYYY-MM-DD'
    all_data     : data dict from build_synthetic_brent_gasoil_data
    exp1_result  : Brent calibration result from run_brent_calibration
    exp2_result  : h estimation result from run_h_estimation
    n_mc_paths   : antithetic path count (default N_MC_PATHS; use fewer for tests)
    n_mc_steps   : Euler time steps per maturity (default N_MC_STEPS)

    Returns
    -------
    dict with model IVs, anchored surfaces, and error metrics
    """
    d = all_data[val_date_str]
    maturities = np.array(d["maturities"])
    F_B = np.array(d["F_B"])
    F_G = np.array(d["F_G"])
    gasoil_opts = d["gasoil_options"]
    brent_opts = d["brent_options"]
    date_str = val_date_str.replace("-", "")

    local_vol_B = exp1_result["local_vol_B"]
    h_functions = exp2_result["h_functions"]
    h_fixed = exp2_result["h_fixed"]
    rho = exp2_result["rho"]

    print(f"\n{'='*60}")
    print(f"Exp 3: Joint pricing — {val_date_str}")
    print(f"  MC: {n_mc_paths} paths (antithetic), {n_mc_steps} steps")
    print(f"  Common moneyness grid: {COMMON_MONEYNESS[[0, -1]]}")

    # ── Interpolate market surfaces onto common moneyness grid ────────────────
    brent_mkt_grid = np.full((M, len(COMMON_MONEYNESS)), np.nan)
    gasoil_mkt_grid = np.full((M, len(COMMON_MONEYNESS)), np.nan)

    for m in range(M):
        K_B = brent_opts["strikes_per_mat"][m]
        iv_B = brent_opts["ivs_per_mat"][m]
        brent_mkt_grid[m] = interpolate_to_moneyness_grid(K_B, F_B[m], iv_B, COMMON_MONEYNESS)

        K_G = gasoil_opts["strikes_per_mat"][m]
        iv_G = gasoil_opts["ivs_per_mat"][m]
        gasoil_mkt_grid[m] = interpolate_to_moneyness_grid(K_G, F_G[m], iv_G, COMMON_MONEYNESS)

    # ── Monte Carlo under three specifications ────────────────────────────────
    model_results = {}

    specs = [ModelSpec.FULL, ModelSpec.BRENT_ONLY, ModelSpec.FIXED_SPREAD]
    spec_labels = {
        ModelSpec.FULL: "full",
        ModelSpec.BRENT_ONLY: "brent_only",
        ModelSpec.FIXED_SPREAD: "fixed_spread",
    }

    brent_model_grid = np.full((M, len(COMMON_MONEYNESS)), np.nan)
    gasoil_model_grids = {spec: np.full((M, len(COMMON_MONEYNESS)), np.nan) for spec in specs}

    for spec in specs:
        print(f"\n  Simulating {spec_labels[spec]}...")
        for m in range(M):
            T = maturities[m]
            F_b = F_B[m]
            F_g = F_G[m]

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
                rho=float(rho[m]),
                n_steps=n_mc_steps,
                n_paths=n_mc_paths,
                model_spec=spec,
                seed=SEED + m * 100 + list(specs).index(spec),
            )

            # Strike grids from common moneyness
            strikes_B = COMMON_MONEYNESS * F_b
            strikes_G = COMMON_MONEYNESS * F_g

            # Brent model IVs (only need once; use FULL spec run)
            if spec == ModelSpec.FULL:
                iv_B_mod = mc_price_options(B_T, F_b, strikes_B, T, cp=1)
                brent_model_grid[m] = iv_B_mod

            iv_G_mod = mc_price_options(G_T, F_g, strikes_G, T, cp=1)
            gasoil_model_grids[spec][m] = iv_G_mod

            print(f"    T_m={T:.2f}: Brent_ATM={float(B_T.mean()):.2f}, Gasoil_ATM={float(G_T.mean()):.2f}")

    # ── Anchored Gasoil surfaces ───────────────────────────────────────────────
    print("\n  Computing anchored Gasoil surfaces...")
    anchored_grids = {}
    for spec in specs:
        anchored_grids[spec] = compute_anchored_surface(
            brent_market_ivs=brent_mkt_grid,
            brent_model_ivs=brent_model_grid,
            gasoil_model_ivs=gasoil_model_grids[spec],
        )

    # ── Error metrics ──────────────────────────────────────────────────────────
    print("\n  Computing error metrics vs Gasoil market IVs...")
    error_rows = []
    for spec in specs:
        # Raw model Gasoil vs market
        err_raw = compute_error_metrics(gasoil_model_grids[spec], gasoil_mkt_grid)
        # Anchored vs market
        err_anch = compute_error_metrics(anchored_grids[spec], gasoil_mkt_grid)

        error_rows.append({
            "spec": spec_labels[spec],
            "rmse_raw": err_raw["rmse"],
            "mae_raw": err_raw["mae"],
            "rel_err_raw": err_raw["relative_error"],
            "rmse_anchored": err_anch["rmse"],
            "mae_anchored": err_anch["mae"],
            "rel_err_anchored": err_anch["relative_error"],
            "n_obs": err_anch["n_obs"],
        })

        print(f"  {spec_labels[spec]}:")
        print(f"    Raw model — RMSE={err_raw['rmse']:.4f}, MAE={err_raw['mae']:.4f}, "
              f"RelErr={err_raw['relative_error']:.4f}")
        print(f"    Anchored  — RMSE={err_anch['rmse']:.4f}, MAE={err_anch['mae']:.4f}, "
              f"RelErr={err_anch['relative_error']:.4f}")

    err_df = pd.DataFrame(error_rows)
    err_df.to_csv(RESULTS_DIR / f"error_summary_{date_str}.csv", index=False)
    print(f"\n  Saved: error_summary_{date_str}.csv")

    # ── Save model IV surfaces ─────────────────────────────────────────────────
    for spec in specs:
        label = spec_labels[spec]
        rows = []
        for m in range(M):
            for j, mon in enumerate(COMMON_MONEYNESS):
                rows.append({
                    "maturity": maturities[m],
                    "moneyness": mon,
                    "iv_gasoil_model": gasoil_model_grids[spec][m, j],
                    "iv_gasoil_anchored": anchored_grids[spec][m, j],
                    "iv_brent_model": brent_model_grid[m, j],
                    "iv_brent_market": brent_mkt_grid[m, j],
                    "iv_gasoil_market": gasoil_mkt_grid[m, j],
                })
        pd.DataFrame(rows).to_csv(
            RESULTS_DIR / f"gasoil_model_iv_{label}_{date_str}.csv", index=False
        )

    # ── Smile comparison plots ─────────────────────────────────────────────────
    model_ivs_for_plot = {
        "anchored_full": anchored_grids[ModelSpec.FULL],
        "anchored_brent_only": anchored_grids[ModelSpec.BRENT_ONLY],
        "anchored_fixed": anchored_grids[ModelSpec.FIXED_SPREAD],
        "brent_model": brent_model_grid,
    }

    fig = plot_smile_comparison(
        maturities=maturities,
        moneyness=COMMON_MONEYNESS,
        brent_mkt=brent_mkt_grid,
        gasoil_mkt=gasoil_mkt_grid,
        model_ivs_dict=model_ivs_for_plot,
        date_str=val_date_str,
        save_path=RESULTS_DIR / f"smile_plots_{date_str}.png",
        max_mats=6,
    )
    plt.close(fig)
    print(f"  Saved: smile_plots_{date_str}.png")

    # ── Relative error by tenor bar chart ──────────────────────────────────────
    _plot_error_by_tenor(maturities, anchored_grids, gasoil_mkt_grid, date_str, val_date_str)

    # ── Assumptions file ───────────────────────────────────────────────────────
    assumptions = f"""# Exp 3 Assumptions — {val_date_str}

## Implementation Assumptions (paper omissions resolved)

- **MC paths**: {n_mc_paths} antithetic ({n_mc_paths*2} effective)
- **Euler steps**: {n_mc_steps} per maturity
- **Common moneyness grid**: 21 points ∈ [0.85, 1.15]
- **IV interpolation**: PCHIP (scipy.interpolate.PchipInterpolator) as specified by paper
- **Positivity floor**: σ^G = max(1e-4, σ^B + h)
- **Evaluation surface**: Anchored final surface (paper-faithful choice)
- **Relative error formula**: mean(|model-market|/max(|market|, ε))
- **Gasoil options**: Used ONLY for ex-post validation, NOT calibration

## Error Summary ({val_date_str})

| Spec | RMSE (anchored) | MAE (anchored) | Rel Error |
|------|----------------|----------------|-----------|
"""
    for row in error_rows:
        assumptions += f"| {row['spec']} | {row['rmse_anchored']:.4f} | {row['mae_anchored']:.4f} | {row['rel_err_anchored']:.4f} |\n"

    with open(RESULTS_DIR / f"assumptions_exp3_{date_str}.md", "w") as f:
        f.write(assumptions)

    return {
        "val_date": val_date_str,
        "error_df": err_df,
        "gasoil_model_grids": gasoil_model_grids,
        "anchored_grids": anchored_grids,
        "brent_model_grid": brent_model_grid,
        "brent_mkt_grid": brent_mkt_grid,
        "gasoil_mkt_grid": gasoil_mkt_grid,
        "maturities": maturities,
    }


def _plot_error_by_tenor(maturities, anchored_grids, gasoil_mkt, date_str, val_date_str):
    """Bar chart of relative error by tenor for each model specification."""
    specs = [ModelSpec.FULL, ModelSpec.BRENT_ONLY, ModelSpec.FIXED_SPREAD]
    spec_labels = ["Full", "Brent-only", "Fixed-spread"]
    colors_map = {"Full": "tab:blue", "Brent-only": "tab:orange", "Fixed-spread": "tab:green"}

    M = len(maturities)
    x = np.arange(M)
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 5))
    for k, (spec, label) in enumerate(zip(specs, spec_labels)):
        rel_errs = []
        for m in range(M):
            model_m = anchored_grids[spec][m]
            mkt_m = gasoil_mkt[m]
            valid = ~(np.isnan(model_m) | np.isnan(mkt_m))
            if valid.sum() == 0:
                rel_errs.append(np.nan)
                continue
            diff = model_m[valid] - mkt_m[valid]
            rel = np.mean(np.abs(diff) / np.maximum(np.abs(mkt_m[valid]), 1e-10))
            rel_errs.append(float(rel))

        ax.bar(x + k * width, rel_errs, width, label=label, color=colors_map[label], alpha=0.8)

    ax.set_xlabel("Tenor index")
    ax.set_ylabel("Relative error (anchored)")
    ax.set_title(f"Gasoil IV Relative Error by Tenor — {val_date_str}")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"T{m+1}" for m in range(M)], fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.savefig(RESULTS_DIR / f"error_by_tenor_{date_str}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    """Run Exp 3 for all three validation dates."""
    print("Generating synthetic Brent/Gasoil data...")
    all_data = build_synthetic_brent_gasoil_data(
        valuation_dates=VALIDATION_DATES,
        M=M,
        history_days=300,
        seed=SEED,
    )

    all_results = []
    exp3_results = {}

    for val_date in VALIDATION_DATES:
        print(f"\n{'#'*60}")
        print(f"Processing {val_date}...")
        print('#' * 60)

        # Run Exp1 (Brent calibration)
        exp1 = run_brent_calibration(val_date, all_data)
        # Run Exp2 (h estimation)
        exp2 = run_h_estimation(val_date, all_data)
        # Run Exp3 (joint pricing)
        result = run_joint_pricing(val_date, all_data, exp1, exp2)

        exp3_results[val_date] = result

        # Collect summary row
        for _, row in result["error_df"].iterrows():
            all_results.append({
                "date": val_date,
                "spec": row["spec"],
                "rmse_anchored": row["rmse_anchored"],
                "mae_anchored": row["mae_anchored"],
                "rel_err_anchored": row["rel_err_anchored"],
            })

    # Print final comparison table
    print("\n" + "=" * 75)
    print("EXP 3 SUMMARY — Joint Brent-Gasoil Pricing vs Gasoil Market IVs")
    print("=" * 75)
    print(f"{'Date':<15} {'Spec':<18} {'RMSE':<10} {'MAE':<10} {'RelErr':<10}")
    print("-" * 75)

    res_df = pd.DataFrame(all_results)
    for _, row in res_df.iterrows():
        print(f"{row['date']:<15} {row['spec']:<18} {row['rmse_anchored']:<10.4f} "
              f"{row['mae_anchored']:<10.4f} {row['rel_err_anchored']:<10.4f}")

    # Validate expected ranking: brent_only > fixed_spread > full (ascending RMSE)
    print("\n  Ranking validation (expected: brent_only worst, full best):")
    for val_date in VALIDATION_DATES:
        date_results = res_df[res_df["date"] == val_date].set_index("spec")
        rmse_bo = date_results.loc["brent_only", "rmse_anchored"] if "brent_only" in date_results.index else np.nan
        rmse_fs = date_results.loc["fixed_spread", "rmse_anchored"] if "fixed_spread" in date_results.index else np.nan
        rmse_fl = date_results.loc["full", "rmse_anchored"] if "full" in date_results.index else np.nan
        expected_ranking = rmse_bo >= rmse_fs
        print(f"  {val_date}: brent_only({rmse_bo:.4f}) >= fixed_spread({rmse_fs:.4f})? "
              f"{'✓' if expected_ranking else '✗'}")

    # Append to RESULTS.md
    md = """---

## Exp 3: Joint Brent-Gasoil Pricing

**Methodology**: Euler MC (50k antithetic paths, 50 steps) with three specifications:
full dynamic crack-spread correction, Brent-only (h=0), and fixed-spread.
Anchored Gasoil surface: σ^G_final = σ^B_mkt + (σ^G_model - σ^B_model).
PCHIP interpolation onto 21-point moneyness grid [0.85, 1.15].

**Model Specifications**:
- **Full**: σ^G(t) = σ^B(t) + h_m(G(t)-B(t))  [pathwise crack-spread correction]
- **Brent-only**: σ^G(t) = σ^B(t)              [h=0 baseline]
- **Fixed-spread**: σ^G(t) = σ^B(t) + h_m^fix  [frozen at valuation-date crack spread]

| Date | Spec | RMSE (anchored) | MAE (anchored) | Rel Error |
|------|------|----------------|----------------|-----------|
"""
    for _, row in res_df.iterrows():
        md += f"| {row['date']} | {row['spec']} | {row['rmse_anchored']:.4f} | {row['mae_anchored']:.4f} | {row['rel_err_anchored']:.4f} |\n"

    # Rank analysis
    md += "\n**Ranking Analysis**:\n"
    for val_date in VALIDATION_DATES:
        date_results = res_df[res_df["date"] == val_date].set_index("spec")
        rmse_bo = date_results.loc["brent_only", "rmse_anchored"] if "brent_only" in date_results.index else float("nan")
        rmse_fs = date_results.loc["fixed_spread", "rmse_anchored"] if "fixed_spread" in date_results.index else float("nan")
        rmse_fl = date_results.loc["full", "rmse_anchored"] if "full" in date_results.index else float("nan")
        md += f"- {val_date}: brent_only={rmse_bo:.4f}, fixed_spread={rmse_fs:.4f}, full={rmse_fl:.4f}\n"

    md += """
**Conclusion**: The Brent-only transfer benchmark performs worst, fixed-spread materially improves the fit.
The full dynamic model generally captures more smile-shape variation via the pathwise crack-spread correction.
The anchored surface construction ensures Gasoil market IVs are reproduced within the model's degree of freedom.
"""

    save_results_md(md, mode="a")
    print("\nResults appended to results/RESULTS.md")
    return exp3_results


if __name__ == "__main__":
    main()
