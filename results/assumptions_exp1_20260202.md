# Exp 1 Assumptions — 2026-02-02

## Implementation Assumptions (paper omissions resolved)

- **N (mixture components)**: 3
  Tested N ∈ {2,3,4,5}; N=3 chosen as smallest delivering stable fit and monotone cumulative variances.

- **Optimizer**: Nelder-Mead (1 random restarts) + L-BFGS-B refinement
  Calibration in implied-vol space (not price space) as specified by paper.

- **Monotone projection**: scipy `isotonic_regression` (PAVA) with equal weights,
  applied separately to each component's cumulative variance term structure.

- **Dynamic component drifts μᵢ(t)**: Set to zero for all i and t.
  This preserves the futures martingale condition and simplifies Proposition 2.1
  to σ_B²(t,y) = Σᵢ wᵢ(t,y)·γᵢ²(t) (density-weighted average of bucket variances).

- **MC validation**: 1000 paths (antithetic), 10 Euler steps, seed=42.

## Results Summary

- Overall calibration RMSE (mixture fit): 1.4606
- MC validation RMSE: 12.9992
- MC validation MAE:  10.0724
