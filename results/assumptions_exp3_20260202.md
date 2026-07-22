# Exp 3 Assumptions — 2026-02-02

## Implementation Assumptions (paper omissions resolved)

- **MC paths**: 1000 antithetic (2000 effective)
- **Euler steps**: 10 per maturity
- **Common moneyness grid**: 21 points ∈ [0.85, 1.15]
- **IV interpolation**: PCHIP (scipy.interpolate.PchipInterpolator) as specified by paper
- **Positivity floor**: σ^G = max(1e-4, σ^B + h)
- **Evaluation surface**: Anchored final surface (paper-faithful choice)
- **Relative error formula**: mean(|model-market|/max(|market|, ε))
- **Gasoil options**: Used ONLY for ex-post validation, NOT calibration

## Error Summary (2026-02-02)

| Spec | RMSE (anchored) | MAE (anchored) | Rel Error |
|------|----------------|----------------|-----------|
| full | 3.7231 | 2.8503 | 0.2452 |
| brent_only | 2.4477 | 2.0333 | 0.2011 |
| fixed_spread | 2.4447 | 1.9861 | 0.1936 |
