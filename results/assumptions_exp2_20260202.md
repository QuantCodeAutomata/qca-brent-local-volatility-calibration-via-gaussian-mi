# Exp 2 Assumptions — 2026-02-02

## Implementation Assumptions (paper omissions resolved)

- **Rolling window n**: 30 business days
  Tested n ∈ {20, 30, 60, 90}; n=30 chosen as baseline.

- **Downsample interval n2**: 15 (= round(30/2))

- **History start date**: 300 calendar days before valuation date.

- **Roll rule**: Select contract with expiry closest to target tenor month.
  Roll 2 business days before expiry (paper example).

- **k-medians**: k=6, n_init=50 random restarts, cityblock distance.
  Custom implementation (no library equivalent confirmed via Context7).

- **Extrapolation**: Flat beyond min/max centroid crack-spread nodes.

- **Duplicate crack nodes**: Average of corresponding vol-spread values.

- **Correlation**: Historical q=90-day arithmetic-return correlations.

## Results Summary

- Feature vectors: 6 (after downsampling)
- k-medians inertia: 0.0000
- Valuation-date h^fix range: [0.0392, 0.1744]
- Correlation range: [0.848, 0.872]
