# AGENTS.md — Repository Knowledge Base

## Project: Brent Local-Volatility Calibration via Gaussian-Mixture Marginals

### Overview
This repository implements three interconnected experiments for a Brent-Gasoil joint pricing framework:

1. **Exp1**: Brent local-vol calibration via Gaussian-mixture terminal marginals (Bachelier framework)
2. **Exp2**: Historical crack-spread k-medians clustering to estimate Gasoil volatility correction h
3. **Exp3**: Joint Brent-Gasoil Monte Carlo pricing with three model specifications + benchmark comparison

### Repository Structure
```
repo/
├── data/           # Data loading from Massive API + synthetic fallback
├── src/            # Core implementation modules
│   ├── bachelier.py        # Bachelier option pricing & IV inversion (uses pyfeng.Norm)
│   ├── gaussian_mixture.py # Gaussian-mixture calibration for Brent terminal marginals
│   ├── local_vol.py        # Local-volatility reconstruction (Proposition 2.1)
│   ├── k_medians.py        # Custom k-medians with cityblock/L1 distance
│   ├── crack_spread.py     # Historical h function estimation from futures data
│   ├── joint_model.py      # Joint Brent-Gasoil Euler MC simulation
│   └── utils.py            # Utility functions (roll adjustment, interpolation, etc.)
├── exp/            # Experiment scripts
│   ├── exp1_brent_localvol.py
│   ├── exp2_crack_spread_clustering.py
│   └── exp3_joint_pricing.py
├── tests/          # pytest tests
├── results/        # Output artifacts + RESULTS.md
└── README.md
```

### Key Library Choices (verified via Context7)
- **pyfeng.Norm**: Bachelier option pricing and Choi2009 IV inversion — USE THIS
- **scipy.optimize.isotonic_regression**: PAVA monotone projection for cumulative variances
- **scipy.optimize.least_squares**: Calibration objective minimization (TRF method)
- **scipy.interpolate.PchipInterpolator**: Smile interpolation onto common moneyness grid
- **sklearn.isotonic.IsotonicRegression**: Alternative for cumulative variance monotone projection
- Custom k-medians with cityblock distance (no standard library equivalent)
- Custom Gaussian-mixture local-vol reconstruction (Proposition 2.1)

### Model Parameters (paper specifications)
- Number of tenors: M = 11
- Number of mixture components N: tested in {2,3,4,5}, default N=3
- Number of clusters k = 6 (k-medians)
- Correlation estimation window: q = 90 days
- Interest rate: r = 0
- Rolling window n: default 30 business days (documented assumption)
- Downsample interval n2 ≈ n/2 = 15

### Data Notes
- Massive API used for ICE Brent/Gasoil futures data
- Ticker: "BZ=F" style futures or equivalent
- Synthetic data used as fallback when real data unavailable
- Validation dates: 2026-02-02, 2026-04-21, 2026-05-26

### Mathematical Framework
- Bachelier/normal model: C = (F-K)Φ(d) + σ_N√T φ(d), d = (F-K)/(σ_N√T)
- Gaussian mixture: p_T(y) = Σᵢ λᵢ N(y; μᵢ, σᵢ²T)
- Local vol (Proposition 2.1): σ_B²(t,y) = Σᵢ wᵢ(t,y) σᵢ²(t), with μᵢ=0 assumption
- Crack spread: S(t,T_m) = G_m(t) - B_m(t)
- h function: linear interpolation over k-medians cluster centroids

### Assumption Documentation
- μᵢ(t) = 0 for all i,t (martingale constraint, simplifies Proposition 2.1)
- N = 3 mixture components (smallest N delivering stable calibration)
- Rolling window n = 30 business days
- n2 = 15 (round(n/2))
- Flat extrapolation of h outside centroid support
- 100k Monte Carlo paths with antithetic variates
- Time grid: min(252, 50*M) steps per maturity

### Testing
- Run tests: `pytest tests/ -v`
- Run experiments: `python exp/exp1_brent_localvol.py`, etc.
