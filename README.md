# Brent Local-Volatility Calibration via Gaussian-Mixture Marginals

A Python implementation of a joint Brent/Gasoil local-volatility pricing
framework based on a research paper studying energy-option surface
reconstruction under the extended correlated Bachelier model.

---

## Overview

The repository implements three self-contained experiments:

| Experiment | Description |
|-----------|-------------|
| **Exp 1** | Calibrate Brent option smiles across 11 tenors using maturity-wise Gaussian-mixture marginals; reconstruct σᴮ(t,y) via Proposition 2.1 |
| **Exp 2** | Estimate a tenor-specific Gasoil volatility correction function h from historical crack-spread data using cityblock-distance k-medians clustering (k=6) |
| **Exp 3** | Simulate a joint correlated Bachelier model, reconstruct the Gasoil IV surface under three specifications, and benchmark against market Gasoil IVs |

---

## Repository Structure

```
.
├── data/
│   └── download_data.py          # Synthetic Brent/Gasoil data generator + Massive API loader
├── exp/
│   ├── exp_1_brent_calibration.py   # Exp 1 orchestrator
│   ├── exp_2_crack_spread_h.py      # Exp 2 orchestrator
│   └── exp_3_joint_pricing.py       # Exp 3 orchestrator
├── src/
│   ├── bachelier.py              # Bachelier pricing + IV inversion
│   ├── gaussian_mixture.py       # Gaussian-mixture calibration, Proposition 2.1 helpers
│   ├── local_vol.py              # BrentLocalVolModel: σᴮ(t,y)
│   ├── crack_spread.py           # Rolling feature engineering, h interpolation, ρ estimation
│   ├── k_medians.py              # Custom cityblock-distance k-medians (paper requirement)
│   ├── joint_model.py            # Joint MC simulation, MC pricing, anchoring, metrics
│   └── utils.py                  # Shared plotting, file I/O, RESULTS_DIR
├── tests/
│   ├── test_bachelier.py
│   ├── test_gaussian_mixture.py
│   ├── test_local_vol.py
│   ├── test_crack_spread.py
│   ├── test_k_medians.py
│   ├── test_joint_model.py
│   └── test_integration.py       # End-to-end pipeline tests (module-scoped fixtures)
├── results/
│   ├── RESULTS.md                # Metrics, tables, discussion
│   └── *.csv / *.parquet / *.png # Per-date artifacts
├── AGENTS.md                     # Persistent agent memory / implementation notes
└── README.md
```

---

## Installation

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn \
            arch hmmlearn ruptures skfolio pyportfolioopt \
            quantstats vectorbt massive
```

> The `massive` package provides authenticated access to the Polygon.io REST
> API for real market data. Set `MASSIVE_TOKEN` in your environment.

---

## Usage

### Run all three experiments on synthetic data

```python
from data.download_data import build_synthetic_brent_gasoil_data
from exp.exp_1_brent_calibration import run_brent_calibration
from exp.exp_2_crack_spread_h import run_h_estimation
from exp.exp_3_joint_pricing import run_joint_pricing

data = build_synthetic_brent_gasoil_data(
    valuation_dates=["2026-02-02"],
    M=11,
    history_days=250,
    seed=42,
)

exp1 = run_brent_calibration("2026-02-02", data)
exp2 = run_h_estimation("2026-02-02", data)
exp3 = run_joint_pricing("2026-02-02", data, exp1, exp2)

print(exp3["error_df"])
```

### Run with real market data (requires `MASSIVE_TOKEN`)

```python
import os
from data.download_data import load_real_data
from exp.exp_1_brent_calibration import run_brent_calibration

data = load_real_data(
    valuation_date="2026-02-02",
    api_key=os.getenv("MASSIVE_TOKEN"),
)
result = run_brent_calibration("2026-02-02", data)
```

### Run the test suite

```bash
pytest tests/ -v
```

---

## Key Design Decisions

### Gaussian-Mixture Marginals (Exp 1)

- **N = 3 components** across all 11 tenors — smallest N yielding stable fits.
- Parameter transforms: `softmax(a)` → weights; `exp(v)` → variances; last
  mean backed out from the futures martingale constraint.
- **Isotonic regression** (equal weights) enforces monotone cumulative
  variances before computing piecewise-constant bucket vols γᵢ,ₘ.
- Component drifts μᵢ(t) = 0 ∀i,t — simplifies Proposition 2.1 to a
  density-weighted average of component variances.

### Cityblock k-Medians (Exp 2)

- **Custom implementation** — scikit-learn KMeans uses Euclidean distance;
  paper specifies cityblock (L1) distance with coordinate-wise median update.
- 50 random restarts; best total L1 inertia retained.
- Flat extrapolation beyond centroid support for h_m(s).

### Joint Simulation (Exp 3)

- Correlated Brownian increments:
  `ΔW^G = ρ ΔW^B + √(1−ρ²) ΔZ`, with ρ estimated per tenor over q=90 days.
- Positivity floor: `σᴳ = max(1e-4, σᴮ + h)`.
- **Anchored surface**: `σ_final^G = σ_mkt^B + (σ_model^G − σ_model^B)` —
  the essential step that ties the Gasoil surface to the liquid Brent market.
- PCHIP interpolation onto a common moneyness grid [0.85, 1.15].

---

## Results Summary

See [`results/RESULTS.md`](results/RESULTS.md) for full tables and plots.

| Spec (anchored, 2026-02-02) | RMSE (IV pts) | MAE (IV pts) |
|-----------------------------|--------------|--------------|
| full dynamic model | 3.72 | 2.85 |
| brent_only (h=0) | 2.45 | 2.03 |
| fixed_spread | **2.44** | **1.99** |

The fixed-spread benchmark slightly outperforms on this synthetic date,
consistent with the paper's reported nuance for 2026-04-21. The anchoring
step reduces errors by ~60–70% compared to raw model surfaces across all specs.
