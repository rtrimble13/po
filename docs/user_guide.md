# portopt User Guide

## Contents

1. [Installation](#installation)
2. [Mathematical background](#mathematical-background)
3. [Input data specification](#input-data-specification)
4. [Parameter files](#parameter-files)
5. [CLI reference](#cli-reference)
6. [Python API walkthrough](#python-api-walkthrough)
7. [Logging and diagnostics](#logging-and-diagnostics)
8. [Jupyter report](#jupyter-report)
9. [Numerical notes](#numerical-notes)

---

## 1. Installation

### Build requirements

| Tool | Minimum version |
|---|---|
| CMake | 3.20 |
| C++ compiler | GCC 10 / Clang 12 / MSVC 2019 (C++17) |
| Python (optional) | 3.8+ with pip |

All C++ dependencies are fetched automatically by CMake via `FetchContent` — no
manual installation of Eigen, spdlog, etc. is needed.

### Build steps

```bash
# Configure (Release build, all features)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --parallel

# Run tests
ctest --test-dir build --output-on-failure

# Install (optional)
cmake --install build --prefix /usr/local
```

To disable a component:
```bash
cmake -B build -DPORTOPT_BUILD_PYTHON=OFF   # no Python module
cmake -B build -DPORTOPT_BUILD_CLI=OFF      # no portopt binary
cmake -B build -DPORTOPT_BUILD_TESTS=OFF    # no test suite
```

### Python bindings

After building, register the module:
```bash
# Linux/macOS
export PYTHONPATH="$PWD/build/python:$PYTHONPATH"

# Windows (PowerShell)
$env:PYTHONPATH = "$PWD\build\python;$env:PYTHONPATH"

python -c "import portopt; print(portopt.__version__)"
```

Optional Python extras:
```bash
pip install matplotlib pandas numpy   # for helper functions + notebook
pip install jupyter nbconvert          # for report generation
```

---

## 2. Mathematical background

### 2.1 Mean-Variance Optimisation (MVO)

The Markowitz (1952) framework seeks a portfolio **w** ∈ ℝⁿ that minimises:

```
L(w, λ) = w'Σw − λ μ'w
```

subject to:
```
1'w = 1              (budget)
lb_i ≤ w_i ≤ ub_i   (weight bounds)
```

where:
- **Σ** (n×n) — asset covariance matrix (positive semi-definite)
- **μ** (n×1) — vector of expected excess returns
- **λ** ≥ 0 — risk aversion parameter

This is a **quadratic programme (QP)** with a simplex-plus-box feasible set.

**Efficient frontier**: sweeping λ from λ_min to λ_max (logarithmically) traces
the risk-return trade-off curve.

**Special cases**:
- λ → 0: minimum-variance portfolio (ignore returns)
- λ → ∞: maximum-return portfolio (ignore risk)
- λ = 1: unit risk aversion

### 2.2 QP Solver (FISTA with simplex projection)

The solver implements **FISTA** (Beck & Teboulle, 2009) — Nesterov-accelerated
projected gradient descent:

```
Initialise: w₀ = Proj(1/n · 1),  y = w₀,  t = 1
Repeat:
  g    = Q w_k + f                    // gradient (Q = 2Σ, f = -λμ)
  z    = y - (1/L) g                  // gradient step  (L = λ_max(Q))
  w_{k+1} = Proj(z)                   // project onto feasible set
  t_{k+1} = (1 + √(1 + 4t²)) / 2
  y    = w_{k+1} + ((t-1)/t_{k+1})(w_{k+1} - w_k)
Until ‖w_{k+1} − w_k‖ < ε
```

**Projection** onto {w : 1'w = 1, lb ≤ w ≤ ub} uses bisection on the dual
Lagrange multiplier (O(n log(1/ε))). Convergence rate: O(1/k²).

The Lipschitz constant **L** is estimated via power iteration.

### 2.3 Black-Litterman Model (Black & Litterman, 1992)

**Step 1 — Equilibrium prior**

Given market-cap weights **w_mkt** and risk aversion **δ**:
```
π = δ Σ w_mkt
```
π is the vector of implied equilibrium excess returns consistent with the CAPM.

**Step 2 — View specification**

k investor views expressed as:
```
P μ = q + ε,   ε ~ N(0, Ω)
```
where:
- **P** (k×n) — pick matrix (each row specifies the assets in a view)
- **q** (k×1) — expected view returns
- **Ω** = diag(ω₁, …, ωₖ) — view uncertainty (confidence inversely proportional to ωᵢ)

**Absolute view** (asset i returns r): Pᵢ = eᵢ (unit vector), qᵢ = r  
**Relative view** (asset i outperforms j by r): Pᵢ = eᵢ − eⱼ, qᵢ = r

**Step 3 — Posterior expected returns**

Using Bayes' theorem with τ-scaled prior covariance:
```
Σ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹
μ_BL = Σ_BL [(τΣ)⁻¹ π + P'Ω⁻¹ q]
```

**τ** (default 0.05): scales the uncertainty of the prior.
Small τ → prior close to equilibrium; large τ → views dominate.

**Step 4 — MVO with posterior**

MVO is run with:
- Expected returns: **μ_BL**
- Covariance: **Σ + Σ_BL** (includes estimation uncertainty)

#### Idzorek (2005) view confidence

Setting Ω directly as a variance is unintuitive. Set
`confidence_mode = ViewConfidenceMode::Idzorek` and pass `confidence ∈ [0, 1]`
on each view to interpret it as a percentage. portopt then derives Ω so that
the posterior tilt for each view k equals `c_k * d_k` where d_k is the
100%-confident tilt:

```
ω_k = (p_k' τΣ p_k) · (1/c_k − 1)
```

- `c_k = 0.5` → posterior moves half-way to the certain answer
- `c_k → 1.0` → posterior matches the view exactly
- `c_k → 0.0` → view ignored

### 2.4 Estimation from a returns time series

`portopt::estimation::fromReturns` (Python: `portopt.estimation.from_returns`,
CLI: `--returns`) converts a T×n returns matrix into a `MarketData` object:

| Estimator        | Use when                              |
|------------------|---------------------------------------|
| `none`           | T ≫ n and Σ is well-conditioned       |
| `linear`         | You want manual control over δ        |
| `ledoit-wolf`    | Default for n ≈ T or correlated assets|
| `oas`            | T < n (small-sample regime)           |

Shrinkage is recommended any time MVO produces concentrated corner solutions
on noisy inputs — typically halves the L2 distance from the true frontier.

### 2.5 Portfolio constraints

| Feature              | Configured by                                              |
|----------------------|-----------------------------------------------------------|
| Long-only / shorts   | `lower_bounds`, `upper_bounds`, `withShorts`              |
| Dollar-neutral L/S   | `PortfolioConstraints::dollarNeutral(n)` (budget = 0)     |
| Fix / forbid assets  | `fixWeight(i, w)` / `forbid(i)` helpers (translate to lb=ub)|
| Group caps (soft)    | `groups` vector; enforced via quadratic penalty           |
| Turnover penalty     | `current_weights` + `turnover_penalty` (κ in L2 form)     |
| Risk-free rate       | `MVOParameters::risk_free_rate` or `MarketData::risk_free_rate` |
| Benchmark            | `MarketData::benchmark_weights` (drives TE / IR / active share / β) |

Group constraints are *soft*: they're enforced by adding a quadratic penalty
`0.5 κ Σ max(0, viol)²` to the objective. Set `MVOParameters::group_penalty`
high (e.g. 1e6) for near-hard enforcement; the trade-off is solver stiffness.

### 2.6 PM-friendly portfolio helpers

| Method                              | Definition                                  |
|-------------------------------------|----------------------------------------------|
| `minVariancePortfolio`              | λ → 0 (global minimum variance)              |
| `maxSharpePortfolio`                | argmax_w (μ'w − r_f) / sqrt(w'Σw)            |
| `optimizeForTargetVolatility(σ_t)`  | Binary search over λ                         |
| `optimizeForTargetReturn(μ_t)`      | Binary search over λ                         |

All four respect the active constraints and turnover penalty.

### 2.7 Portfolio metrics

Each `OptimizationResult.metrics` exposes:

- `expected_return`, `volatility`, `variance`
- `sharpe_ratio = (μ_p − r_f) / σ_p`
- `risk_contribution[i] = w_i (Σw)_i / σ_p` (sums to σ_p)
- `diversification_ratio = (Σ |w_i| σ_i) / σ_p`
- `effective_n_assets = 1 / Σ w_i²`
- Benchmark-relative (NaN if no benchmark): `tracking_error`, `information_ratio`,
  `active_share`, `beta_to_benchmark`
- `turnover = ½‖w − w_prev‖₁` (NaN if no `current_weights`)

---

## 3. Input data specification

### JSON format

```json
{
  "assets": [
    {
      "ticker":          "AAPL",         // required — unique identifier
      "name":            "Apple Inc.",    // optional, defaults to ticker
      "expected_return": 0.152,           // annualised expected excess return
      "market_cap":      2.8e12           // optional, used for BL prior
    }
  ],
  "expected_returns": [0.152, 0.141],     // optional; overrides per-asset field
  "covariance": [                         // n×n positive semi-definite matrix
    [0.0529, 0.0224],
    [0.0224, 0.0441]
  ],
  "market_weights": [0.55, 0.45]          // optional; required for BL
}
```

### CSV format

Two files are required. Pass the assets CSV to the CLI; portopt
looks for `covariance.csv` in the same directory.

**assets.csv**
```
ticker,name,expected_return,market_cap
AAPL,Apple Inc.,0.152,2800000000000
MSFT,Microsoft Corp.,0.141,2500000000000
```

**covariance.csv** (optional ticker column in first position)
```
ticker,AAPL,MSFT
AAPL,0.0529,0.0224
MSFT,0.0224,0.0441
```

**weights.csv** (optional)
```
ticker,weight
AAPL,0.55
MSFT,0.45
```

---

## 4. Parameter files

Both JSON and TOML are supported. The schemas are identical in structure.

### MVO parameters

| Key | Type | Default | Description |
|---|---|---|---|
| `risk_aversion` | float | 1.0 | λ (risk aversion parameter) |
| `frontier_points` | int | 50 | Points on the efficient frontier |
| `min_risk_aversion` | float | 0.01 | Frontier sweep lower bound |
| `max_risk_aversion` | float | 100.0 | Frontier sweep upper bound |
| `constraints.lower_bounds` | float[] | [0,...,0] | Per-asset lower bounds |
| `constraints.upper_bounds` | float[] | [1,...,1] | Per-asset upper bounds |
| `constraints.allow_short_selling` | bool | false | Informational flag |

### Black-Litterman parameters

| Key | Type | Default | Description |
|---|---|---|---|
| `tau` | float | 0.05 | Prior uncertainty scaling |
| `risk_aversion` | float | 2.5 | Market risk aversion (δ) |
| `views[].description` | string | "" | Human-readable label |
| `views[].pick_vector` | float[] | — | Row of the P matrix |
| `views[].expected_return` | float | — | View expected return (q) |
| `views[].confidence` | float | 0.1 | View variance (ω_i) |

**Note**: `mvo` sub-section under `black_litterman` (or at root level) applies
to the MVO step after computing posterior returns.

---

## 5. CLI reference

```
portopt [global options] <subcommand> [options]

Global options:
  --log-level LEVEL    trace|debug|info|warn|error|off  (default: info)
  --log-file  FILE     Append logs to FILE (rotating, 5 MB, 3 files)

Subcommands:
  mvo          -d DATA [-p PARAMS] [-o OUT] [-f FORMAT] [--show-zero]
  frontier     -d DATA [-p PARAMS] [-o OUT] [-f FORMAT]
  bl           -d DATA [-p PARAMS] [-o OUT] [-f FORMAT] [--show-model]
  bl-frontier  -d DATA [-p PARAMS] [-o OUT] [-f FORMAT]
  report       -d DATA [-p PARAMS] [-n NOTEBOOK] [-o DIR] [-m METHOD]

Options:
  -d, --data     Market data file (.json or assets.csv)
  -p, --params   Parameter file (.json or .toml)
  -o, --output   Output file or directory
  -f, --format   console|json|csv  (auto-inferred from extension)
  --json-indent  JSON indent spaces (default: 2)
  --show-zero    Include near-zero weights in output (MVO/BL)
  --show-model   Print BL prior/posterior comparison (BL only)
  -m, --method   mvo|bl|both  (report only, default: both)
  -n, --notebook Jupyter template path (report only)
```

**Exit codes**: 0 on success, 1 on solver non-convergence or error.

---

## 6. Python API walkthrough

```python
import portopt
import numpy as np

# ── Setup ─────────────────────────────────────────────────────────────────────
portopt.init_logging(portopt.LogLevel.Info)
portopt.add_file_log("portopt.log")

# ── Load data ─────────────────────────────────────────────────────────────────
data = portopt.read_market_data("assets.json")
# Or from arrays:
n = 5
data = portopt.MarketData.from_arrays(
    assets=assets_list,
    expected_returns=mu_array,
    covariance=sigma_matrix,
    market_weights=mw_array,  # optional, needed for BL
)

# ── MVO ───────────────────────────────────────────────────────────────────────
params = portopt.MVOParameters()
params.risk_aversion = 3.0
params.constraints   = portopt.PortfolioConstraints.long_only(n)
params.constraints.upper_bounds[:] = 0.30   # 30% cap per asset

opt    = portopt.MVOptimizer(params)
result = opt.optimize(data)

# Access results
weights    = np.array(result.weights)     # numpy array
ret        = result.metrics.expected_return
vol        = result.metrics.volatility
sharpe     = result.metrics.sharpe_ratio

# Efficient frontier
frontier = opt.efficient_frontier(data)
df = frontier.to_dataframe()   # pandas DataFrame (requires pandas)

# Serialise
json_str = portopt.result_to_json(result)
csv_str  = portopt.frontier_to_json(frontier)

# ── Black-Litterman ───────────────────────────────────────────────────────────
bl_params = portopt.BlackLittermanParameters()
bl_params.tau = 0.05
bl_params.risk_aversion = 2.5

v = portopt.View()
v.description     = "Tech outperforms market by 4%"
v.pick_vector     = np.array([1.0, -0.5, -0.5, 0.0, 0.0])
v.expected_return = 0.04
v.confidence      = 0.0008
bl_params.views   = [v]

bl_params.mvo_params.constraints = portopt.PortfolioConstraints.long_only(n)

bl      = portopt.BlackLittermanOptimizer(bl_params)
bl_res  = bl.optimize(data)
model   = bl.model_output(data)  # inspect prior/posterior returns

# BL model diagnostics
for i, a in enumerate(data.assets):
    print(f"{a.ticker}: prior={model.prior_returns[i]:.4f}  "
          f"posterior={model.posterior_returns[i]:.4f}")
```

### Constraints

```python
# Long-only (default)
c = portopt.PortfolioConstraints.long_only(n)

# Allow short positions up to 20%
c = portopt.PortfolioConstraints.with_shorts(n, max_short=0.20)

# Custom per-asset bounds
c = portopt.PortfolioConstraints()
c.lower_bounds = np.array([0.05]*n)  # minimum 5% per asset
c.upper_bounds = np.array([0.35]*n)  # maximum 35% per asset
```

### Diagnostic helpers

```python
from portopt.helpers import (
    make_sample_data,
    plot_efficient_frontier,
    plot_weights,
    plot_bl_comparison,
    frontier_to_dataframe,
)

# Generate synthetic data for testing
data = make_sample_data(n_assets=8, seed=42)

# Plot
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
plot_efficient_frontier(frontier, ax=ax1)
plot_weights(result, ax=ax2)
plt.show()
```

---

## 7. Logging and diagnostics

The library uses **spdlog** for all internal logging.

### Log levels (in order of severity)

| Level | Usage |
|---|---|
| `Trace` | QP solver iteration details |
| `Debug` | Per-frontier-point metrics, matrix dimensions |
| `Info`  | High-level progress (default) |
| `Warn`  | Non-convergence, parameter overrides |
| `Error` | Unrecoverable errors |

### C++ setup

```cpp
#include <portopt/logging.hpp>

portopt::log::init(portopt::log::Level::Debug, true);       // console
portopt::log::addFileLog("portopt.log", portopt::log::Level::Trace); // file
```

### Python setup

```python
portopt.init_logging(portopt.LogLevel.Debug)
portopt.add_file_log("portopt.log", portopt.LogLevel.Trace)
```

---

## 8. Jupyter report

The diagnostic report is generated by `notebooks/generate_report.py` or via
`portopt report`. It:

1. Runs all optimisations and writes JSON intermediates to the output directory.
2. Executes `diagnostic_template.ipynb` via `jupyter nbconvert`.
3. Exports an HTML copy of the executed notebook.

**Sections of the generated report:**

1. MVO optimal portfolio weights (bar + pie)
2. Efficient frontier (risk-return curve, Sharpe vs. λ)
3. Portfolio weight heatmap along the frontier
4. BL prior vs. posterior expected returns
5. MVO vs. BL comparison table and weight differences
6. Asset risk contributions (marginal risk contribution)
7. Summary table

**Environment variable**: the notebook reads `PORTOPT_OUTPUT_DIR` to locate
the JSON intermediates. This is set automatically by `generate_report.py`.

---

## 9. Numerical notes

### Covariance matrix

- Must be **symmetric positive semi-definite** (PSD).
- Symmetry is checked; mild asymmetry (< 1e-6 × ‖Σ‖) is accepted.
- Near-singular matrices (highly correlated assets) are handled by the FISTA
  solver but may produce poorly-conditioned portfolios.
- Suggested preprocessing: factor correction, shrinkage (Ledoit-Wolf), or
  adding a ridge term ε·I before passing to the library.

### Convergence

- Default tolerance: `1e-9` (‖w_{k+1} − w_k‖₂)
- Default max iterations: 10,000
- FISTA converges at O(1/k²); typical problems converge in 200–2000 iterations.
- If `converged = false`: the result is still the best iterate found but may
  not satisfy KKT conditions exactly. Check `status_message`.

### Black-Litterman stability

- Views with `confidence → 0` (extremely confident) force the posterior close
  to the view return. Use realistic confidence values (0.0001–0.01).
- The `tau` parameter should be in (0, 0.1] for standard use.
- The posterior covariance LDLT factorisation can fail if views are collinear
  or the pick matrix is ill-conditioned; an exception is thrown in this case.
