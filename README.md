# portopt — Portfolio Optimisation Library

A high-performance C++ library for portfolio optimisation, with Python bindings, a cross-platform CLI, and a Jupyter diagnostic notebook.

## Features

| Feature | Detail |
|---|---|
| **Core algorithms** | MVO, Black-Litterman (variance- and Idzorek-mode confidences), minimum-CVaR (Rockafellar-Uryasev) |
| **PM-friendly portfolios** | Min-variance, max-Sharpe (with A3 analytical tangent fast path), target-volatility, target-return |
| **Convenience portfolios** | Equal-weight, inverse-variance, inverse-volatility, market-cap, equal-risk-contribution (risk parity), **hierarchical risk parity**, **maximum diversification**, **resampled / Michaud MVO** |
| **Constraints** | Box bounds, budget (dollar-neutral & 130/30), fix/forbid, group caps (soft + **hard via augmented Lagrangian**), L2 turnover, **tracking-error**, **gross-exposure / leverage**, **factor-neutral**, **beta-neutral**, **currency-exposure** |
| **Risk model** | Multi-factor `FactorRiskModel(B, Ω_F, D)` with Σ = BΩᶠB' + D reconstruction, per-factor risk decomposition, systematic vs specific split |
| **Transaction costs** | Linear (sign-iteration) and quadratic (Almgren-Chriss style) per-asset costs |
| **Multi-currency** | `Asset.currency`, currency-exposure aggregation, hedged-return conversion with [0,1] hedge ratio |
| **Estimation** | Sample / **exact Ledoit-Wolf** / OAS / linear / EWMA / Michaud-resampled |
| **Metrics** | Sharpe (rf-aware), Sortino, Calmar, Omega, max-drawdown, Ulcer, downside-deviation, risk contributions, diversification ratio, effective N, tracking error, IR, active share, beta, turnover, **realised CVaR** |
| **Backtesting** | `portopt.backtest.walk_forward()` — rolling-window estimation + periodic rebalance + trade list + per-rebalance turnover and transaction costs |
| **Attribution** | Brinson-Fachler and Brinson-Hood-Beebower (allocation / selection / interaction) |
| **Solver diagnostics** | Primal residual, KKT residual, dual estimate ν̂ — reported on every result |
| **Solver controls** | Cancellation token + `timeout_ms` (FISTA polls between iterations) |
| **Audit trail** | Library version + input/params FNV-1a hash stamped on every result (reproducibility) |
| **Black-Litterman diagnostics** | Pick-matrix rank / smallest singular value, posterior condition number |
| **Errors** | Typed hierarchy: `PortoptError`, `InvalidMarketData`, `InvalidParameters`, `InfeasibleProblem`, `SolverDidNotConverge`, `SolverCancelled`, `SolverTimeout`, each with a stable machine-readable `.code` |
| **MCP-ready** | Pydantic schemas (`portopt.schemas`) with JSON-Schema export, reference dispatcher (`portopt.mcp_server.PortoptToolServer`) exposing 10 tools |
| **Language** | C++17 library; Python bindings via pybind11 |
| **CLI** | Cross-platform binary (`po`); ASCII-fallback console for Windows |
| **Input formats** | JSON / CSV (market data); JSON / TOML (parameters); returns CSV; **JSON strings** for in-memory MCP use |
| **Output formats** | Console table, JSON (full / summary), CSV; notional $ exposure; "explain" mode for active bounds; pandas-free frontier records |
| **Logging** | spdlog-backed, configurable level + rotating file sink |
| **Diagnostics** | Jupyter notebook template with automatic report generation |
| **Tests** | Catch2 (C++) — 134 test cases, 530 assertions |

## Quick start

### Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `PORTOPT_BUILD_PYTHON` | `ON` | Build Python bindings |
| `PORTOPT_BUILD_CLI`    | `ON` | Build the `portopt` CLI |
| `PORTOPT_BUILD_TESTS`  | `ON` | Build the Catch2 test suite |

### Run tests

```bash
ctest --test-dir build --output-on-failure
```

### Install Python package

```bash
# After building, add the build directory to PYTHONPATH:
export PYTHONPATH="$PWD/build/python:$PYTHONPATH"
python -c "import portopt; print(portopt.__version__)"
```

## CLI usage

The CLI binary is `po`. (The Python module is still `import portopt` —
only the user-facing command is shortened.)

```
po <subcommand> [options]

Subcommands:
  mvo            Single MVO-optimal portfolio
  frontier       MVO efficient frontier
  bl             Black-Litterman optimal portfolio
  bl-frontier    Black-Litterman efficient frontier
  min-variance   Minimum-variance portfolio
  max-sharpe     Maximum-Sharpe (tangency) portfolio
  target-vol     Portfolio with a target volatility (e.g. --target 0.15)
  target-return  Portfolio with a target expected return
  report         Generate Jupyter diagnostic report

Common flags (most subcommands):
  --returns                 Treat -d as a returns time-series CSV
  --periods-per-year N      Annualisation factor (252 daily, 12 monthly)
  --shrinkage {none|linear|ledoit-wolf|oas}
                            Covariance shrinkage (with --returns)
  --shrinkage-delta D       Manual δ for --shrinkage=linear
  --risk-aversion λ         Override λ from params
  --risk-free-rate rf       Risk-free rate used in Sharpe
  --budget B                Sum-of-weights (1.0=fully invested, 0.0=long/short)
  --turnover-penalty κ      L2 penalty on |w − current_weights|²
  --total-capital $X        Print notional dollar exposure
  --explain                 Print active constraints
  --ascii                   Use ASCII separators in console output (Windows-friendly)
  --show-zero               Include near-zero weights in output
```

### Examples

```bash
# MVO — print to console
po mvo -d assets.json

# MVO — save to JSON with custom risk aversion from parameter file
po mvo -d assets.json -p params.toml -o result.json

# Efficient frontier to CSV
po frontier -d assets.json -p params.json -o frontier.csv

# Black-Litterman with model diagnostics
po bl -d assets.json -p params.toml --show-model -o bl_result.json

# PM-friendly: minimum-variance / max-Sharpe / 15% target vol
po min-variance -d assets.json
po max-sharpe   -d assets.json --risk-free-rate 0.04
po target-vol   -d assets.json --target 0.15

# Daily returns → MVO with Ledoit-Wolf shrinkage
po mvo -d daily_returns.csv --returns --shrinkage ledoit-wolf

# Show dollar notionals on a $10M book, with active constraints explained
po mvo -d assets.json --total-capital 10000000 --explain

# Windows console: use ASCII separators
po mvo -d assets.json --ascii

# Jupyter diagnostic report
po report -d assets.json -p params.toml -o reports/
```

### Walkthroughs

A runnable, walkthrough-style introduction to every CLI feature lives under
[examples/terminal/](examples/terminal/) — one shell script + Markdown
explainer per optimisation feature, shared dummy data, generated output
collected under `examples/terminal/var/`. Build the binary and run any
example:

```bash
cmake --build build --target portopt_cli
examples/terminal/scripts/01_mvo_basic.sh
```

Or run the whole suite as a smoke test with
[examples/terminal/scripts/run_all.sh](examples/terminal/scripts/run_all.sh).

## Python usage

```python
import portopt

portopt.init_logging(portopt.LogLevel.Info)

# Load data — from file…
data = portopt.read_market_data("assets.json")
# …or entirely inline (MCP-friendly):
data = portopt.market_data_from_dict({
    "assets": [{"ticker": "AAPL"}, {"ticker": "MSFT"}],
    "covariance": [[0.04, 0.01], [0.01, 0.09]],
    "risk_free_rate": 0.04,
})
import numpy as np
data.expected_returns = np.array([0.15, 0.12])

# MVO with budget, turnover penalty, and per-asset caps
params = portopt.MVOParameters()
params.risk_aversion = 2.5
params.risk_free_rate = 0.04
params.constraints = portopt.PortfolioConstraints.long_only(len(data.assets))
params.constraints.upper_bounds[:] = 0.35
params.constraints.current_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
params.constraints.turnover_penalty = 0.5

opt = portopt.MVOptimizer(params)

# PM-friendly portfolio constructors
mv      = opt.min_variance_portfolio(data)
ms      = opt.max_sharpe_portfolio(data)
tgt_vol = opt.optimize_for_target_volatility(data, 0.15)
tgt_ret = opt.optimize_for_target_return(data, 0.10)

result = opt.optimize(data)
print(f"Sharpe (rf-adjusted):    {result.metrics.sharpe_ratio:.3f}")
print(f"Tracking error vs B/M:   {result.metrics.tracking_error * 100:.2f}%")
print(f"Information ratio:       {result.metrics.information_ratio:.3f}")
print(f"Active share:            {result.metrics.active_share * 100:.2f}%")
print(f"Risk contributions:      {list(result.metrics.risk_contribution)}")

# Black-Litterman with Idzorek-style confidence
bl_params = portopt.BlackLittermanParameters()
bl_params.tau = 0.05
bl_params.risk_aversion = 2.5
bl_params.confidence_mode = portopt.ViewConfidenceMode.Idzorek
v = portopt.View()
v.pick_vector     = [1.0, -1.0, 0.0]
v.expected_return = 0.03
v.confidence      = 0.65   # 65% confident
bl_params.views   = [v]

bl     = portopt.BlackLittermanOptimizer(bl_params)
bl_res = bl.optimize(data)
model  = bl.model_output(data)

# Build MarketData from a returns time series with shrinkage
import numpy as np
R = np.random.randn(252, 5) * 0.01
data2 = portopt.estimation.from_returns(
    ["A", "B", "C", "D", "E"], R, periods_per_year=252,
    shrinkage="ledoit-wolf",
)

# EWMA / RiskMetrics covariance
S_ewma = portopt.estimation.ewma_covariance(R, 0.94, 252)

# Convenience portfolios — no optimisation required
w_eq  = portopt.portfolios.equal_weight(5)
w_iv  = portopt.portfolios.inverse_variance(S_ewma)
w_erc = portopt.portfolios.equal_risk_contribution(S_ewma)   # risk parity

# Diagnostics on the optimisation result
print(f"KKT residual:     {result.kkt_residual:.3e}")
print(f"Library version:  {result.library_version}")
print(f"Input  hash:      {result.input_hash}")
print(f"Params hash:      {result.params_hash}")

# Pandas-free frontier output (suitable for MCP responses)
frontier = opt.efficient_frontier(data)
records = frontier.to_records()       # list[dict] — no pandas import needed
```

See `examples/example_mvo.py`, `examples/example_bl.py`,
`examples/example_from_json.py`, and `examples/example_from_returns.py`
for complete walkthroughs.

## Input file formats

### Market data — JSON

```json
{
  "assets": [
    { "ticker": "AAPL", "name": "Apple",
      "expected_return": 0.15, "market_cap": 2.8e12, "sector": "Tech" }
  ],
  "covariance":        [[0.04]],
  "market_weights":    [1.0],
  "benchmark_weights": [1.0],
  "risk_free_rate":    0.04
}
```

### Market data — CSV

Two files are required:

**assets.csv**
```
ticker,name,expected_return,market_cap
AAPL,Apple Inc.,0.152,2800000000000
```

**covariance.csv** (with header row)
```
ticker,AAPL,MSFT
AAPL,0.0529,0.0224
MSFT,0.0224,0.0441
```

### Parameter file — JSON

```json
{
  "mvo": {
    "risk_aversion": 2.5,
    "frontier_points": 50,
    "risk_free_rate": 0.04,
    "constraints": {
      "lower_bounds": [0.0, 0.0],
      "upper_bounds": [0.4, 0.4],
      "budget": 1.0,
      "turnover_penalty": 0.0,
      "current_weights": [0.5, 0.5],
      "groups": [
        { "description": "Tech ≤ 30%",
          "coefficients": [1.0, 0.0],
          "lower": 0.0, "upper": 0.3 }
      ]
    }
  },
  "black_litterman": {
    "tau": 0.05,
    "risk_aversion": 2.5,
    "confidence_mode": "idzorek",
    "views": [
      {
        "description": "Asset A outperforms B",
        "pick_vector": [1.0, -1.0],
        "expected_return": 0.03,
        "confidence": 0.65
      }
    ]
  }
}
```

### Parameter file — TOML

```toml
[mvo]
risk_aversion   = 2.5
frontier_points = 50

[mvo.constraints]
lower_bounds = [0.0, 0.0]
upper_bounds = [0.4, 0.4]

[black_litterman]
tau           = 0.05
risk_aversion = 2.5

[[black_litterman.views]]
description     = "Asset A outperforms B"
pick_vector     = [1.0, -1.0]
expected_return = 0.03
confidence      = 0.001
```

## Diagnostic notebook

Generate a full diagnostic report with plots:

```bash
python notebooks/generate_report.py \
    --data   tests/data/assets.json \
    --params tests/data/params.json \
    --method both \
    --output reports/
```

Or via the CLI:

```bash
po report -d assets.json -p params.toml -o reports/
```

Produces: executed `.ipynb`, HTML report, and PNG figures in `reports/`.

## Algorithm details

See [docs/user_guide.md](docs/user_guide.md) for full mathematical derivations.

### MVO

Solves:
```
min  w'Σw − λ μ'w
s.t. 1'w = 1
     lb_i ≤ w_i ≤ ub_i
```

Using a Nesterov-accelerated projected gradient method (FISTA) with projection onto the generalised simplex in O(n log n) via bisection on the dual variable.

### Black-Litterman

1. **Prior**: π = δ Σ w_mkt (equilibrium returns)
2. **Views**: P μ ~ N(q, Ω)
3. **Posterior**: μ_BL = Σ_BL [(τΣ)⁻¹π + P'Ω⁻¹q]
4. **MVO on posterior**: same solver, posterior μ_BL and Σ + Σ_BL

## Dependencies (auto-fetched by CMake FetchContent)

| Library | Version | Purpose |
|---|---|---|
| Eigen | 3.4.0 | Linear algebra |
| nlohmann/json | 3.11.3 | JSON I/O |
| toml++ | 3.4.0 | TOML parameter files |
| spdlog | 1.14.1 | Structured logging |
| CLI11 | 2.4.2 | CLI argument parsing |
| pybind11 | 2.12.0 | Python bindings |
| Catch2 | 3.6.0 | Test framework |

## Project structure

```
portopt/
├── include/portopt/        C++ public headers
│   ├── types.hpp           Core data structures
│   ├── optimizer.hpp       Abstract optimizer interface
│   ├── qp_solver.hpp       QP solver (FISTA + simplex projection)
│   ├── mvo.hpp             Mean-Variance Optimizer
│   ├── black_litterman.hpp Black-Litterman Optimizer
│   ├── io/reader.hpp       Input readers (JSON/CSV/TOML)
│   ├── io/writer.hpp       Output writers (console/JSON/CSV)
│   └── logging.hpp         spdlog façade
├── src/                    C++ implementations
├── python/                 pybind11 bindings + Python package
│   └── portopt/
│       ├── __init__.py
│       └── helpers.py      Matplotlib/pandas utilities
├── cli/                    CLI frontend (main.cpp)
├── tests/                  Catch2 test suite
│   └── data/               JSON/CSV/TOML fixture files
├── notebooks/
│   ├── diagnostic_template.ipynb
│   └── generate_report.py
├── examples/
│   ├── example_mvo.py
│   └── example_bl.py
└── docs/
    ├── user_guide.md
    └── api_reference.md
```

## MCP integration

The library ships a reference tool dispatcher under
`portopt.mcp_server` that wraps every optimiser entry point as a stateless
JSON RPC. It requires `pydantic >= 2`.

```python
from portopt.mcp_server import PortoptToolServer

server = PortoptToolServer()

# Discovery — drives MCP "list_tools"
tools = server.list_tools()           # [{name, description, input_schema, output_schema}, ...]

# Call any tool by name; inputs are plain JSON-compatible dicts.
resp = server.call(
    "optimize_mvo",
    data   = {"assets": [...], "expected_returns": [...], "covariance": [[...]]},
    params = {"risk_aversion": 2.0, "constraints": {"lower_bounds": [...], "upper_bounds": [...]}}
)
# resp = {"data": {weights, metrics, ...}}  on success
# resp = {"error": {"type": "InfeasibleProblem", "code": "...", "message": "..."}}  on failure
```

JSON schemas for inputs and outputs are also available directly:

```python
from portopt.schemas import MVOParametersSchema, tool_manifest
schema = MVOParametersSchema.model_json_schema()
manifest = tool_manifest()
```

Wire the dispatcher to your preferred MCP framework (e.g. the official
`anthropic-ai/mcp` Python SDK) — `PortoptToolServer` is transport-agnostic.

## Advanced features (B-track)

```python
# B3 risk parity
w_erc = portopt.portfolios.equal_risk_contribution(cov)

# B4 hierarchical risk parity
w_hrp = portopt.portfolios.hierarchical_risk_parity(cov)

# B5 minimum-CVaR
w_cvar = portopt.portfolios.minimum_cvar(returns, alpha=0.95)

# B7 factor risk model + B8 factor-neutral constraint
fm = portopt.FactorRiskModel()
fm.loadings = B                     # n × k
fm.factor_covariance = Omega_F      # k × k
fm.specific_variance = D            # length n
decomp = portopt.decompose_risk(fm, weights)
g = portopt.factor_neutral_constraint(fm, factor_index=1, lower=-0.02, upper=0.02)

# B9 walk-forward backtester (Python)
res = portopt.backtest.walk_forward(
    returns=R, window=126, step=21,
    build_weights=my_strategy,
    transaction_cost=0.0005,
    benchmark_returns=B)
print(res.summary(rf=0.04))

# B10 advanced analytics
stats = portopt.analytics.summarise(res.portfolio_returns)

# B11 attribution
attr = portopt.attribution.brinson_fachler(
    group_weights_p, group_weights_b,
    group_returns_p, group_returns_b,
    group_names=["Tech","Health","Energy"])

# B13 multi-currency
exp = portopt.fx.currency_exposure(assets, weights)
mu_base = portopt.fx.convert_expected_returns(
    assets, mu_local, {"USD": 0.0, "EUR": 0.03}, hedge_ratio=0.5)
```

## Solver controls

```python
# Cancellation + timeout (C5)
params = portopt.MVOParameters()
params.timeout_ms = 5000
tok = portopt.CancellationToken()
params.cancellation = tok
# ... from another thread: tok.cancel()
```

Typed exceptions (C4) carry a stable code:

```python
try:
    opt.optimize(data)
except portopt.InfeasibleProblem as e:
    print(e)            # "[budget_outside_bounds] Infeasible: budget=..."
except portopt.SolverTimeout:
    ...
```

## License

MIT
