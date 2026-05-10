# portopt — Portfolio Optimisation Library

A high-performance C++ library for portfolio optimisation, with Python bindings, a cross-platform CLI, and a Jupyter diagnostic notebook.

## Features

| Feature | Detail |
|---|---|
| **Algorithms** | Mean-Variance Optimisation (MVO) and Black-Litterman |
| **Language** | C++17 library; Python bindings via pybind11 |
| **CLI** | Cross-platform binary for Linux and Windows |
| **Input formats** | JSON, CSV (market data); JSON, TOML (parameters) |
| **Output formats** | Console (formatted table), JSON, CSV |
| **Logging** | spdlog-backed, configurable level + rotating file sink |
| **Diagnostics** | Jupyter notebook template with automatic report generation |
| **Tests** | Catch2 test suite (QP solver, MVO, BL, IO, integration) |

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

```
portopt <subcommand> [options]

Subcommands:
  mvo          Single MVO-optimal portfolio
  frontier     MVO efficient frontier
  bl           Black-Litterman optimal portfolio
  bl-frontier  Black-Litterman efficient frontier
  report       Generate Jupyter diagnostic report
```

### Examples

```bash
# MVO — print to console
portopt mvo -d assets.json

# MVO — save to JSON with custom risk aversion from parameter file
portopt mvo -d assets.json -p params.toml -o result.json

# Efficient frontier to CSV
portopt frontier -d assets.json -p params.json -o frontier.csv

# Black-Litterman with model diagnostics
portopt bl -d assets.json -p params.toml --show-model -o bl_result.json

# BL efficient frontier
portopt bl-frontier -d assets.json -p params.toml -o bl_frontier.csv

# Jupyter diagnostic report
portopt report -d assets.json -p params.toml -o reports/

# Control log level
portopt mvo -d assets.json --log-level debug --log-file portopt.log
```

## Python usage

```python
import portopt

portopt.init_logging(portopt.LogLevel.Info)

# Load data
data = portopt.read_market_data("assets.json")

# MVO
params = portopt.MVOParameters()
params.risk_aversion = 2.5
params.constraints   = portopt.PortfolioConstraints.long_only(len(data.assets))

opt      = portopt.MVOptimizer(params)
result   = opt.optimize(data)
frontier = opt.efficient_frontier(data)

print(f"Sharpe: {result.metrics.sharpe_ratio:.3f}")
df = frontier.to_dataframe()  # pandas DataFrame

# Black-Litterman
bl_params = portopt.BlackLittermanParameters()
bl_params.tau = 0.05
v = portopt.View()
v.pick_vector     = [1.0, -1.0, 0.0]
v.expected_return = 0.03
v.confidence      = 0.001
bl_params.views   = [v]

bl     = portopt.BlackLittermanOptimizer(bl_params)
bl_res = bl.optimize(data)
model  = bl.model_output(data)   # inspect prior vs. posterior returns
```

See `examples/example_mvo.py` and `examples/example_bl.py` for complete walkthroughs.

## Input file formats

### Market data — JSON

```json
{
  "assets": [
    { "ticker": "AAPL", "name": "Apple", "expected_return": 0.15, "market_cap": 2.8e12 }
  ],
  "covariance": [[0.04]],
  "market_weights": [1.0]
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
    "constraints": {
      "lower_bounds": [0.0, 0.0],
      "upper_bounds": [0.4, 0.4]
    }
  },
  "black_litterman": {
    "tau": 0.05,
    "risk_aversion": 2.5,
    "views": [
      {
        "description": "Asset A outperforms B",
        "pick_vector": [1.0, -1.0],
        "expected_return": 0.03,
        "confidence": 0.001
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
portopt report -d assets.json -p params.toml -o reports/
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

## License

MIT
