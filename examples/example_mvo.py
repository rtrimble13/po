#!/usr/bin/env python3
"""
example_mvo.py — Mean-Variance Optimisation walkthrough.

Demonstrates:
  - Loading market data from JSON
  - Building constraints
  - Computing a single optimal portfolio
  - Computing and plotting the efficient frontier
  - Writing results to JSON and CSV
"""

import numpy as np
import sys
import os

try:
    import portopt
except ImportError:
    sys.exit(
        "portopt not found. Build with CMake (PORTOPT_BUILD_PYTHON=ON) "
        "and add the build directory to PYTHONPATH."
    )

# ── 1. Initialise logging ─────────────────────────────────────────────────────
portopt.init_logging(portopt.LogLevel.Info)

# ── 2. Build market data (can also load from file) ────────────────────────────
# You can do:  data = portopt.read_market_data("tests/data/assets.json")

from portopt.helpers import make_sample_data
data = make_sample_data(n_assets=6, seed=0)
print(f"Universe: {[a.ticker for a in data.assets]}")

# ── 3. Configure MVO ──────────────────────────────────────────────────────────
n = len(data.assets)
params = portopt.MVOParameters()
params.risk_aversion = 2.5
params.frontier_points = 60
params.min_risk_aversion = 0.05
params.max_risk_aversion = 200.0
params.constraints = portopt.PortfolioConstraints.long_only(n)

# Optional: per-asset weight caps
params.constraints.upper_bounds[:] = 0.35

# ── 4. Single optimal portfolio ───────────────────────────────────────────────
opt = portopt.MVOptimizer(params)
result = opt.optimize(data)

print(f"\nOptimal Portfolio (λ={params.risk_aversion})")
print(f"  Converged  : {result.converged}")
print(f"  Return     : {result.metrics.expected_return*100:.2f}%")
print(f"  Volatility : {result.metrics.volatility*100:.2f}%")
print(f"  Sharpe     : {result.metrics.sharpe_ratio:.3f}")
print(f"\nWeights:")
for a, w in zip(result.assets, result.weights):
    if abs(w) > 1e-4:
        print(f"  {a.ticker:12s} {w:.4f}  ({w*100:.1f}%)")

# ── 5. Efficient frontier ─────────────────────────────────────────────────────
frontier = opt.efficient_frontier(data)
print(f"\nEfficient frontier: {len(frontier.points)} points")

# Find max-Sharpe portfolio
best = max(frontier.points, key=lambda p: p.metrics.sharpe_ratio)
print(f"Max-Sharpe point: λ={best.risk_aversion:.2f}  "
      f"ret={best.metrics.expected_return*100:.2f}%  "
      f"vol={best.metrics.volatility*100:.2f}%  "
      f"Sharpe={best.metrics.sharpe_ratio:.3f}")

# ── 6. Write results ──────────────────────────────────────────────────────────
json_out = portopt.result_to_json(result)
print(f"\nJSON output (first 300 chars):\n{json_out[:300]}...")

csv_out = portopt.frontier_to_json(frontier)
print(f"\nFrontier JSON (first 200 chars):\n{csv_out[:200]}...")

# ── 7. Plot (requires matplotlib) ────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    from portopt.helpers import plot_efficient_frontier, plot_weights

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_efficient_frontier(frontier, ax=axes[0])
    plot_weights(result, ax=axes[1])
    plt.tight_layout()
    plt.savefig("example_mvo_output.png", dpi=120)
    print("\nPlot saved to example_mvo_output.png")
    plt.show()
except ImportError:
    print("\n(matplotlib not available — skipping plots)")

# ── 8. pandas DataFrame of frontier ──────────────────────────────────────────
try:
    df = frontier.to_dataframe()
    print(f"\nFrontier DataFrame:\n{df.head()}")
except ImportError:
    print("\n(pandas not available — skipping DataFrame)")
