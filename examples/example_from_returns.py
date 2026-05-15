#!/usr/bin/env python3
"""
example_from_returns.py — End-to-end pipeline from a daily-returns CSV.

Demonstrates:
  - Loading a returns time series and estimating μ, Σ via Ledoit-Wolf shrinkage
  - Comparing the resulting frontier with the plain-sample-covariance version
  - Min-variance / max-Sharpe / target-vol shortcuts
  - Risk contributions and benchmark-relative metrics
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import portopt
    import numpy as np
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}. See README for build instructions.")


# ── 1. Generate synthetic daily returns and write to CSV ─────────────────────
TICKERS = ["US_EQ", "EU_EQ", "EM_EQ", "GOV_BD", "CMDTY"]
TRUE_VOLS = np.array([0.16, 0.20, 0.24, 0.07, 0.18])
CORR = np.array([
    [1.00, 0.65, 0.55, -0.10, 0.20],
    [0.65, 1.00, 0.60, -0.05, 0.18],
    [0.55, 0.60, 1.00,  0.00, 0.25],
    [-0.10, -0.05, 0.00, 1.00, -0.05],
    [0.20, 0.18, 0.25, -0.05, 1.00],
])

rng = np.random.default_rng(7)
TRUE_COV = (TRUE_VOLS[:, None] * CORR * TRUE_VOLS[None, :]) / 252.0
TRUE_MU  = np.array([0.09, 0.075, 0.085, 0.035, 0.05]) / 252.0
T = 750  # ~ 3 years of daily

L = np.linalg.cholesky(TRUE_COV)
R = TRUE_MU + rng.standard_normal((T, len(TICKERS))) @ L.T

csv_path = Path("returns_sample.csv")
with csv_path.open("w") as f:
    f.write("date," + ",".join(TICKERS) + "\n")
    for t in range(T):
        f.write(f"{t}," + ",".join(f"{x:.8f}" for x in R[t]) + "\n")
print(f"Wrote synthetic returns to {csv_path} (T={T}, n={len(TICKERS)})")

# ── 2. Load with three different shrinkage estimators ────────────────────────
portopt.init_logging(portopt.LogLevel.Warn)

estimators = ["none", "ledoit-wolf", "oas"]
print("\n  Estimator      σ̂ trace    diag(Σ̂)/diag(Σ_true)")
print("  " + "-" * 55)
for est in estimators:
    data = portopt.read_returns_csv(str(csv_path), periods_per_year=252.0,
                                     shrinkage=est)
    diag_ratio = np.array(data.covariance).diagonal() / TRUE_COV.diagonal() / 252.0
    print(f"  {est:14s} {np.trace(data.covariance):8.4f}  "
          f"min={diag_ratio.min():.3f} max={diag_ratio.max():.3f}")

# ── 3. Use Ledoit-Wolf for the rest of the analysis ──────────────────────────
data = portopt.read_returns_csv(str(csv_path), periods_per_year=252.0,
                                 shrinkage="ledoit-wolf")
n = len(data.assets)

# Attach benchmark = equal-weighted "market"
data.benchmark_weights = np.ones(n) / n
data.risk_free_rate    = 0.04

# ── 4. Build MVO with a 35% per-asset cap and 1% transaction cost ────────────
params = portopt.MVOParameters()
params.risk_aversion = 3.0
params.frontier_points = 30
params.constraints = portopt.PortfolioConstraints.long_only(n)
params.constraints.upper_bounds[:] = 0.35

# Suppose we already hold 20% in each — penalise reshaping
params.constraints.current_weights = np.full(n, 1.0 / n)
params.constraints.turnover_penalty = 0.5

opt = portopt.MVOptimizer(params)

# ── 5. Three PM-friendly portfolios ──────────────────────────────────────────
mv     = opt.min_variance_portfolio(data)
ms     = opt.max_sharpe_portfolio(data)
tgt15  = opt.optimize_for_target_volatility(data, 0.15)

def summarise(label, r):
    print(f"\n  {label}")
    print(f"    return     : {r.metrics.expected_return * 100:6.2f} %")
    print(f"    volatility : {r.metrics.volatility * 100:6.2f} %")
    print(f"    sharpe     : {r.metrics.sharpe_ratio:6.3f}")
    print(f"    turnover   : {r.metrics.turnover * 100:6.2f} % (one-way)")
    print(f"    track err  : {r.metrics.tracking_error * 100:6.2f} %")
    print(f"    info ratio : {r.metrics.information_ratio:6.3f}")
    print(f"    weights    : ",
          ", ".join(f"{a.ticker}={w*100:5.1f}%" for a, w in zip(r.assets, r.weights)))

summarise("Minimum variance", mv)
summarise("Max Sharpe",       ms)
summarise("Target 15% vol",   tgt15)

# ── 6. Frontier + plots ──────────────────────────────────────────────────────
frontier = opt.efficient_frontier(data)

try:
    import matplotlib.pyplot as plt
    from portopt.helpers import (plot_efficient_frontier,
                                  plot_weights, plot_risk_contributions)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    plot_efficient_frontier(frontier, ax=ax1,
                             risk_free_rate=data.risk_free_rate)
    ax1.scatter([ms.metrics.volatility * 100],
                [ms.metrics.expected_return * 100],
                marker="o", s=120, color="red", label="Max-Sharpe (helper)")
    ax1.legend()

    plot_weights(ms, ax=fig.add_subplot(gs[1, 0]))
    plot_risk_contributions(ms, ax=fig.add_subplot(gs[1, 1]))

    plt.suptitle("Returns-based MVO with Ledoit-Wolf shrinkage",
                 fontsize=13, fontweight="bold")
    plt.savefig("example_from_returns_output.png", dpi=120, bbox_inches="tight")
    print("\nPlot saved to example_from_returns_output.png")
except ImportError:
    print("\n(matplotlib not available — skipping plots)")
