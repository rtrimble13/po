#!/usr/bin/env python3
"""
example_bl.py — Black-Litterman optimisation walkthrough.

Demonstrates:
  - Setting up equilibrium prior from market-cap weights
  - Specifying investor views with varying confidence levels
  - Comparing BL posterior returns to the equilibrium prior
  - Computing optimal BL portfolio and efficient frontier
  - Diagnostic model output
"""

import numpy as np
import sys

try:
    import portopt
except ImportError:
    sys.exit("portopt not found. See README for build instructions.")

# ── 1. Logging ────────────────────────────────────────────────────────────────
portopt.init_logging(portopt.LogLevel.Info)

# ── 2. Market data ────────────────────────────────────────────────────────────
# Classic 5-asset example used in BL literature
assets = []
for ticker, name, er, mc in [
    ("US_EQ",  "US Equities",        0.110, 0.35),
    ("EU_EQ",  "European Equities",  0.085, 0.20),
    ("EM_EQ",  "Emerging Markets",   0.095, 0.15),
    ("GOV_BD", "Government Bonds",   0.040, 0.20),
    ("CMDTY",  "Commodities",        0.060, 0.10),
]:
    a = portopt.Asset()
    a.ticker = ticker
    a.name   = name
    a.expected_return = er
    a.market_cap = mc * 1e13
    assets.append(a)

n = len(assets)

# Covariance matrix (annualised)
cov = np.array([
    [0.0400,  0.0180,  0.0220, -0.0050,  0.0100],
    [0.0180,  0.0361,  0.0200, -0.0040,  0.0080],
    [0.0220,  0.0200,  0.0529, -0.0030,  0.0120],
    [-0.0050,-0.0040, -0.0030,  0.0100, -0.0020],
    [0.0100,  0.0080,  0.0120, -0.0020,  0.0225],
])

er = np.array([a.expected_return for a in assets])
# Market-cap weights (must sum to 1)
mw = np.array([0.35, 0.20, 0.15, 0.20, 0.10])

data = portopt.MarketData.from_arrays(
    assets=assets,
    expected_returns=er,
    covariance=cov,
    market_weights=mw,
)

print(f"Universe: {[a.ticker for a in assets]}")
print(f"Market weights: {mw}")

# ── 3. Build Black-Litterman parameters ──────────────────────────────────────
bl_params = portopt.BlackLittermanParameters()
bl_params.tau           = 0.05   # uncertainty scaling of prior
bl_params.risk_aversion = 2.5    # market risk aversion (δ)

# Idzorek mode: confidence ∈ [0, 1] is a percentage rather than a variance.
# This is more intuitive for portfolio managers — 0.5 = "50% confident".
bl_params.confidence_mode = portopt.ViewConfidenceMode.Idzorek

# View 1: US Equities will outperform European Equities by 3%
v1 = portopt.View()
v1.description     = "US Equities outperform European Equities by 3%"
v1.pick_vector     = np.array([1.0, -1.0, 0.0, 0.0, 0.0])
v1.expected_return = 0.03
v1.confidence      = 0.65   # 65% confident in this relative view

# View 2: Bonds will return 4.5% (absolute view)
v2 = portopt.View()
v2.description     = "Government Bonds absolute return 4.5%"
v2.pick_vector     = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
v2.expected_return = 0.045
v2.confidence      = 0.75   # 75% confident

bl_params.views = [v1, v2]

# MVO parameters for the posterior optimisation
bl_params.mvo_params.risk_aversion = 2.5
bl_params.mvo_params.constraints   = portopt.PortfolioConstraints.long_only(n)
bl_params.mvo_params.frontier_points = 40

# ── 4. Inspect BL model outputs ───────────────────────────────────────────────
bl = portopt.BlackLittermanOptimizer(bl_params)
model = bl.model_output(data)

print("\nBlack-Litterman Model:")
print(f"  {'Ticker':10s}  {'Prior (%)':>12s}  {'Posterior (%)':>13s}  {'Delta (%)':>10s}")
print("  " + "-"*50)
for i, a in enumerate(assets):
    pr = model.prior_returns[i] * 100
    po = model.posterior_returns[i] * 100
    print(f"  {a.ticker:10s}  {pr:12.3f}  {po:13.3f}  {po-pr:+10.3f}")

# ── 5. Optimise ───────────────────────────────────────────────────────────────
result   = bl.optimize(data)
frontier = bl.efficient_frontier(data)

print(f"\nBL Optimal Portfolio")
print(f"  Return     : {result.metrics.expected_return*100:.2f}%")
print(f"  Volatility : {result.metrics.volatility*100:.2f}%")
print(f"  Sharpe     : {result.metrics.sharpe_ratio:.3f}")
print(f"\nWeights:")
for a, w in zip(result.assets, result.weights):
    print(f"  {a.ticker:12s} {w:.4f}  ({w*100:.1f}%)")

# ── 6. Compare to plain MVO ───────────────────────────────────────────────────
mvo_params = portopt.MVOParameters()
mvo_params.risk_aversion = 2.5
mvo_params.constraints   = portopt.PortfolioConstraints.long_only(n)
mvo_params.frontier_points = 40

mvo = portopt.MVOptimizer(mvo_params)
mvo_result   = mvo.optimize(data)
mvo_frontier = mvo.efficient_frontier(data)

print(f"\nMVO Optimal Portfolio (same λ, no views)")
print(f"  Return     : {mvo_result.metrics.expected_return*100:.2f}%")
print(f"  Volatility : {mvo_result.metrics.volatility*100:.2f}%")
print(f"  Sharpe     : {mvo_result.metrics.sharpe_ratio:.3f}")

# ── 7. Visualise ─────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    from portopt.helpers import (plot_efficient_frontier, plot_weights,
                                  plot_bl_comparison)

    fig = plt.figure(figsize=(18, 12))
    gs  = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

    # Efficient frontiers
    ax1 = fig.add_subplot(gs[0, :2])
    plot_efficient_frontier(mvo_frontier, ax=ax1, label="MVO",
                            color="#1565C0")
    plot_efficient_frontier(frontier, ax=ax1, label="Black-Litterman",
                            color="#AD1457", linestyle="--")
    ax1.set_title("Efficient Frontier: MVO vs. Black-Litterman",
                  fontweight="bold")

    # BL prior vs posterior
    ax2 = fig.add_subplot(gs[0, 2])
    plot_bl_comparison(model, assets, ax=ax2)

    # MVO weights
    ax3 = fig.add_subplot(gs[1, 0])
    plot_weights(mvo_result, ax=ax3)
    ax3.set_title("MVO Weights", fontweight="bold")

    # BL weights
    ax4 = fig.add_subplot(gs[1, 1])
    plot_weights(result, ax=ax4)
    ax4.set_title("BL Weights", fontweight="bold")

    # Weight difference
    ax5 = fig.add_subplot(gs[1, 2])
    diff = result.weights - mvo_result.weights
    colors = ["#1B5E20" if d >= 0 else "#B71C1C" for d in diff]
    ax5.bar([a.ticker for a in assets], diff * 100, color=colors,
            edgecolor="white")
    ax5.axhline(0, color="black", linewidth=0.8)
    ax5.set_title("Weight Difference: BL − MVO (%)", fontweight="bold")
    ax5.set_xticklabels([a.ticker for a in assets], rotation=45, ha="right")
    ax5.set_ylabel("Δ Weight (%)")
    ax5.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("Black-Litterman Portfolio Optimisation Analysis",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.savefig("example_bl_output.png", dpi=120, bbox_inches="tight")
    print("\nPlot saved to example_bl_output.png")
    plt.show()

except ImportError:
    print("\n(matplotlib not available — skipping plots)")
