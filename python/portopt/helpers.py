"""
portopt.helpers — Pure-Python utilities for analysis and visualisation.

These functions require optional dependencies (matplotlib, pandas, numpy).
They are NOT required for core optimisation functionality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np


def _require(pkg: str):
    """Import a package, raising a helpful error if missing."""
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError:
        raise ImportError(
            f"portopt.helpers requires '{pkg}'. "
            f"Install it with: pip install {pkg}"
        )


def make_sample_data(n_assets: int = 5, seed: int = 42):
    """
    Generate synthetic MarketData for testing and demonstration.

    Parameters
    ----------
    n_assets : int
        Number of assets.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    portopt.MarketData
        Populated MarketData with random but plausible μ, Σ, and market weights.
    """
    from . import MarketData, Asset
    np = _require("numpy")
    rng = np.random.default_rng(seed)

    tickers = [f"ASSET{i+1}" for i in range(n_assets)]
    assets = []
    for i, ticker in enumerate(tickers):
        a = Asset()
        a.ticker = ticker
        a.name = f"Sample Asset {i + 1}"
        a.expected_return = float(rng.uniform(0.04, 0.18))
        a.market_cap = float(rng.uniform(1e10, 3e12))
        assets.append(a)

    # Generate a random positive-definite covariance matrix
    A = rng.normal(0, 0.1, (n_assets, n_assets))
    sigma = A @ A.T + np.eye(n_assets) * 0.01  # ensure PD

    # Equal market weights
    mw = np.ones(n_assets) / n_assets
    er = np.array([a.expected_return for a in assets])

    return MarketData.from_arrays(
        assets=assets,
        expected_returns=er,
        covariance=sigma,
        market_weights=mw,
    )


def plot_efficient_frontier(frontier, ax=None, label: Optional[str] = None,
                             show_sharpe: bool = True, **kwargs):
    """
    Plot an efficient frontier on a risk-return diagram.

    Parameters
    ----------
    frontier : portopt.EfficientFrontier
        Frontier to plot.
    ax : matplotlib.axes.Axes, optional
        Target axes; creates a new figure if None.
    label : str, optional
        Legend label; defaults to frontier.method.
    show_sharpe : bool
        If True, mark the maximum-Sharpe-ratio portfolio with a star.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _require("matplotlib.pyplot")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    vols = [p.metrics.volatility * 100 for p in frontier.points]
    rets = [p.metrics.expected_return * 100 for p in frontier.points]

    lbl = label or frontier.method
    ax.plot(vols, rets, label=lbl, **kwargs)

    if show_sharpe:
        sharpes = [p.metrics.sharpe_ratio for p in frontier.points]
        best_idx = sharpes.index(max(sharpes))
        ax.scatter(vols[best_idx], rets[best_idx],
                   marker="*", s=200, zorder=5, label=f"{lbl} max Sharpe")

    ax.set_xlabel("Volatility (%)")
    ax.set_ylabel("Expected Return (%)")
    ax.set_title("Efficient Frontier")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    return ax


def plot_weights(result, ax=None, threshold: float = 1e-4, **kwargs):
    """
    Bar chart of portfolio weights.

    Parameters
    ----------
    result : portopt.OptimizationResult
        Optimisation result.
    ax : matplotlib.axes.Axes, optional
    threshold : float
        Weights below this value are excluded.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _require("matplotlib.pyplot")
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    tickers = [a.ticker for a in result.assets]
    weights = list(result.weights)

    # Filter near-zero weights
    pairs = [(t, w) for t, w in zip(tickers, weights) if abs(w) >= threshold]
    if not pairs:
        pairs = list(zip(tickers, weights))

    tickers_f, weights_f = zip(*pairs)
    colors = ["#2196F3" if w >= 0 else "#F44336" for w in weights_f]

    ax.bar(tickers_f, weights_f, color=colors, edgecolor="black", linewidth=0.5,
           **kwargs)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Weight")
    ax.set_title(f"Portfolio Weights — {result.method}")
    ax.set_xticklabels(tickers_f, rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    return ax


def plot_bl_comparison(bl_model, assets, ax=None, **kwargs):
    """
    Bar chart comparing Black-Litterman prior vs. posterior expected returns.

    Parameters
    ----------
    bl_model : portopt.BLModelOutput
    assets : list of portopt.Asset
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _require("matplotlib.pyplot")
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    n = len(assets)
    tickers = [a.ticker for a in assets]
    prior     = list(bl_model.prior_returns * 100)
    posterior = list(bl_model.posterior_returns * 100)

    x = np.arange(n)
    width = 0.35

    ax.bar(x - width / 2, prior,     width, label="Prior (equilibrium)",
           color="#90CAF9", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, posterior, width, label="Posterior (BL)",
           color="#1565C0", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=45, ha="right")
    ax.set_ylabel("Expected Return (%)")
    ax.set_title("Black-Litterman: Prior vs. Posterior Returns")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    return ax


def frontier_to_dataframe(frontier) -> "pd.DataFrame":
    """
    Convert an EfficientFrontier to a pandas DataFrame.

    Columns: risk_aversion, expected_return, volatility, sharpe_ratio,
             plus one column per asset with its weight.
    """
    pd = _require("pandas")
    rows = []
    for p in frontier.points:
        row = {
            "risk_aversion":   p.risk_aversion,
            "expected_return": p.metrics.expected_return,
            "volatility":      p.metrics.volatility,
            "sharpe_ratio":    p.metrics.sharpe_ratio,
        }
        for i, a in enumerate(frontier.assets):
            row[a.ticker] = p.weights[i]
        rows.append(row)
    return pd.DataFrame(rows)
