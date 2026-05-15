"""
portopt.helpers — Pure-Python utilities for analysis and visualisation.

These functions require optional dependencies (matplotlib, pandas, numpy).
They are NOT required for core optimisation functionality.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    import matplotlib.pyplot as plt  # noqa: F401
    import pandas as pd              # noqa: F401
    import numpy as np               # noqa: F401


def _require(pkg: str):
    """Import a package, raising a helpful error if missing."""
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError as exc:
        raise ImportError(
            f"portopt.helpers requires '{pkg}'. "
            f"Install it with: pip install {pkg}"
        ) from exc


# ── Synthetic data ───────────────────────────────────────────────────────────

def make_sample_data(n_assets: int = 5, seed: int = 42, include_benchmark: bool = True):
    """
    Generate synthetic MarketData for testing and demonstration.

    Returns
    -------
    portopt.MarketData
        Populated MarketData with random but plausible μ, Σ, market weights,
        and (optionally) benchmark weights.
    """
    from . import MarketData, Asset
    np = _require("numpy")
    rng = np.random.default_rng(seed)

    tickers = [f"ASSET{i + 1}" for i in range(n_assets)]
    assets = []
    for i, ticker in enumerate(tickers):
        a = Asset()
        a.ticker = ticker
        a.name = f"Sample Asset {i + 1}"
        a.expected_return = float(rng.uniform(0.04, 0.18))
        a.market_cap = float(rng.uniform(1e10, 3e12))
        a.sector = ["Tech", "Financials", "Healthcare", "Energy",
                    "Consumer"][i % 5]
        assets.append(a)

    A = rng.normal(0, 0.1, (n_assets, n_assets))
    sigma = A @ A.T + np.eye(n_assets) * 0.01

    mw = np.ones(n_assets) / n_assets
    er = np.array([a.expected_return for a in assets])

    bench = (np.array([a.market_cap for a in assets])
             / sum(a.market_cap for a in assets)) if include_benchmark else None

    return MarketData.from_arrays(
        assets=assets,
        expected_returns=er,
        covariance=sigma,
        market_weights=mw,
        benchmark_weights=bench,
    )


# ── DataFrame conversions ────────────────────────────────────────────────────

def frontier_to_dataframe(frontier) -> "pd.DataFrame":
    """Convert an EfficientFrontier to a long-form pandas DataFrame."""
    pd = _require("pandas")
    rows = []
    for p in frontier.points:
        row = {
            "risk_aversion":          p.risk_aversion,
            "expected_return":        p.metrics.expected_return,
            "volatility":             p.metrics.volatility,
            "sharpe_ratio":           p.metrics.sharpe_ratio,
            "diversification_ratio":  p.metrics.diversification_ratio,
            "effective_n_assets":     p.metrics.effective_n_assets,
        }
        for i, a in enumerate(frontier.assets):
            row[a.ticker] = float(p.weights[i])
        rows.append(row)
    return pd.DataFrame(rows)


def weights_to_dataframe(result) -> "pd.DataFrame":
    """Convert an OptimizationResult to a per-asset DataFrame."""
    pd = _require("pandas")
    has_rc = len(result.metrics.risk_contribution) == len(result.weights)
    rows = []
    for i, a in enumerate(result.assets):
        rows.append({
            "ticker":            a.ticker,
            "name":              a.name,
            "sector":            getattr(a, "sector", ""),
            "weight":            float(result.weights[i]),
            "weight_pct":        float(result.weights[i]) * 100.0,
            "risk_contribution": float(result.metrics.risk_contribution[i]) if has_rc else math.nan,
        })
    return pd.DataFrame(rows)


def risk_contributions_to_dataframe(result) -> "pd.DataFrame":
    """Per-asset risk contribution table (weight, σ_contribution, % of total)."""
    pd = _require("pandas")
    np = _require("numpy")
    rc = np.array(list(result.metrics.risk_contribution))
    total = rc.sum() if rc.size else math.nan
    rows = []
    for i, a in enumerate(result.assets):
        rows.append({
            "ticker":         a.ticker,
            "weight":         float(result.weights[i]),
            "risk_contrib":   float(rc[i]) if rc.size else math.nan,
            "risk_share_pct": float(rc[i] / total * 100.0)
                              if rc.size and total else math.nan,
        })
    return pd.DataFrame(rows)


# ── Plotting ─────────────────────────────────────────────────────────────────

def _bar_with_xticks(ax, xs, ys, **kw):
    """Helper: set ticks before labels to avoid the matplotlib UserWarning."""
    import numpy as np
    positions = np.arange(len(xs))
    bars = ax.bar(positions, ys, **kw)
    ax.set_xticks(positions)
    ax.set_xticklabels(xs, rotation=45, ha="right")
    return bars


def plot_efficient_frontier(frontier, ax=None, label: Optional[str] = None,
                             show_sharpe: bool = True,
                             risk_free_rate: float = 0.0,
                             **kwargs):
    """
    Plot an efficient frontier on a risk-return diagram.

    If ``risk_free_rate`` > 0, draws the capital-allocation line from (0, rf)
    through the tangency portfolio.
    """
    plt = _require("matplotlib.pyplot")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    vols = [p.metrics.volatility * 100 for p in frontier.points]
    rets = [p.metrics.expected_return * 100 for p in frontier.points]

    lbl = label or frontier.method
    ax.plot(vols, rets, label=lbl, **kwargs)

    if show_sharpe and len(frontier.points) > 0:
        # Recompute Sharpe with provided rf
        sharpes = [
            (p.metrics.expected_return - risk_free_rate) /
            p.metrics.volatility if p.metrics.volatility > 1e-12 else float("-inf")
            for p in frontier.points
        ]
        best_idx = max(range(len(sharpes)), key=lambda i: sharpes[i])
        ax.scatter(vols[best_idx], rets[best_idx],
                   marker="*", s=200, zorder=5, label=f"{lbl} max Sharpe")

        if risk_free_rate > 0.0:
            x = [0.0, vols[best_idx]]
            y = [risk_free_rate * 100.0, rets[best_idx]]
            ax.plot(x, y, linestyle=":", linewidth=1.0,
                    label=f"CAL @ rf={risk_free_rate * 100:.2f}%")

    ax.set_xlabel("Volatility (%)")
    ax.set_ylabel("Expected Return (%)")
    ax.set_title("Efficient Frontier")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    return ax


def plot_weights(result, ax=None, threshold: float = 1e-4, **kwargs):
    """Bar chart of portfolio weights."""
    plt = _require("matplotlib.pyplot")

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    tickers = [a.ticker for a in result.assets]
    weights = list(result.weights)

    pairs = [(t, w) for t, w in zip(tickers, weights) if abs(w) >= threshold]
    if not pairs:
        pairs = list(zip(tickers, weights))

    tickers_f, weights_f = zip(*pairs)
    colors = ["#2196F3" if w >= 0 else "#F44336" for w in weights_f]

    _bar_with_xticks(ax, list(tickers_f), list(weights_f),
                     color=colors, edgecolor="black", linewidth=0.5, **kwargs)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Weight")
    ax.set_title(f"Portfolio Weights — {result.method}")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    return ax


def plot_risk_contributions(result, ax=None, **kwargs):
    """Bar chart of per-asset risk contributions (RC_i / σ in %)."""
    plt = _require("matplotlib.pyplot")
    import numpy as np

    rc = np.array(list(result.metrics.risk_contribution))
    if rc.size == 0:
        raise ValueError("Result has no risk_contribution — recompute metrics.")

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    total = rc.sum()
    if total <= 0:
        share = rc
    else:
        share = rc / total * 100.0

    tickers = [a.ticker for a in result.assets]
    colors = ["#673AB7" if v >= 0 else "#FF5722" for v in share]
    _bar_with_xticks(ax, tickers, share.tolist(), color=colors,
                     edgecolor="black", linewidth=0.5, **kwargs)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Risk share (%)")
    ax.set_title(f"Risk Contributions — {result.method}")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    return ax


def plot_bl_comparison(bl_model, assets, ax=None, **kwargs):
    """Bar chart comparing Black-Litterman prior vs. posterior expected returns."""
    plt = _require("matplotlib.pyplot")
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    n = len(assets)
    tickers = [a.ticker for a in assets]
    prior = list(np.array(list(bl_model.prior_returns)) * 100)
    posterior = list(np.array(list(bl_model.posterior_returns)) * 100)

    x = np.arange(n)
    width = 0.35

    ax.bar(x - width / 2, prior, width, label="Prior (equilibrium)",
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


# Backwards-compatible alias
plot_prior_vs_posterior = plot_bl_comparison
