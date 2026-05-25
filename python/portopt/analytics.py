"""
portopt.analytics — advanced performance and risk analytics (B10).

Functions in this module operate on a 1-D NumPy array of periodic returns
(or a DataFrame whose values are returns). They are stateless and do not
require the C++ core; the library still works without pandas / numpy
installed, but these helpers require both.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np


# ── Drawdown statistics ──────────────────────────────────────────────────────

def equity_curve(returns: np.ndarray, initial: float = 1.0) -> np.ndarray:
    """Cumulative-product equity curve from periodic returns."""
    r = np.asarray(returns, dtype=float).ravel()
    return initial * np.cumprod(1.0 + r)


def max_drawdown(returns: np.ndarray) -> Tuple[float, int, int]:
    """Return (max_drawdown, peak_index, trough_index).

    `max_drawdown` is reported as a positive fraction (e.g. 0.23 for a 23 %
    drawdown). Indices refer to the period that was the peak and the
    period at the bottom of the worst drawdown.
    """
    eq = equity_curve(returns)
    running_max = np.maximum.accumulate(eq)
    dd = (eq - running_max) / running_max
    trough = int(np.argmin(dd))
    peak = int(np.argmax(eq[: trough + 1]))
    return float(-dd[trough]), peak, trough


def ulcer_index(returns: np.ndarray) -> float:
    """Ulcer index — RMS of percentage drawdowns over the period."""
    eq = equity_curve(returns)
    running_max = np.maximum.accumulate(eq)
    dd_pct = (eq - running_max) / running_max * 100.0
    return float(np.sqrt(np.mean(dd_pct ** 2)))


# ── Risk-adjusted return ratios ──────────────────────────────────────────────

def sharpe_ratio(returns: np.ndarray, rf: float = 0.0,
                 periods_per_year: float = 252.0) -> float:
    r = np.asarray(returns, dtype=float).ravel() - rf / periods_per_year
    if r.std(ddof=1) <= 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def sortino_ratio(returns: np.ndarray, rf: float = 0.0, mar: float = 0.0,
                  periods_per_year: float = 252.0) -> float:
    """Downside-deviation-adjusted Sharpe.

    `mar` is the minimum-acceptable-return per period (default 0).
    """
    r = np.asarray(returns, dtype=float).ravel() - rf / periods_per_year
    excess = r - mar
    downside = np.minimum(excess, 0.0)
    dd = np.sqrt(np.mean(downside ** 2))
    if dd <= 0:
        return float("nan")
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def calmar_ratio(returns: np.ndarray,
                 periods_per_year: float = 252.0) -> float:
    """Annualised return divided by max drawdown (positive fraction)."""
    r = np.asarray(returns, dtype=float).ravel()
    ann_ret = (np.prod(1.0 + r)) ** (periods_per_year / len(r)) - 1.0
    dd, _, _ = max_drawdown(r)
    if dd <= 0:
        return float("nan")
    return float(ann_ret / dd)


def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """Ratio of average gains above `threshold` to average losses below."""
    r = np.asarray(returns, dtype=float).ravel() - threshold
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses <= 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def downside_deviation(returns: np.ndarray, mar: float = 0.0) -> float:
    r = np.asarray(returns, dtype=float).ravel() - mar
    return float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)))


# ── Aggregator ───────────────────────────────────────────────────────────────

def summarise(returns: np.ndarray, rf: float = 0.0,
              periods_per_year: float = 252.0) -> dict:
    """Compute the standard B10 risk/return summary as a plain dict."""
    r = np.asarray(returns, dtype=float).ravel()
    dd, peak, trough = max_drawdown(r)
    return {
        "n_periods":          int(r.size),
        "total_return":       float(np.prod(1.0 + r) - 1.0),
        "annualised_return":  float(
            (np.prod(1.0 + r)) ** (periods_per_year / len(r)) - 1.0),
        "annualised_vol":     float(r.std(ddof=1) * np.sqrt(periods_per_year)),
        "sharpe":             sharpe_ratio(r, rf, periods_per_year),
        "sortino":            sortino_ratio(r, rf, 0.0, periods_per_year),
        "calmar":             calmar_ratio(r, periods_per_year),
        "omega":              omega_ratio(r, 0.0),
        "max_drawdown":       dd,
        "max_drawdown_peak":  peak,
        "max_drawdown_trough":trough,
        "ulcer_index":        ulcer_index(r),
        "downside_deviation": downside_deviation(r, 0.0),
    }
