"""
portopt.backtest — walk-forward rolling rebalance simulation (B9).

A thin, dependency-light backtester that drives the optimiser over a
rolling window of historical returns. Designed for research-grade use:
deterministic, reproducible, with a clear separation between data
preparation, signal generation, and execution.

Quick example
-------------
>>> import numpy as np, portopt
>>> R = np.random.randn(500, 5) * 0.01      # daily returns, 5 assets
>>> tickers = list("ABCDE")
>>> def build_weights(window):
...     data = portopt.estimation.from_returns(
...         tickers, window, periods_per_year=252, shrinkage="ledoit-wolf")
...     p = portopt.MVOParameters()
...     p.risk_aversion = 2.0
...     p.constraints = portopt.PortfolioConstraints.long_only(5)
...     return np.asarray(portopt.MVOptimizer(p).optimize(data).weights)
>>> result = portopt.backtest.walk_forward(
...         returns=R, window=126, step=21, build_weights=build_weights)
>>> print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from . import analytics


@dataclass
class BacktestResult:
    """Container for a walk-forward backtest's outputs."""
    weights:           np.ndarray            # (n_rebalances, n_assets)
    rebalance_dates:   np.ndarray            # period indices when rebalanced
    portfolio_returns: np.ndarray            # (T_out_of_sample,)
    turnover:          np.ndarray            # per-rebalance L1 / 2
    trade_list:        List[np.ndarray] = field(default_factory=list)
    transaction_costs: np.ndarray   = field(default_factory=lambda: np.array([]))
    benchmark_returns: Optional[np.ndarray] = None

    def summary(self, rf: float = 0.0, periods_per_year: float = 252.0) -> dict:
        """Standard portopt.analytics summary on the out-of-sample series."""
        s = analytics.summarise(self.portfolio_returns, rf, periods_per_year)
        s["mean_turnover"] = float(self.turnover.mean()) if self.turnover.size else 0.0
        s["n_rebalances"]  = int(self.weights.shape[0])
        if self.benchmark_returns is not None:
            active = self.portfolio_returns - self.benchmark_returns
            s["active_return"] = float(np.prod(1 + active) - 1)
            s["tracking_error"] = float(
                active.std(ddof=1) * np.sqrt(periods_per_year))
            s["information_ratio"] = (
                float(active.mean() / active.std(ddof=1) *
                      np.sqrt(periods_per_year))
                if active.std(ddof=1) > 0 else float("nan"))
        return s


def walk_forward(
    returns: np.ndarray,
    window: int,
    step: int,
    build_weights: Callable[[np.ndarray], np.ndarray],
    *,
    transaction_cost: float = 0.0,
    benchmark_returns: Optional[np.ndarray] = None,
) -> BacktestResult:
    """Run a rolling-window walk-forward backtest.

    Parameters
    ----------
    returns
        T × n historical periodic returns. Row t is the period-t return
        per asset (e.g. daily).
    window
        Estimation lookback. The first weight vector is fit on rows
        [0, window) and applied over [window, window+step). The next
        weight vector is fit on rows [step, window+step) and applied
        over [window+step, window+2*step), and so on.
    step
        Rebalance frequency in rows (e.g. 21 for monthly daily data).
    build_weights
        Callable that takes a (window, n) ndarray of returns and returns
        an (n,) ndarray of long-only / signed portfolio weights summing
        to 1 (or to the user's chosen budget).
    transaction_cost
        Per-unit-traded linear cost; subtracted from the portfolio's
        period-of-rebalance return as `cost * turnover`.
    benchmark_returns
        Optional benchmark return series aligned with `returns` — used
        for tracking-error / IR reporting in `BacktestResult.summary()`.

    Returns
    -------
    BacktestResult
    """
    R = np.asarray(returns, dtype=float)
    if R.ndim != 2:
        raise ValueError("returns must be a 2-D array")
    T, n = R.shape
    if window < 2 or step < 1 or window + step > T:
        raise ValueError("invalid window/step for the supplied returns length")

    weights_list: List[np.ndarray] = []
    dates: List[int]               = []
    port_ret: List[float]          = []
    turnover_list: List[float]     = []
    trade_list: List[np.ndarray]   = []
    tx_costs: List[float]          = []
    prev_w: Optional[np.ndarray]   = None

    t = window
    while t + step <= T:
        train = R[t - window : t, :]
        w = np.asarray(build_weights(train), dtype=float).ravel()
        if w.shape != (n,):
            raise ValueError(
                f"build_weights returned {w.shape}, expected ({n},)")
        weights_list.append(w)
        dates.append(t)
        if prev_w is None:
            turnover = 0.0
            trade = w.copy()
        else:
            trade = w - prev_w
            turnover = 0.5 * np.abs(trade).sum()
        turnover_list.append(turnover)
        trade_list.append(trade)
        cost = transaction_cost * turnover * 2.0   # buy+sell cycle
        tx_costs.append(cost)

        # Out-of-sample period return: weight is fixed across [t, t+step).
        # First period absorbs the rebalance cost.
        for offset in range(step):
            r_t = R[t + offset, :]
            ret = float(w @ r_t)
            if offset == 0:
                ret -= cost
            port_ret.append(ret)
        prev_w = w
        t += step

    return BacktestResult(
        weights           = np.vstack(weights_list),
        rebalance_dates   = np.asarray(dates, dtype=int),
        portfolio_returns = np.asarray(port_ret, dtype=float),
        turnover          = np.asarray(turnover_list, dtype=float),
        trade_list        = trade_list,
        transaction_costs = np.asarray(tx_costs, dtype=float),
        benchmark_returns = (
            None if benchmark_returns is None else
            np.asarray(benchmark_returns, dtype=float)[window:window + len(port_ret)]),
    )


def trade_list(prev_weights: np.ndarray,
               new_weights: np.ndarray,
               portfolio_value: float = 1.0) -> dict:
    """Build a trade list from old → new weights (B9 helper).

    Returns a dict of {ticker_index: {"weight_change": Δw, "notional": Δw * V}}
    suitable for downstream display or order-routing logic.
    """
    prev = np.asarray(prev_weights, dtype=float).ravel()
    new = np.asarray(new_weights, dtype=float).ravel()
    if prev.shape != new.shape:
        raise ValueError("prev_weights and new_weights must have the same shape")
    delta = new - prev
    out = {}
    for i, dw in enumerate(delta):
        if abs(dw) > 1e-12:
            out[i] = {
                "weight_change": float(dw),
                "notional":      float(dw * portfolio_value),
                "side":          "buy" if dw > 0 else "sell",
            }
    return out
