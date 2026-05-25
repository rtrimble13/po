"""
portopt — Portfolio Optimisation Library
=========================================

Python wrapper around the portopt C++ library.
Exposes MVO and Black-Litterman portfolio optimisation, plus PM-friendly
helpers: target-volatility / target-return / max-Sharpe / min-variance
portfolios, covariance shrinkage, risk contributions, and benchmark-relative
metrics (tracking error, IR, active share, beta).

Quick start
-----------
>>> import portopt
>>> portopt.init_logging(portopt.LogLevel.Info)
>>>
>>> data = portopt.read_market_data("assets.json")
>>>
>>> params = portopt.MVOParameters()
>>> params.risk_aversion = 2.0
>>> params.constraints   = portopt.PortfolioConstraints.long_only(len(data.assets))
>>>
>>> opt    = portopt.MVOptimizer(params)
>>> result = opt.optimize(data)
>>> print(f"Sharpe: {result.metrics.sharpe_ratio:.3f}")
>>>
>>> # PM-friendly shortcuts:
>>> tgt_vol = opt.optimize_for_target_volatility(data, 0.15)
>>> max_shp = opt.max_sharpe_portfolio(data)
>>>
>>> frontier = opt.efficient_frontier(data)
>>> df = frontier.to_dataframe()  # requires pandas

Estimation from returns
-----------------------
>>> import numpy as np
>>> R = np.random.randn(252, 5) * 0.01
>>> data = portopt.estimation.from_returns(
...     ["A", "B", "C", "D", "E"], R, periods_per_year=252,
...     shrinkage="ledoit-wolf")
"""

from ._portopt import (  # noqa: F401
    # Logging
    LogLevel,
    init_logging,
    set_log_level,
    add_file_log,
    # Typed exceptions (C4)
    PortoptError,
    InvalidMarketData,
    InvalidParameters,
    InfeasibleProblem,
    SolverDidNotConverge,
    SolverCancelled,
    SolverTimeout,
    # Cancellation (C5)
    CancellationToken,
    # Data types
    Asset,
    MarketData,
    PortfolioConstraints,
    GroupConstraint,
    MVOParameters,
    View,
    ViewConfidenceMode,
    BlackLittermanParameters,
    PortfolioMetrics,
    OptimizationResult,
    EfficientFrontierPoint,
    EfficientFrontier,
    BLModelOutput,
    # Optimisers
    MVOptimizer,
    BlackLittermanOptimizer,
    # IO
    read_market_data,
    read_market_data_json,
    read_returns_csv,
    read_mvo_parameters,
    read_bl_parameters,
    read_mvo_parameters_json,    # C2
    read_bl_parameters_json,      # C2
    result_to_json,
    result_to_csv,
    frontier_to_json,
    frontier_to_csv,
    bl_model_to_json,
    # Estimation submodule
    estimation,
    # Portfolios submodule (B15 / B3)
    portfolios,
    # Factor risk model (B7 / B8)
    FactorRiskModel,
    FactorRiskDecomposition,
    decompose_risk,
    factor_neutral_constraint,
    beta_neutral_constraint,
    # Version
    __version__,
)

from .helpers import (  # noqa: F401
    plot_efficient_frontier,
    plot_weights,
    plot_bl_comparison,
    plot_risk_contributions,
    plot_prior_vs_posterior,
    make_sample_data,
    frontier_to_dataframe,
    weights_to_dataframe,
    risk_contributions_to_dataframe,
)

# B9 walk-forward backtester, B10 advanced analytics, B11 attribution
# All three are pure-Python and require only NumPy.
from . import analytics       # noqa: F401
from . import backtest        # noqa: F401
from . import attribution     # noqa: F401

# C3: Pydantic schemas — opt-in (only imported when pydantic is available).
# C7: Reference MCP tool dispatcher depends on schemas.
try:
    from . import schemas         # noqa: F401
    from . import mcp_server      # noqa: F401
except ImportError:
    pass                          # pydantic not installed; schemas not available

# ── C1: dict ↔ struct helpers (MCP-friendly) ─────────────────────────────────
# These thin wrappers let callers (especially MCP servers) build/inspect
# parameter objects entirely through Python dicts, without touching the
# field-by-field C++ struct accessors.

import json as _json


def mvo_params_from_dict(d):
    """Construct MVOParameters from a plain Python dict / JSON-compatible mapping.

    The mapping may be wrapped under an outer "mvo" key (as in the file
    format) or be the bare parameters. Field names and shapes are identical
    to the JSON parameter schema documented in the README.
    """
    return read_mvo_parameters_json(_json.dumps(d, default=_json_default))


def bl_params_from_dict(d):
    """Construct BlackLittermanParameters from a plain Python dict.

    The mapping may be wrapped under an outer "black_litterman" key or be
    the bare parameters. Field names match the JSON parameter schema.
    """
    return read_bl_parameters_json(_json.dumps(d, default=_json_default))


def market_data_from_dict(d):
    """Construct MarketData from a JSON-compatible Python dict.

    Wraps read_market_data_json so MCP tools can take inline payloads.
    """
    return read_market_data_json(_json.dumps(d, default=_json_default))


def _json_default(obj):
    """JSON encoder for numpy arrays / Eigen vectors handed in as values."""
    try:
        return obj.tolist()
    except AttributeError:
        raise TypeError(f"Cannot JSON-encode object of type {type(obj).__name__}")


def mvo_params_to_dict(p):
    """Dump MVOParameters to a plain Python dict (round-trips through JSON)."""
    return {
        "risk_aversion":     p.risk_aversion,
        "frontier_points":   p.frontier_points,
        "min_risk_aversion": p.min_risk_aversion,
        "max_risk_aversion": p.max_risk_aversion,
        "risk_free_rate":    p.risk_free_rate,
        "risk_free_rate_is_set": p.risk_free_rate_is_set,
        "group_penalty":     p.group_penalty,
        "hard_group_constraints": p.hard_group_constraints,
        "group_tolerance":   p.group_tolerance,
        "use_tangent_reformulation": p.use_tangent_reformulation,
        "linear_transaction_cost": list(p.linear_transaction_cost)
                                   if p.linear_transaction_cost.size > 0 else [],
        "quadratic_transaction_cost": list(p.quadratic_transaction_cost)
                                      if p.quadratic_transaction_cost.size > 0 else [],
        "timeout_ms": p.timeout_ms,
        "constraints": {
            "lower_bounds":     list(p.constraints.lower_bounds),
            "upper_bounds":     list(p.constraints.upper_bounds),
            "allow_short_selling": p.constraints.allow_short_selling,
            "budget":           p.constraints.budget,
            "turnover_penalty": p.constraints.turnover_penalty,
            "tracking_error_limit": p.constraints.tracking_error_limit,
            "gross_exposure_limit": p.constraints.gross_exposure_limit,
            "current_weights":  list(p.constraints.current_weights)
                                if p.constraints.current_weights.size > 0 else [],
            "groups": [
                {
                    "description":  g.description,
                    "coefficients": list(g.coefficients),
                    "lower":        g.lower,
                    "upper":        g.upper,
                }
                for g in p.constraints.groups
            ],
        },
    }


def bl_params_to_dict(p):
    """Dump BlackLittermanParameters to a plain Python dict."""
    return {
        "tau":                     p.tau,
        "risk_aversion":           p.risk_aversion,
        "confidence_mode":         "idzorek"
                                    if p.confidence_mode == ViewConfidenceMode.Idzorek
                                    else "variance",
        "propagate_risk_aversion": p.propagate_risk_aversion,
        "views": [
            {
                "description":      v.description,
                "pick_vector":      list(v.pick_vector),
                "expected_return":  v.expected_return,
                "confidence":       v.confidence,
            }
            for v in p.views
        ],
        "mvo_params": mvo_params_to_dict(p.mvo_params),
    }


def market_data_to_dict(d):
    """Dump MarketData to a plain Python dict (mirrors the JSON file schema)."""
    out = {
        "assets": [
            {
                "ticker":          a.ticker,
                "name":            a.name,
                "expected_return": a.expected_return,
                "market_cap":      a.market_cap,
                "sector":          a.sector,
                "currency":        a.currency,
            }
            for a in d.assets
        ],
        "covariance":     [list(row) for row in d.covariance],
        "risk_free_rate": d.risk_free_rate,
    }
    if d.expected_returns is not None:
        out["expected_returns"] = list(d.expected_returns)
    if d.market_weights is not None:
        out["market_weights"] = list(d.market_weights)
    if d.benchmark_weights is not None:
        out["benchmark_weights"] = list(d.benchmark_weights)
    return out
