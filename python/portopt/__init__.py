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
    result_to_json,
    result_to_csv,
    frontier_to_json,
    frontier_to_csv,
    bl_model_to_json,
    # Estimation submodule
    estimation,
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
