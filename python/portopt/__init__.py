"""
portopt — Portfolio Optimisation Library
=========================================

Python wrapper around the portopt C++ library.
Exposes MVO and Black-Litterman portfolio optimisation.

Quick start
-----------
>>> import portopt
>>> portopt.init_logging(portopt.LogLevel.Info)
>>>
>>> data = portopt.read_market_data("assets.json")
>>>
>>> params = portopt.MVOParameters()
>>> params.risk_aversion = 2.0
>>> params.constraints = portopt.PortfolioConstraints.long_only(len(data.assets))
>>>
>>> opt    = portopt.MVOptimizer(params)
>>> result = opt.optimize(data)
>>> print(f"Sharpe: {result.metrics.sharpe_ratio:.3f}")
>>>
>>> frontier = opt.efficient_frontier(data)
>>> df = frontier.to_dataframe()  # requires pandas
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
    MVOParameters,
    View,
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
    read_mvo_parameters,
    read_bl_parameters,
    result_to_json,
    frontier_to_json,
    bl_model_to_json,
    # Version
    __version__,
)

from .helpers import (  # noqa: F401
    plot_efficient_frontier,
    plot_weights,
    plot_bl_comparison,
    make_sample_data,
)
