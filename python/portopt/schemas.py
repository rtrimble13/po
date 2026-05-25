"""
portopt.schemas — Pydantic input/output schemas (C3).

These schemas mirror the C++ structs in `include/portopt/types.hpp` and
are used:

  * As validators when an MCP server / HTTP endpoint deserialises a
    user payload before handing it to the C++ optimiser.
  * As JSON-Schema sources for auto-generating MCP tool definitions
    (`MVOParametersSchema.model_json_schema()`).
  * As convenient `from_dict` / `to_dict` round-trippers in places
    where the existing C++ JSON readers would otherwise require a temp
    file.

Pydantic v2 is a hard dependency of this module; importing
`portopt.schemas` triggers `import pydantic`. The C++ core is fully
usable without pydantic — schemas are opt-in.

Quick example
-------------
>>> from portopt.schemas import MVOParametersSchema
>>> p = MVOParametersSchema.model_validate({
...     "risk_aversion": 2.0,
...     "constraints":   {"lower_bounds": [0, 0], "upper_bounds": [1, 1]},
... })
>>> mvo_params = p.to_portopt()         # construct the pybind11 type
>>> # Or export the schema for an MCP tool definition:
>>> schema = MVOParametersSchema.model_json_schema()
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

try:
    from pydantic import BaseModel, ConfigDict, Field
except ImportError as _ex:                # pragma: no cover
    raise ImportError(
        "portopt.schemas requires pydantic >= 2.0. "
        "Install with: pip install pydantic"
    ) from _ex

# The C++ binding module is imported lazily to keep `import portopt.schemas`
# usable even when only the schemas — not the optimiser — are needed.
def _binding():
    from . import _portopt   # type: ignore
    return _portopt


# ── Asset / MarketData ───────────────────────────────────────────────────────

class AssetSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker:          str   = ""
    name:            str   = ""
    expected_return: float = 0.0
    market_cap:      float = 0.0
    sector:          str   = ""
    currency:        str   = ""

    def to_portopt(self):
        a = _binding().Asset()
        a.ticker          = self.ticker
        a.name            = self.name
        a.expected_return = self.expected_return
        a.market_cap      = self.market_cap
        a.sector          = self.sector
        a.currency        = self.currency
        return a


class MarketDataSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    assets:             List[AssetSchema]
    expected_returns:   List[float]
    covariance:         List[List[float]]
    market_weights:     Optional[List[float]] = None
    benchmark_weights:  Optional[List[float]] = None
    risk_free_rate:     float = 0.0

    def to_portopt(self):
        import numpy as np
        pp = _binding()
        return pp.MarketData.from_arrays(
            [a.to_portopt() for a in self.assets],
            np.asarray(self.expected_returns, dtype=float),
            np.asarray(self.covariance, dtype=float),
            None if self.market_weights is None
                 else np.asarray(self.market_weights, dtype=float),
            None if self.benchmark_weights is None
                 else np.asarray(self.benchmark_weights, dtype=float),
            self.risk_free_rate,
        )


# ── Constraints ──────────────────────────────────────────────────────────────

class GroupConstraintSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description:  str
    coefficients: List[float]
    lower:        float = Field(-1e30,
        description="Lower bound on a'w; -1e30 = effectively disabled")
    upper:        float = Field(1e30,
        description="Upper bound on a'w; +1e30 = effectively disabled")

    def to_portopt(self):
        import numpy as np
        g = _binding().GroupConstraint()
        g.description  = self.description
        g.coefficients = np.asarray(self.coefficients, dtype=float)
        g.lower        = self.lower
        g.upper        = self.upper
        return g


class PortfolioConstraintsSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lower_bounds:          List[float]      = []
    upper_bounds:          List[float]      = []
    allow_short_selling:   bool             = False
    budget:                float            = 1.0
    current_weights:       List[float]      = []
    turnover_penalty:      float            = 0.0
    groups:                List[GroupConstraintSchema] = []
    tracking_error_limit:  float            = 0.0
    gross_exposure_limit:  float            = 0.0

    def to_portopt(self):
        import numpy as np
        c = _binding().PortfolioConstraints()
        if self.lower_bounds:
            c.lower_bounds = np.asarray(self.lower_bounds, dtype=float)
        if self.upper_bounds:
            c.upper_bounds = np.asarray(self.upper_bounds, dtype=float)
        c.allow_short_selling   = self.allow_short_selling
        c.budget                = self.budget
        if self.current_weights:
            c.current_weights = np.asarray(self.current_weights, dtype=float)
        c.turnover_penalty      = self.turnover_penalty
        c.groups                = [g.to_portopt() for g in self.groups]
        c.tracking_error_limit  = self.tracking_error_limit
        c.gross_exposure_limit  = self.gross_exposure_limit
        return c


# ── MVO parameters ───────────────────────────────────────────────────────────

class MVOParametersSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    risk_aversion:             float = 1.0
    frontier_points:           int   = 50
    min_risk_aversion:         float = 0.01
    max_risk_aversion:         float = 100.0
    risk_free_rate:            float = 0.0
    group_penalty:             float = 1e3
    hard_group_constraints:    bool  = False
    group_tolerance:           float = 1e-6
    use_tangent_reformulation: bool  = True
    linear_transaction_cost:   List[float] = []
    quadratic_transaction_cost:List[float] = []
    timeout_ms:                float = 0.0
    constraints:               PortfolioConstraintsSchema = PortfolioConstraintsSchema()

    def to_portopt(self):
        import numpy as np
        pp = _binding()
        p = pp.MVOParameters()
        p.risk_aversion             = self.risk_aversion
        p.frontier_points           = self.frontier_points
        p.min_risk_aversion         = self.min_risk_aversion
        p.max_risk_aversion         = self.max_risk_aversion
        p.risk_free_rate            = self.risk_free_rate
        p.group_penalty             = self.group_penalty
        p.hard_group_constraints    = self.hard_group_constraints
        p.group_tolerance           = self.group_tolerance
        p.use_tangent_reformulation = self.use_tangent_reformulation
        if self.linear_transaction_cost:
            p.linear_transaction_cost = np.asarray(
                self.linear_transaction_cost, dtype=float)
        if self.quadratic_transaction_cost:
            p.quadratic_transaction_cost = np.asarray(
                self.quadratic_transaction_cost, dtype=float)
        p.timeout_ms                = self.timeout_ms
        p.constraints               = self.constraints.to_portopt()
        return p


# ── Black-Litterman parameters ───────────────────────────────────────────────

class ViewSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description:     str
    pick_vector:     List[float]
    expected_return: float
    confidence:      float = 0.1

    def to_portopt(self):
        import numpy as np
        v = _binding().View()
        v.description     = self.description
        v.pick_vector     = np.asarray(self.pick_vector, dtype=float)
        v.expected_return = self.expected_return
        v.confidence      = self.confidence
        return v


class BlackLittermanParametersSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tau:                      float = 0.05
    risk_aversion:            float = 2.5
    confidence_mode:          str   = "variance"   # "variance" | "idzorek"
    propagate_risk_aversion:  bool  = True
    views:                    List[ViewSchema] = []
    mvo_params:               MVOParametersSchema = MVOParametersSchema()

    def to_portopt(self):
        pp = _binding()
        p = pp.BlackLittermanParameters()
        p.tau                     = self.tau
        p.risk_aversion           = self.risk_aversion
        p.confidence_mode         = (
            pp.ViewConfidenceMode.Idzorek
            if self.confidence_mode.lower() == "idzorek"
            else pp.ViewConfidenceMode.Variance)
        p.propagate_risk_aversion = self.propagate_risk_aversion
        p.views                   = [v.to_portopt() for v in self.views]
        p.mvo_params              = self.mvo_params.to_portopt()
        return p


# ── Output schemas (for tool response shapes) ────────────────────────────────

class PortfolioMetricsSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    expected_return:        float
    volatility:             float
    sharpe_ratio:           float
    variance:               float
    risk_contribution:      List[float]
    diversification_ratio:  float
    effective_n_assets:     float
    tracking_error:         float
    information_ratio:      float
    active_share:           float
    beta_to_benchmark:      float
    turnover:               float


class OptimizationResultSchema(BaseModel):
    """Stable JSON-friendly view of an OptimizationResult.

    Useful as an MCP tool's output_schema for tool discovery.
    """
    model_config = ConfigDict(extra="ignore")
    weights:           List[float]
    tickers:           List[str]
    method:            str
    converged:         bool
    iterations:        int
    status_message:    str
    solve_time_ms:     float
    primal_residual:   float
    kkt_residual:      float
    dual_estimate:     float
    library_version:   str
    input_hash:        str
    params_hash:       str
    metrics:           PortfolioMetricsSchema


# ── Tool manifest helper for MCP discovery ───────────────────────────────────

def tool_manifest() -> dict:
    """Return a dict of {tool_name: {"input_schema": ..., "description": ...}}.

    Designed for an MCP server to enumerate the optimiser's surface and
    auto-publish tool definitions. The schemas are produced by
    `model_json_schema()` so they're already JSON-Schema-Draft-2020-12.
    """
    return {
        "optimize_mvo": {
            "description":
                "Run Mean-Variance Optimisation over the supplied market data "
                "with the given parameters. Returns the optimal weights, "
                "convergence diagnostics, and full portfolio metrics.",
            "input_schema":  {
                "data":   MarketDataSchema.model_json_schema(),
                "params": MVOParametersSchema.model_json_schema(),
            },
            "output_schema": OptimizationResultSchema.model_json_schema(),
        },
        "optimize_black_litterman": {
            "description":
                "Run Black-Litterman optimisation. Combines the equilibrium "
                "market prior with investor views to produce posterior expected "
                "returns and the corresponding MVO portfolio.",
            "input_schema":  {
                "data":   MarketDataSchema.model_json_schema(),
                "params": BlackLittermanParametersSchema.model_json_schema(),
            },
            "output_schema": OptimizationResultSchema.model_json_schema(),
        },
        "min_variance":   {"description":
            "Minimum-variance portfolio (λ → 0)."},
        "max_sharpe":     {"description":
            "Maximum-Sharpe (tangency) portfolio."},
        "target_volatility": {"description":
            "Portfolio with realised volatility ≈ target."},
        "target_return":  {"description":
            "Portfolio with expected return ≈ target."},
        "build_efficient_frontier": {"description":
            "Sweep λ over [min, max] and return the full efficient frontier."},
        "estimate_covariance": {"description":
            "Estimate Σ from a returns matrix via sample / Ledoit-Wolf / OAS."},
        "summarise_portfolio": {"description":
            "Compute metrics (Sharpe, TE, IR, risk contributions…) for "
            "user-supplied weights and market data."},
        "validate_market_data": {"description":
            "Validate that MarketData is well-formed (dimensions, PSD-ness, "
            "etc.) without running an optimisation."},
    }
