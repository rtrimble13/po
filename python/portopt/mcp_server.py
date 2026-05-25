"""
portopt.mcp_server — reference MCP-style tool server (C7).

A small, transport-agnostic tool dispatcher that exposes the optimiser
through a uniform call/response interface. It does NOT speak the MCP
wire protocol itself — wire that up in a separate file with your
preferred MCP framework (e.g. anthropic-ai/mcp Python SDK) — but it
provides the tool registry, JSON-Schema discovery, structured error
mapping, and the call dispatcher that an MCP server needs.

Quick example
-------------
>>> from portopt.mcp_server import PortoptToolServer
>>> server = PortoptToolServer()
>>> server.list_tools()              # discovery
>>> server.call("min_variance", data={"assets":[...], "covariance":[...], ...})
>>> server.call("optimize_mvo", data=..., params=...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import json

from . import (
    MVOptimizer,
    BlackLittermanOptimizer,
    estimation,
    portfolios,
    PortoptError,
    InvalidMarketData,
    InvalidParameters,
    InfeasibleProblem,
    SolverDidNotConverge,
    SolverCancelled,
    SolverTimeout,
)
from . import schemas as _S


# ── Response shapes ──────────────────────────────────────────────────────────

@dataclass
class ToolResponse:
    """Generic structured response: either `data` or `error` is set."""
    data:  Optional[dict] = None
    error: Optional[dict] = None

    def to_dict(self) -> dict:
        return {"data": self.data} if self.error is None else {"error": self.error}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _result_to_dict(result) -> dict:
    """Convert OptimizationResult to a JSON-safe dict."""
    m = result.metrics
    return {
        "weights":         list(result.weights),
        "tickers":         [a.ticker for a in result.assets],
        "method":          result.method,
        "converged":       bool(result.converged),
        "iterations":      int(result.iterations),
        "status_message":  result.status_message,
        "solve_time_ms":   float(result.solve_time_ms),
        "primal_residual": float(result.primal_residual),
        "kkt_residual":    float(result.kkt_residual),
        "dual_estimate":   float(result.dual_estimate),
        "library_version": result.library_version,
        "input_hash":      result.input_hash,
        "params_hash":     result.params_hash,
        "metrics": {
            "expected_return":       float(m.expected_return),
            "volatility":            float(m.volatility),
            "sharpe_ratio":          float(m.sharpe_ratio),
            "variance":              float(m.variance),
            "risk_contribution":     list(m.risk_contribution),
            "diversification_ratio": float(m.diversification_ratio),
            "effective_n_assets":    float(m.effective_n_assets),
            "tracking_error":        float(m.tracking_error),
            "information_ratio":     float(m.information_ratio),
            "active_share":          float(m.active_share),
            "beta_to_benchmark":     float(m.beta_to_benchmark),
            "turnover":              float(m.turnover),
        },
    }


def _frontier_to_dict(ef) -> dict:
    return {
        "method": ef.method,
        "tickers": [a.ticker for a in ef.assets],
        "points": [
            {
                "risk_aversion":   float(pt.risk_aversion),
                "weights":         list(pt.weights),
                "expected_return": float(pt.metrics.expected_return),
                "volatility":      float(pt.metrics.volatility),
                "sharpe_ratio":    float(pt.metrics.sharpe_ratio),
            }
            for pt in ef.points
        ],
    }


def _parse_code(msg: str) -> tuple:
    """Pull the `[code]` prefix the C++ layer attaches to PortoptError messages."""
    if msg.startswith("[") and "] " in msg:
        end = msg.index("]")
        return msg[1:end], msg[end + 2:]
    return "unknown", msg


def _error_response(err: BaseException) -> dict:
    """Translate any caught exception to a structured error payload."""
    if isinstance(err, PortoptError):
        code, message = _parse_code(str(err))
        return {
            "type":    type(err).__name__,
            "code":    code,
            "message": message,
        }
    return {
        "type":    type(err).__name__,
        "code":    "unhandled_exception",
        "message": str(err),
    }


# ── Server / dispatcher ──────────────────────────────────────────────────────

class PortoptToolServer:
    """Stateless tool dispatcher for portopt.

    Each public method maps to one MCP tool. The class is dependency-light
    and can be wrapped by any MCP framework that needs a Python entry
    point per tool.
    """

    # Tool 1 — list available tools (discovery)
    def list_tools(self) -> List[dict]:
        manifest = _S.tool_manifest()
        out = []
        for name, info in manifest.items():
            out.append({
                "name":        name,
                "description": info.get("description", ""),
                "input_schema":  info.get("input_schema"),
                "output_schema": info.get("output_schema"),
            })
        return out

    # Tool 2 — generic call dispatcher (mirrors MCP "call_tool")
    def call(self, tool_name: str, **kwargs) -> dict:
        try:
            method = getattr(self, "_tool_" + tool_name, None)
            if method is None:
                return ToolResponse(error={
                    "code":    "unknown_tool",
                    "message": f"Unknown tool: {tool_name}",
                }).to_dict()
            return ToolResponse(data=method(**kwargs)).to_dict()
        except (InvalidMarketData, InvalidParameters,
                InfeasibleProblem, SolverDidNotConverge,
                SolverCancelled, SolverTimeout,
                PortoptError, Exception) as e:
            return ToolResponse(error=_error_response(e)).to_dict()

    # ── Tool implementations ─────────────────────────────────────────────────

    def _tool_validate_market_data(self, *, data: dict) -> dict:
        d = _S.MarketDataSchema.model_validate(data).to_portopt()
        MVOptimizer.validate_market_data(d)
        return {"valid": True, "n_assets": len(d.assets)}

    def _tool_optimize_mvo(self, *, data: dict, params: dict) -> dict:
        d = _S.MarketDataSchema.model_validate(data).to_portopt()
        p = _S.MVOParametersSchema.model_validate(params).to_portopt()
        opt = MVOptimizer(p)
        return _result_to_dict(opt.optimize(d))

    def _tool_optimize_black_litterman(self, *, data: dict, params: dict) -> dict:
        d = _S.MarketDataSchema.model_validate(data).to_portopt()
        p = _S.BlackLittermanParametersSchema.model_validate(params).to_portopt()
        opt = BlackLittermanOptimizer(p)
        return _result_to_dict(opt.optimize(d))

    def _tool_min_variance(self, *, data: dict, params: dict = None) -> dict:
        d = _S.MarketDataSchema.model_validate(data).to_portopt()
        p = (_S.MVOParametersSchema.model_validate(params).to_portopt()
             if params is not None else None)
        opt = MVOptimizer(p) if p is not None else MVOptimizer()
        return _result_to_dict(opt.min_variance_portfolio(d))

    def _tool_max_sharpe(self, *, data: dict, params: dict = None) -> dict:
        d = _S.MarketDataSchema.model_validate(data).to_portopt()
        p = (_S.MVOParametersSchema.model_validate(params).to_portopt()
             if params is not None else None)
        opt = MVOptimizer(p) if p is not None else MVOptimizer()
        return _result_to_dict(opt.max_sharpe_portfolio(d))

    def _tool_target_volatility(self, *, data: dict, target: float,
                                 params: dict = None) -> dict:
        d = _S.MarketDataSchema.model_validate(data).to_portopt()
        p = (_S.MVOParametersSchema.model_validate(params).to_portopt()
             if params is not None else None)
        opt = MVOptimizer(p) if p is not None else MVOptimizer()
        return _result_to_dict(opt.optimize_for_target_volatility(d, target))

    def _tool_target_return(self, *, data: dict, target: float,
                             params: dict = None) -> dict:
        d = _S.MarketDataSchema.model_validate(data).to_portopt()
        p = (_S.MVOParametersSchema.model_validate(params).to_portopt()
             if params is not None else None)
        opt = MVOptimizer(p) if p is not None else MVOptimizer()
        return _result_to_dict(opt.optimize_for_target_return(d, target))

    def _tool_build_efficient_frontier(self, *, data: dict,
                                        params: dict = None) -> dict:
        d = _S.MarketDataSchema.model_validate(data).to_portopt()
        p = (_S.MVOParametersSchema.model_validate(params).to_portopt()
             if params is not None else None)
        opt = MVOptimizer(p) if p is not None else MVOptimizer()
        return _frontier_to_dict(opt.efficient_frontier(d))

    def _tool_estimate_covariance(self, *,
                                   tickers: list,
                                   returns: list,
                                   periods_per_year: float = 252.0,
                                   shrinkage: str = "none") -> dict:
        import numpy as np
        R = np.asarray(returns, dtype=float)
        d = estimation.from_returns(
            list(tickers), R, periods_per_year=periods_per_year,
            shrinkage=shrinkage)
        return {
            "tickers": [a.ticker for a in d.assets],
            "expected_returns": list(d.expected_returns),
            "covariance":  [list(row) for row in d.covariance],
            "shrinkage":   shrinkage,
        }

    def _tool_summarise_portfolio(self, *, data: dict,
                                   weights: list,
                                   risk_free_rate: float = 0.0) -> dict:
        import numpy as np
        d = _S.MarketDataSchema.model_validate(data).to_portopt()
        w = np.asarray(weights, dtype=float)
        m = MVOptimizer.compute_metrics(w, d.expected_returns, d.covariance,
                                         risk_free_rate)
        if d.benchmark_weights is not None:
            MVOptimizer.augment_benchmark_metrics(m, w, d.expected_returns,
                                                    d.covariance,
                                                    d.benchmark_weights,
                                                    risk_free_rate)
        return {
            "expected_return":       float(m.expected_return),
            "volatility":            float(m.volatility),
            "sharpe_ratio":          float(m.sharpe_ratio),
            "variance":              float(m.variance),
            "risk_contribution":     list(m.risk_contribution),
            "diversification_ratio": float(m.diversification_ratio),
            "effective_n_assets":    float(m.effective_n_assets),
            "tracking_error":        float(m.tracking_error),
            "information_ratio":     float(m.information_ratio),
            "active_share":          float(m.active_share),
            "beta_to_benchmark":     float(m.beta_to_benchmark),
        }


def serve_stdio() -> None:                                  # pragma: no cover
    """Trivial line-protocol server for ad-hoc testing.

    Reads one JSON request per line on stdin:
        {"tool": "...", "args": { ... }}
    Writes one JSON response per line on stdout. Useful as a smoke-test
    harness; production MCP wiring should use the official MCP SDK and
    delegate to PortoptToolServer.
    """
    import sys
    server = PortoptToolServer()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"error": {"code": "bad_json",
                                         "message": str(e)}}))
            continue
        if req.get("tool") == "list_tools":
            print(json.dumps({"data": server.list_tools()}))
            continue
        resp = server.call(req.get("tool", ""), **req.get("args", {}))
        print(json.dumps(resp))
        sys.stdout.flush()
