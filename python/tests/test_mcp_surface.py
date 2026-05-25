"""
Tests for the MCP-connector surface (Track C):

* C1: dict ↔ struct helpers (mvo_params_from_dict, market_data_from_dict, …)
* C2: JSON-string parameter readers (no temp files)
* C8: pandas-free EfficientFrontier.to_records()
* C9: summary-mode JSON output for large-n payloads
"""

import json
import numpy as np
import pytest

import portopt


def _three_asset_data():
    md = portopt.market_data_from_dict({
        "assets": [{"ticker": "A"}, {"ticker": "B"}, {"ticker": "C"}],
        "covariance": [
            [0.04, 0.01, 0.00],
            [0.01, 0.09, 0.00],
            [0.00, 0.00, 0.01],
        ],
        "risk_free_rate": 0.02,
    })
    md.expected_returns = np.array([0.10, 0.12, 0.08])
    md.market_weights = np.array([0.5, 0.3, 0.2])
    return md


def test_mvo_params_from_dict_round_trip():
    spec = {
        "mvo": {
            "risk_aversion": 2.5,
            "constraints": {
                "lower_bounds": [0.0, 0.0, 0.0],
                "upper_bounds": [0.4, 0.4, 0.4],
                "budget": 1.0,
            },
        }
    }
    p = portopt.mvo_params_from_dict(spec)
    assert p.risk_aversion == 2.5
    assert list(p.constraints.upper_bounds) == [0.4, 0.4, 0.4]
    assert p.constraints.budget == 1.0

    # to_dict round-trip is structurally equivalent
    back = portopt.mvo_params_to_dict(p)
    assert back["risk_aversion"] == 2.5
    assert back["constraints"]["upper_bounds"] == [0.4, 0.4, 0.4]


def test_bl_params_from_dict():
    spec = {
        "black_litterman": {
            "tau": 0.05,
            "risk_aversion": 2.5,
            "confidence_mode": "idzorek",
            "views": [
                {
                    "description":     "A outperforms B by 3%",
                    "pick_vector":     [1.0, -1.0, 0.0],
                    "expected_return": 0.03,
                    "confidence":      0.65,
                }
            ],
        }
    }
    p = portopt.bl_params_from_dict(spec)
    assert p.tau == 0.05
    assert p.confidence_mode == portopt.ViewConfidenceMode.Idzorek
    assert len(p.views) == 1
    assert p.views[0].expected_return == 0.03


def test_market_data_from_dict_drives_full_optimisation():
    md = _three_asset_data()
    params = portopt.mvo_params_from_dict({
        "mvo": {"risk_aversion": 2.0},
    })
    params.constraints = portopt.PortfolioConstraints.long_only(3)

    result = portopt.MVOptimizer(params).optimize(md)
    assert result.converged
    assert abs(sum(result.weights) - 1.0) < 1e-6
    # Audit trail
    assert result.library_version
    assert result.input_hash
    assert result.params_hash


def test_frontier_to_records_no_pandas():
    md = _three_asset_data()
    params = portopt.MVOParameters()
    params.constraints = portopt.PortfolioConstraints.long_only(3)
    params.frontier_points = 10

    frontier = portopt.MVOptimizer(params).efficient_frontier(md)
    records = frontier.to_records()
    assert len(records) == 10
    first = records[0]
    assert "risk_aversion" in first
    assert "expected_return" in first
    assert "weights" in first
    assert set(first["weights"].keys()) == {"A", "B", "C"}
    # JSON-serialisable — no pandas / numpy in there
    json.dumps(records, default=lambda o: float(o))


def test_bl_json_summary_omits_large_matrices():
    md = _three_asset_data()
    params = portopt.bl_params_from_dict({
        "black_litterman": {
            "tau": 0.05,
            "risk_aversion": 2.5,
            "views": [
                {"pick_vector": [1.0, -1.0, 0.0],
                 "expected_return": 0.03,
                 "confidence": 0.01},
            ],
        }
    })
    params.mvo_params.constraints = portopt.PortfolioConstraints.long_only(3)

    bl = portopt.BlackLittermanOptimizer(params)
    model = bl.model_output(md)

    full    = portopt.bl_model_to_json(model, md.assets, indent=2, summary=False)
    summary = portopt.bl_model_to_json(model, md.assets, indent=2, summary=True)

    assert "posterior_cov" in full
    assert "blended_cov"   in full
    assert "posterior_cov" not in summary
    assert "blended_cov"   not in summary
    assert "matrices_omitted" in summary
    # Summary still carries the diagnostics block.
    summary_obj = json.loads(summary)
    assert "diagnostics" in summary_obj
    assert summary_obj["diagnostics"]["pick_matrix_rank"] >= 1


def test_in_memory_param_readers_skip_temp_files():
    p1 = portopt.read_mvo_parameters_json(
        json.dumps({"mvo": {"risk_aversion": 4.0}})
    )
    assert p1.risk_aversion == 4.0
    p2 = portopt.read_bl_parameters_json(
        json.dumps({"black_litterman": {"tau": 0.1}})
    )
    assert p2.tau == 0.1


def test_portfolios_module_round_trip():
    md = _three_asset_data()
    erc = portopt.portfolios.equal_risk_contribution(md.covariance)
    assert abs(sum(erc) - 1.0) < 1e-9

    ew = portopt.portfolios.equal_weight(3)
    assert all(abs(w - 1/3) < 1e-12 for w in ew)

    iv = portopt.portfolios.inverse_variance(md.covariance)
    assert iv[2] > iv[0] > iv[1]   # variance ordering: C(0.01) < A(0.04) < B(0.09)
