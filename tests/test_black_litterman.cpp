/**
 * @file test_black_litterman.cpp
 * @brief Unit tests for the Black-Litterman model.
 */

#include <catch2/catch_all.hpp>
#include <portopt/black_litterman.hpp>

using namespace portopt;
using Catch::Approx;

// ── Fixtures ──────────────────────────────────────────────────────────────────

static MarketData makeThreeAssetData() {
    MarketData d;
    d.assets = {
        {"EQUITY", "Global Equity",  0.06, 6e12},
        {"BOND",   "Gov. Bonds",     0.03, 3e12},
        {"CMDTY",  "Commodities",    0.04, 1e12}
    };
    const int n = 3;
    d.expected_returns.resize(n);
    d.expected_returns << 0.06, 0.03, 0.04;

    d.covariance.resize(n, n);
    d.covariance <<
        0.0400, -0.0050,  0.0100,
       -0.0050,  0.0100, -0.0020,
        0.0100, -0.0020,  0.0225;

    // Market-cap weights
    Vector mw(n);
    mw << 0.60, 0.30, 0.10;
    d.market_weights = mw;
    return d;
}

static BlackLittermanParameters noViewParams() {
    BlackLittermanParameters p;
    p.tau = 0.05;
    p.risk_aversion = 2.5;
    p.mvo_params.constraints = PortfolioConstraints::longOnly(3);
    return p;
}

static BlackLittermanParameters oneViewParams() {
    auto p = noViewParams();
    View v;
    v.description     = "Equity outperforms bonds by 3%";
    v.pick_vector     = Vector(3);
    v.pick_vector << 1.0, -1.0, 0.0; // long equity, short bonds
    v.expected_return = 0.03;
    v.confidence      = 0.001;
    p.views.push_back(v);
    return p;
}

// ── No-views case ─────────────────────────────────────────────────────────────

TEST_CASE("BL — no views: posterior == prior", "[bl][no_views]") {
    auto data  = makeThreeAssetData();
    auto params = noViewParams();

    BlackLittermanOptimizer bl(params);
    auto model = bl.modelOutput(data);

    // With no views, posterior should equal prior (π)
    CHECK(model.posterior_returns.isApprox(model.prior_returns, 1e-9));
    CHECK(model.view_returns.size() == 0);
}

TEST_CASE("BL — no views: prior recovers market weights (δΣw = π)", "[bl][prior]") {
    auto data  = makeThreeAssetData();
    auto params = noViewParams();

    BlackLittermanOptimizer bl(params);
    auto model = bl.modelOutput(data);

    // Verify π = δ Σ w_mkt
    Vector pi_expected = params.risk_aversion *
                         data.covariance * *data.market_weights;
    CHECK(model.prior_returns.isApprox(pi_expected, 1e-10));
}

// ── Single view ───────────────────────────────────────────────────────────────

TEST_CASE("BL — single view shifts posterior toward view", "[bl][view]") {
    auto data   = makeThreeAssetData();
    auto params = oneViewParams();

    BlackLittermanOptimizer bl(params);
    auto model = bl.modelOutput(data);

    // View says equity outperforms bonds — posterior equity return should
    // be higher relative to bonds than the prior
    double prior_diff    = model.prior_returns[0]    - model.prior_returns[1];
    double post_diff     = model.posterior_returns[0] - model.posterior_returns[1];
    CHECK(post_diff > prior_diff - 1e-6);
}

TEST_CASE("BL — high confidence view pulls posterior strongly", "[bl][view]") {
    auto data = makeThreeAssetData();

    // Very confident view: equity return = 10% (much higher than prior)
    BlackLittermanParameters p = noViewParams();
    View v;
    v.description     = "Strong equity view";
    v.pick_vector     = Vector::Zero(3);
    v.pick_vector[0]  = 1.0;
    v.expected_return = 0.10;
    v.confidence      = 1e-8; // extremely confident
    p.views.push_back(v);

    BlackLittermanOptimizer bl(p);
    auto model = bl.modelOutput(data);

    // Very high confidence → posterior approaches view
    CHECK(model.posterior_returns[0] == Approx(0.10).epsilon(0.01));
}

TEST_CASE("BL — zero confidence throws", "[bl][validation]") {
    auto data = makeThreeAssetData();
    BlackLittermanParameters p = noViewParams();
    View v;
    v.pick_vector = Vector::Zero(3);
    v.expected_return = 0.03;
    v.confidence = 0.0; // invalid
    p.views.push_back(v);

    BlackLittermanOptimizer bl(p);
    CHECK_THROWS_AS(bl.optimize(data), std::invalid_argument);
}

// ── Optimisation ─────────────────────────────────────────────────────────────

TEST_CASE("BL — optimize returns valid portfolio", "[bl][optimize]") {
    auto data   = makeThreeAssetData();
    auto params = oneViewParams();

    BlackLittermanOptimizer bl(params);
    auto result = bl.optimize(data);

    REQUIRE(result.converged);
    CHECK(result.weights.sum() == Approx(1.0).epsilon(1e-6));
    CHECK((result.weights.array() >= -1e-8).all());
    CHECK(result.method == "Black-Litterman");
    CHECK(result.metrics.volatility > 0.0);
}

TEST_CASE("BL — efficient frontier has correct shape", "[bl][frontier]") {
    auto data   = makeThreeAssetData();
    auto params = oneViewParams();
    params.mvo_params.frontier_points = 15;

    BlackLittermanOptimizer bl(params);
    auto frontier = bl.efficientFrontier(data);

    REQUIRE(frontier.points.size() == 15);
    CHECK(frontier.method == "Black-Litterman");
    for (const auto& pt : frontier.points) {
        CHECK(pt.weights.sum() == Approx(1.0).epsilon(1e-6));
        CHECK(pt.metrics.volatility > 0.0);
    }
}

// ── Validation ────────────────────────────────────────────────────────────────

TEST_CASE("BL — missing market weights throws", "[bl][validation]") {
    MarketData data = makeThreeAssetData();
    data.market_weights = std::nullopt;

    BlackLittermanOptimizer bl(noViewParams());
    CHECK_THROWS_AS(bl.optimize(data), std::invalid_argument);
}

TEST_CASE("BL — wrong market weights sum throws", "[bl][validation]") {
    MarketData data = makeThreeAssetData();
    *data.market_weights *= 2.0; // sums to 2, not 1

    BlackLittermanOptimizer bl(noViewParams());
    CHECK_THROWS_AS(bl.optimize(data), std::invalid_argument);
}

TEST_CASE("BL — pick vector dimension mismatch throws", "[bl][validation]") {
    MarketData data = makeThreeAssetData();
    BlackLittermanParameters p = noViewParams();
    View v;
    v.pick_vector = Vector::Zero(5); // wrong dimension
    v.expected_return = 0.02;
    v.confidence = 0.001;
    p.views.push_back(v);

    BlackLittermanOptimizer bl(p);
    CHECK_THROWS_AS(bl.optimize(data), std::invalid_argument);
}

TEST_CASE("BL — BL result is different from plain MVO (views matter)", "[bl][mvo_compare]") {
    auto data = makeThreeAssetData();

    MVOParameters mvo_params;
    mvo_params.risk_aversion = 2.5;
    mvo_params.constraints   = PortfolioConstraints::longOnly(3);

    MVOptimizer mvo(mvo_params);
    auto mvo_result = mvo.optimize(data);

    BlackLittermanOptimizer bl(oneViewParams());
    auto bl_result = bl.optimize(data);

    // With a meaningful view, BL weights should differ from plain MVO
    bool any_different = false;
    for (int i = 0; i < 3; ++i) {
        if (std::abs(bl_result.weights[i] - mvo_result.weights[i]) > 1e-4)
            any_different = true;
    }
    CHECK(any_different);
}
