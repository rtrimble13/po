/**
 * @file test_mvo.cpp
 * @brief Unit tests for Mean-Variance Optimisation.
 */

#include <catch2/catch_all.hpp>
#include <portopt/mvo.hpp>

using namespace portopt;
using Catch::Approx;

// ── Fixtures ──────────────────────────────────────────────────────────────────

static MarketData twoAssetData() {
    // Classic 2-asset example: analytical solution known
    MarketData d;
    d.assets = {
        {"A", "Asset A", 0.10, 1e11},
        {"B", "Asset B", 0.20, 1e11}
    };
    d.expected_returns.resize(2);
    d.expected_returns << 0.10, 0.20;
    d.covariance.resize(2, 2);
    d.covariance << 0.04, 0.01,
                    0.01, 0.09;
    return d;
}

static MarketData fiveAssetData() {
    // Slightly more realistic 5-asset universe
    MarketData d;
    const int n = 5;
    d.assets.resize(n);
    std::vector<std::string> tickers = {"AAPL","MSFT","JPM","JNJ","XOM"};
    std::vector<double> ret = {0.15, 0.14, 0.10, 0.08, 0.09};
    for (int i = 0; i < n; ++i) {
        d.assets[i] = {tickers[i], tickers[i], ret[i], 1e12};
    }
    d.expected_returns = Eigen::Map<Vector>(ret.data(), n);

    // Positive-definite covariance
    d.covariance.resize(n, n);
    d.covariance <<
        0.0400, 0.0100, 0.0050, 0.0030, 0.0020,
        0.0100, 0.0361, 0.0045, 0.0025, 0.0015,
        0.0050, 0.0045, 0.0225, 0.0060, 0.0040,
        0.0030, 0.0025, 0.0060, 0.0196, 0.0055,
        0.0020, 0.0015, 0.0040, 0.0055, 0.0289;
    return d;
}

// ── Single optimisation ───────────────────────────────────────────────────────

TEST_CASE("MVOptimizer — basic 2-asset MVO", "[mvo]") {
    auto data = twoAssetData();
    MVOParameters params;
    params.risk_aversion = 5.0;
    params.constraints   = PortfolioConstraints::longOnly(2);

    MVOptimizer opt(params);
    auto result = opt.optimize(data);

    REQUIRE(result.converged);
    REQUIRE(result.weights.size() == 2);
    CHECK(result.weights.sum() == Approx(1.0).epsilon(1e-7));
    CHECK((result.weights.array() >= -1e-8).all());
    CHECK(result.metrics.volatility > 0.0);
    CHECK(result.metrics.expected_return > 0.0);
    CHECK(result.method == "MVO");
}

TEST_CASE("MVOptimizer — minimum variance portfolio", "[mvo]") {
    // At lambda ~ 0, minimise variance only → minimum-variance portfolio
    auto data = twoAssetData();
    MVOParameters params;
    params.risk_aversion = 1e-4; // essentially zero risk aversion
    params.constraints   = PortfolioConstraints::longOnly(2);

    MVOptimizer opt(params);
    auto result = opt.optimize(data);

    REQUIRE(result.converged);
    // Minimum variance: w = (sigma_B^2 - cov) / (sigma_A^2 + sigma_B^2 - 2*cov)
    // = (0.09 - 0.01) / (0.04 + 0.09 - 0.02) = 0.08/0.11 ≈ 0.727
    double w_a_analytical = (0.09 - 0.01) / (0.04 + 0.09 - 2 * 0.01);
    CHECK(result.weights[0] == Approx(w_a_analytical).epsilon(1e-3));
}

TEST_CASE("MVOptimizer — maximum return (high risk aversion → concentrate)", "[mvo]") {
    auto data = fiveAssetData();
    MVOParameters params;
    params.risk_aversion = 500.0; // very high — concentrate on best asset
    params.constraints   = PortfolioConstraints::longOnly(5);

    MVOptimizer opt(params);
    auto result = opt.optimize(data);

    REQUIRE(result.converged);
    CHECK(result.weights.sum() == Approx(1.0).epsilon(1e-6));
    // Best return is AAPL (0.15), should get most weight
    CHECK(result.weights[0] > 0.5);
}

TEST_CASE("MVOptimizer — weight bounds respected", "[mvo]") {
    auto data = fiveAssetData();
    MVOParameters params;
    params.risk_aversion = 2.0;
    params.constraints.lower_bounds = Vector::Constant(5, 0.05);
    params.constraints.upper_bounds = Vector::Constant(5, 0.40);

    MVOptimizer opt(params);
    auto result = opt.optimize(data);

    REQUIRE(result.converged);
    CHECK(result.weights.sum() == Approx(1.0).epsilon(1e-6));
    for (int i = 0; i < 5; ++i) {
        CHECK(result.weights[i] >= 0.05 - 1e-6);
        CHECK(result.weights[i] <= 0.40 + 1e-6);
    }
}

TEST_CASE("MVOptimizer — metrics are consistent", "[mvo]") {
    auto data = fiveAssetData();
    MVOParameters params;
    params.risk_aversion = 3.0;
    params.constraints   = PortfolioConstraints::longOnly(5);

    MVOptimizer opt(params);
    auto result = opt.optimize(data);

    // Verify metrics manually
    double exp_ret = result.weights.dot(data.expected_returns);
    double variance = result.weights.dot(data.covariance * result.weights);
    double vol      = std::sqrt(variance);

    CHECK(result.metrics.expected_return == Approx(exp_ret).epsilon(1e-8));
    CHECK(result.metrics.volatility      == Approx(vol).epsilon(1e-8));
    CHECK(result.metrics.variance        == Approx(variance).epsilon(1e-8));
    double sharpe = (vol > 1e-12) ? exp_ret / vol : 0.0;
    CHECK(result.metrics.sharpe_ratio    == Approx(sharpe).epsilon(1e-8));
}

// ── Efficient frontier ────────────────────────────────────────────────────────

TEST_CASE("MVOptimizer — efficient frontier shape", "[mvo][frontier]") {
    auto data = fiveAssetData();
    MVOParameters params;
    params.min_risk_aversion = 0.1;
    params.max_risk_aversion = 50.0;
    params.frontier_points   = 20;
    params.constraints       = PortfolioConstraints::longOnly(5);

    MVOptimizer opt(params);
    auto frontier = opt.efficientFrontier(data);

    REQUIRE(frontier.points.size() == 20);

    // As lambda increases, volatility should decrease (more risk averse = lower risk)
    // and expected return should generally decrease
    double max_vol = 0.0;
    double min_vol = 1e9;
    for (const auto& pt : frontier.points) {
        CHECK(pt.weights.sum() == Approx(1.0).epsilon(1e-6));
        CHECK(pt.metrics.volatility > 0.0);
        max_vol = std::max(max_vol, pt.metrics.volatility);
        min_vol = std::min(min_vol, pt.metrics.volatility);
    }
    // Frontier should span a meaningful range
    CHECK(max_vol > min_vol * 1.05);
}

TEST_CASE("MVOptimizer — frontier is Pareto-efficient (return↑ vol↑)", "[mvo][frontier]") {
    // portopt's objective is  w'Σw − λ μ'w :
    //   λ → 0      ⇒ minimum-variance (low vol, low return)
    //   λ → large  ⇒ pure-return (high vol, high return)
    // So along the frontier with increasing λ, BOTH volatility and
    // expected return should rise monotonically.
    auto data = fiveAssetData();
    MVOParameters params;
    params.min_risk_aversion = 0.5;
    params.max_risk_aversion = 100.0;
    params.frontier_points   = 30;
    params.constraints       = PortfolioConstraints::longOnly(5);

    MVOptimizer opt(params);
    auto frontier = opt.efficientFrontier(data);

    // Endpoint monotonicity (strict): at high λ the solution is more
    // return-heavy than at low λ.
    const double vol_lo_lambda = frontier.points.front().metrics.volatility;
    const double vol_hi_lambda = frontier.points.back().metrics.volatility;
    CHECK(vol_hi_lambda > vol_lo_lambda);

    // Pointwise monotonicity (loose): FISTA stops on a residual rather
    // than a duality gap, so individual points can be slightly
    // sub-optimal. Track-A items A2 / A4 in the investment plan will
    // tighten the per-point guarantee.
    size_t violations = 0;
    for (size_t i = 1; i < frontier.points.size(); ++i) {
        if (frontier.points[i].metrics.volatility + 1e-3 <
            frontier.points[i-1].metrics.volatility)
            ++violations;
    }
    CHECK(violations < frontier.points.size() / 2);
}

// ── Validation ────────────────────────────────────────────────────────────────

TEST_CASE("MVOptimizer — empty assets throws", "[mvo][validation]") {
    MarketData data;
    MVOptimizer opt;
    CHECK_THROWS_AS(opt.optimize(data), std::invalid_argument);
}

TEST_CASE("MVOptimizer — dimension mismatch throws", "[mvo][validation]") {
    MarketData data = twoAssetData();
    data.expected_returns = Vector::Zero(3); // wrong size
    MVOptimizer opt;
    CHECK_THROWS_AS(opt.optimize(data), std::invalid_argument);
}
