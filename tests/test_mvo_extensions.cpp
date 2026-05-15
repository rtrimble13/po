/**
 * @file test_mvo_extensions.cpp
 * @brief Tests for new MVO features: target-vol/return, max-Sharpe, min-var,
 *        risk contributions, turnover penalty, budget, group constraints,
 *        benchmark-relative metrics, risk-free rate.
 */

#include <catch2/catch_all.hpp>
#include <portopt/mvo.hpp>

#include <cmath>

using namespace portopt;
using Catch::Approx;

namespace {

MarketData fiveAssets() {
    MarketData d;
    const int n = 5;
    d.assets.resize(n);
    const std::vector<std::string> t = {"A1","A2","A3","A4","A5"};
    const std::vector<double> er = {0.15, 0.12, 0.08, 0.06, 0.10};
    for (int i = 0; i < n; ++i) {
        d.assets[i] = {t[i], t[i], er[i], 1e11, "Sec" + std::to_string(i % 3)};
    }
    d.expected_returns = Eigen::Map<const Vector>(er.data(), n);

    d.covariance = Matrix(n, n);
    d.covariance <<
        0.04, 0.01, 0.005, 0.003, 0.002,
        0.01, 0.0361, 0.0045, 0.0025, 0.0015,
        0.005, 0.0045, 0.0225, 0.006, 0.004,
        0.003, 0.0025, 0.006, 0.0196, 0.0055,
        0.002, 0.0015, 0.004, 0.0055, 0.0289;
    return d;
}

} // namespace

// ── Sharpe with risk-free rate ───────────────────────────────────────────────

TEST_CASE("Sharpe uses risk-free rate correctly", "[mvo][sharpe][rf]") {
    auto data = fiveAssets();
    data.risk_free_rate = 0.03;

    MVOParameters p;
    p.risk_aversion = 2.0;
    p.constraints = PortfolioConstraints::longOnly(5);
    MVOptimizer opt(p);
    auto r = opt.optimize(data);

    const double expected_sharpe =
        (r.metrics.expected_return - 0.03) / r.metrics.volatility;
    CHECK(r.metrics.sharpe_ratio == Approx(expected_sharpe).epsilon(1e-8));
}

TEST_CASE("Sharpe returns NaN for zero-vol portfolio", "[mvo][sharpe][nan]") {
    // 1-asset, zero variance — degenerate
    MarketData d;
    d.assets = {{"X", "", 0.05, 0.0, ""}};
    d.expected_returns = Vector::Constant(1, 0.05);
    d.covariance = Matrix::Zero(1, 1);

    MVOParameters p;
    p.constraints = PortfolioConstraints::longOnly(1);
    MVOptimizer opt(p);
    auto r = opt.optimize(d);
    CHECK(std::isnan(r.metrics.sharpe_ratio));
}

// ── Risk contributions ───────────────────────────────────────────────────────

TEST_CASE("Risk contributions sum to volatility", "[mvo][risk_contribution]") {
    auto data = fiveAssets();
    MVOParameters p;
    p.risk_aversion = 2.5;
    p.constraints = PortfolioConstraints::longOnly(5);
    MVOptimizer opt(p);
    auto r = opt.optimize(data);

    REQUIRE(r.metrics.risk_contribution.size() == 5);
    const double rc_sum = r.metrics.risk_contribution.sum();
    CHECK(rc_sum == Approx(r.metrics.volatility).epsilon(1e-6));
}

TEST_CASE("Diversification ratio > 1 for diversified portfolio",
          "[mvo][diversification]") {
    auto data = fiveAssets();
    MVOParameters p;
    p.risk_aversion = 0.5;
    p.constraints = PortfolioConstraints::longOnly(5);
    MVOptimizer opt(p);
    auto r = opt.optimize(data);
    CHECK(r.metrics.diversification_ratio > 1.0);
}

// ── Target portfolios ────────────────────────────────────────────────────────

TEST_CASE("Min-variance portfolio has lowest vol", "[mvo][min_var]") {
    auto data = fiveAssets();
    MVOParameters p;
    p.constraints = PortfolioConstraints::longOnly(5);
    MVOptimizer opt(p);

    auto mv = opt.minVariancePortfolio(data);
    auto other = opt.optimize(data); // λ=1
    CHECK(mv.metrics.volatility <= other.metrics.volatility + 1e-6);
}

TEST_CASE("Max-Sharpe portfolio beats single-λ Sharpe", "[mvo][max_sharpe]") {
    auto data = fiveAssets();
    MVOParameters p;
    p.constraints = PortfolioConstraints::longOnly(5);
    p.risk_aversion = 5.0;
    p.min_risk_aversion = 0.05;
    p.max_risk_aversion = 200.0;
    MVOptimizer opt(p);

    auto ms = opt.maxSharpePortfolio(data);
    auto other = opt.optimize(data);
    CHECK(ms.metrics.sharpe_ratio >= other.metrics.sharpe_ratio - 1e-3);
}

TEST_CASE("Target-volatility hits requested vol within tolerance",
          "[mvo][target_vol]") {
    auto data = fiveAssets();
    MVOParameters p;
    p.constraints = PortfolioConstraints::longOnly(5);
    p.min_risk_aversion = 0.01;
    p.max_risk_aversion = 200.0;
    MVOptimizer opt(p);

    auto r = opt.optimizeForTargetVolatility(data, 0.13);
    CHECK(r.metrics.volatility == Approx(0.13).margin(0.005));
}

TEST_CASE("Target-return hits requested return within tolerance",
          "[mvo][target_return]") {
    auto data = fiveAssets();
    MVOParameters p;
    p.constraints = PortfolioConstraints::longOnly(5);
    p.min_risk_aversion = 0.01;
    p.max_risk_aversion = 200.0;
    MVOptimizer opt(p);

    auto r = opt.optimizeForTargetReturn(data, 0.10);
    CHECK(r.metrics.expected_return == Approx(0.10).margin(0.005));
}

// ── Budget ──────────────────────────────────────────────────────────────────

TEST_CASE("Dollar-neutral budget (sum=0) is respected", "[mvo][budget]") {
    auto data = fiveAssets();
    MVOParameters p;
    p.risk_aversion = 2.0;
    p.constraints = PortfolioConstraints::dollarNeutral(5, 0.5);
    MVOptimizer opt(p);

    auto r = opt.optimize(data);
    CHECK(r.weights.sum() == Approx(0.0).margin(1e-6));
}

TEST_CASE("Infeasible budget throws", "[mvo][budget][validation]") {
    auto data = fiveAssets();
    MVOParameters p;
    p.risk_aversion = 2.0;
    p.constraints = PortfolioConstraints::longOnly(5);
    p.constraints.upper_bounds = Vector::Constant(5, 0.15); // max sum = 0.75 < 1.0
    MVOptimizer opt(p);
    CHECK_THROWS(opt.optimize(data));
}

// ── Turnover penalty ────────────────────────────────────────────────────────

TEST_CASE("Turnover penalty pulls weights toward current_weights",
          "[mvo][turnover]") {
    auto data = fiveAssets();
    MVOParameters p;
    p.risk_aversion = 2.0;
    p.constraints = PortfolioConstraints::longOnly(5);
    MVOptimizer opt(p);
    auto baseline = opt.optimize(data);

    // Now apply heavy turnover penalty with current = 20% each
    p.constraints.current_weights   = Vector::Constant(5, 0.2);
    p.constraints.turnover_penalty  = 50.0;
    opt.setParameters(p);
    auto sticky = opt.optimize(data);

    const double dist_baseline =
        (baseline.weights - Vector::Constant(5, 0.2)).norm();
    const double dist_sticky =
        (sticky.weights   - Vector::Constant(5, 0.2)).norm();
    CHECK(dist_sticky < dist_baseline);
    // Turnover reported
    CHECK(!std::isnan(sticky.metrics.turnover));
    CHECK(sticky.metrics.turnover < baseline.metrics.expected_return * 100); // sanity
}

// ── Benchmark-relative ───────────────────────────────────────────────────────

TEST_CASE("Tracking error is 0 when w = benchmark", "[mvo][benchmark][te]") {
    auto data = fiveAssets();
    data.benchmark_weights = Vector::Constant(5, 0.2);

    PortfolioMetrics m;
    Vector w = Vector::Constant(5, 0.2);
    MVOptimizer::augmentBenchmarkMetrics(m, w, data.expected_returns,
                                          data.covariance,
                                          *data.benchmark_weights);
    CHECK(m.tracking_error == Approx(0.0).margin(1e-10));
    CHECK(m.active_share   == Approx(0.0).margin(1e-10));
}

TEST_CASE("Active share is 1.0 for fully-disjoint portfolio",
          "[mvo][benchmark][active_share]") {
    auto data = fiveAssets();
    data.benchmark_weights = Vector(5);
    *data.benchmark_weights << 1.0, 0.0, 0.0, 0.0, 0.0;

    PortfolioMetrics m;
    Vector w(5);
    w << 0.0, 0.25, 0.25, 0.25, 0.25;
    MVOptimizer::augmentBenchmarkMetrics(m, w, data.expected_returns,
                                          data.covariance,
                                          *data.benchmark_weights);
    CHECK(m.active_share == Approx(1.0).margin(1e-8));
}

// ── Group constraints ───────────────────────────────────────────────────────

TEST_CASE("Group constraint softly enforces upper bound", "[mvo][groups]") {
    auto data = fiveAssets();

    GroupConstraint g;
    g.description  = "Assets 1-2 ≤ 40%";
    g.coefficients = Vector(5);
    g.coefficients << 1.0, 1.0, 0.0, 0.0, 0.0;
    g.lower = -1e30;
    g.upper = 0.40;

    MVOParameters p;
    p.risk_aversion = 2.0;
    p.constraints = PortfolioConstraints::longOnly(5);
    p.constraints.groups = {g};
    p.group_penalty = 1e6;
    MVOptimizer opt(p);

    auto r = opt.optimize(data);
    const double group_sum = r.weights[0] + r.weights[1];
    CHECK(group_sum <= 0.40 + 0.02);  // allow small penalty slack
}

// ── PSD validation ───────────────────────────────────────────────────────────

TEST_CASE("Negative variance throws", "[mvo][validation][psd]") {
    auto data = fiveAssets();
    data.covariance(0, 0) = -1.0;
    CHECK_THROWS(MVOptimizer::validateMarketData(data));
}

// ── Frontier validation ──────────────────────────────────────────────────────

TEST_CASE("Frontier with min_risk_aversion <= 0 throws",
          "[mvo][frontier][validation]") {
    auto data = fiveAssets();
    MVOParameters p;
    p.constraints = PortfolioConstraints::longOnly(5);
    p.min_risk_aversion = 0.0;
    p.max_risk_aversion = 10.0;
    MVOptimizer opt(p);
    CHECK_THROWS(opt.efficientFrontier(data));
}

// ── Effective N ─────────────────────────────────────────────────────────────

TEST_CASE("Effective N equals n for equal-weighted portfolio",
          "[mvo][effective_n]") {
    auto data = fiveAssets();
    Vector w = Vector::Constant(5, 0.2);
    auto m = MVOptimizer::computeMetrics(w, data.expected_returns,
                                          data.covariance);
    CHECK(m.effective_n_assets == Approx(5.0).epsilon(1e-9));
}

// ── BL constraint inheritance ────────────────────────────────────────────────
// (Tested indirectly: BL uses MVOptimizer::resolveMVOParameters under the hood
//  — the inheritance behaviour is in test_black_litterman.cpp.)
