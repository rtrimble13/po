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

// ── A1: Risk-free-rate override semantics ────────────────────────────────────

TEST_CASE("MVOParameters.risk_free_rate overrides MarketData.risk_free_rate",
          "[mvo][risk_free_rate]") {
    auto data = fiveAssets();
    data.risk_free_rate = 0.02;

    MVOParameters p_no_override;
    p_no_override.constraints   = PortfolioConstraints::longOnly(5);
    p_no_override.risk_aversion = 2.0;
    auto r_data_rf = MVOptimizer(p_no_override).optimize(data);

    MVOParameters p_override = p_no_override;
    p_override.risk_free_rate = 0.05;
    auto r_param_rf = MVOptimizer(p_override).optimize(data);

    // The two Sharpe ratios should differ because rf differs.
    const double sh_d = r_data_rf.metrics.sharpe_ratio;
    const double sh_p = r_param_rf.metrics.sharpe_ratio;
    CHECK(std::abs(sh_d - sh_p) > 1e-6);

    // And the param override should give EXACTLY the value computed with
    // rf = 0.05 — i.e., not 0.02 + 0.05 = 0.07 (the old additive bug).
    const double expected =
        (r_param_rf.metrics.expected_return - 0.05) /
        r_param_rf.metrics.volatility;
    CHECK(sh_p == Approx(expected).margin(1e-9));
}

// ── A2: KKT residual is exposed on OptimizationResult ─────────────────────────

TEST_CASE("OptimizationResult exposes KKT residual",
          "[mvo][kkt]") {
    auto data = fiveAssets();
    MVOParameters p;
    p.constraints   = PortfolioConstraints::longOnly(5);
    p.risk_aversion = 2.0;
    auto r = MVOptimizer(p).optimize(data);
    REQUIRE(r.converged);
    CHECK(std::isfinite(r.kkt_residual));
    CHECK(std::isfinite(r.dual_estimate));
    CHECK(r.kkt_residual < 1e-3);
}

// ── A4: Target-volatility / target-return out-of-range handling ──────────────

TEST_CASE("Target volatility out of range returns nearest feasible portfolio",
          "[mvo][target_vol]") {
    auto data = fiveAssets();
    MVOParameters p;
    p.constraints = PortfolioConstraints::longOnly(5);
    p.min_risk_aversion = 0.5;
    p.max_risk_aversion = 100.0;

    MVOptimizer opt(p);

    // A wildly large target — should boundary-return with a status message.
    auto r = opt.optimizeForTargetVolatility(data, 10.0);
    CHECK(r.converged);
    CHECK(r.status_message.find("exceeds max") != std::string::npos);

    // A wildly small target — same.
    auto r2 = opt.optimizeForTargetVolatility(data, 0.001);
    CHECK(r2.converged);
    CHECK(r2.status_message.find("below min") != std::string::npos);
}

// ── A10: Singular Σ (perfectly-correlated assets) ────────────────────────────

TEST_CASE("MVO with rank-1 covariance still produces a sensible portfolio",
          "[mvo][singular_cov]") {
    MarketData d;
    const int n = 3;
    d.assets.resize(n);
    d.assets[0] = {"X", "X", 0.10, 1e11};
    d.assets[1] = {"Y", "Y", 0.12, 1e11};
    d.assets[2] = {"Z", "Z", 0.08, 1e11};
    d.expected_returns = Vector(n);
    d.expected_returns << 0.10, 0.12, 0.08;
    // Rank-1: Σ = σ² · vv'  with v = [1, 1, 1]'
    Vector v = Vector::Ones(n);
    d.covariance = 0.04 * v * v.transpose();
    // Add tiny ridge so LDLT succeeds and the QP is well-posed.
    d.covariance.diagonal().array() += 1e-8;

    MVOParameters p;
    p.constraints   = PortfolioConstraints::longOnly(n);
    p.risk_aversion = 5.0;
    MVOptimizer opt(p);
    auto r = opt.optimize(d);
    CHECK(r.converged);
    CHECK(r.weights.sum() == Approx(1.0).margin(1e-6));
    CHECK((r.weights.array() >= -1e-8).all());
}

// ── B16: Audit trail (version + input/params hash) ───────────────────────────

TEST_CASE("OptimizationResult is stamped with library version and hashes",
          "[mvo][audit][b16]") {
    auto data = fiveAssets();
    MVOParameters p;
    p.constraints   = PortfolioConstraints::longOnly(5);
    p.risk_aversion = 2.0;

    auto r1 = MVOptimizer(p).optimize(data);
    auto r2 = MVOptimizer(p).optimize(data);

    CHECK(!r1.library_version.empty());
    CHECK(!r1.input_hash.empty());
    CHECK(!r1.params_hash.empty());
    // Repeatable for identical input / params.
    CHECK(r1.input_hash  == r2.input_hash);
    CHECK(r1.params_hash == r2.params_hash);

    // Hash changes when params change.
    p.risk_aversion = 3.0;
    auto r3 = MVOptimizer(p).optimize(data);
    CHECK(r3.params_hash != r1.params_hash);
    CHECK(r3.input_hash  == r1.input_hash);

    // Hash changes when input data changes.
    data.expected_returns[0] += 0.001;
    auto r4 = MVOptimizer(p).optimize(data);
    CHECK(r4.input_hash != r3.input_hash);
}

// ── A10: Tightly-binding bounds (lb = ub for several assets) ─────────────────

TEST_CASE("MVO honours lb = ub (pinned positions)", "[mvo][tight_bounds]") {
    auto data = fiveAssets();
    MVOParameters p;
    p.constraints   = PortfolioConstraints::longOnly(5);
    p.risk_aversion = 2.0;
    // Pin asset 0 at 0.30, asset 1 at 0.10.
    p.constraints.fixWeight(0, 0.30);
    p.constraints.fixWeight(1, 0.10);

    MVOptimizer opt(p);
    auto r = opt.optimize(data);
    REQUIRE(r.converged);
    CHECK(r.weights[0] == Approx(0.30).margin(1e-6));
    CHECK(r.weights[1] == Approx(0.10).margin(1e-6));
    CHECK(r.weights.sum() == Approx(1.0).margin(1e-6));
}

// ── A3: max-Sharpe analytical tangent fast path ──────────────────────────────

TEST_CASE("Max-Sharpe analytical tangent matches closed form (no bounds binding)",
          "[mvo][max_sharpe][a3]") {
    // Hand-built 2-asset problem with a known closed-form tangent.
    MarketData d;
    d.assets = { {"A", "A", 0.10, 1e11, ""}, {"B", "B", 0.05, 1e11, ""} };
    d.expected_returns = Vector(2); d.expected_returns << 0.10, 0.05;
    d.covariance = Matrix(2, 2);
    d.covariance << 0.04, 0.01, 0.01, 0.02;
    d.risk_free_rate = 0.02;

    // Closed-form tangent: y = Σ⁻¹(μ − rf), w = y / sum(y)
    const Vector excess = d.expected_returns -
                          Vector::Constant(2, d.risk_free_rate);
    Vector y_ref = d.covariance.ldlt().solve(excess);
    Vector w_ref = y_ref / y_ref.sum();

    MVOParameters p;
    p.constraints = PortfolioConstraints::longOnly(2);
    MVOptimizer opt(p);
    auto r = opt.maxSharpePortfolio(d);
    REQUIRE(r.converged);
    // Both weights should match closed form
    CHECK(r.weights[0] == Approx(w_ref[0]).margin(1e-6));
    CHECK(r.weights[1] == Approx(w_ref[1]).margin(1e-6));
    // Status message should call out the analytical path
    CHECK(r.status_message.find("analytical") != std::string::npos);
}

TEST_CASE("Max-Sharpe falls back to heuristic when bounds bind",
          "[mvo][max_sharpe][a3]") {
    auto data = fiveAssets();
    data.risk_free_rate = 0.03;

    MVOParameters p;
    p.constraints = PortfolioConstraints::longOnly(5);
    // Force a binding upper bound so analytical tangent fails feasibility
    p.constraints.upper_bounds = Vector::Constant(5, 0.20);
    MVOptimizer opt(p);
    auto r = opt.maxSharpePortfolio(data);
    REQUIRE(r.converged);
    for (int i = 0; i < 5; ++i)
        CHECK(r.weights[i] <= 0.20 + 1e-6);
    // Should NOT have taken the analytical path
    CHECK(r.status_message.find("analytical") == std::string::npos);
}

// ── A5: hard group constraints via augmented Lagrangian ──────────────────────

TEST_CASE("Hard group constraint is enforced exactly (A5)", "[mvo][groups][a5]") {
    auto data = fiveAssets();
    MVOParameters p;
    p.constraints   = PortfolioConstraints::longOnly(5);
    p.risk_aversion = 2.0;

    // Force the first two assets (the highest-return ones) into a sector cap
    // that they would otherwise breach.
    GroupConstraint g;
    g.description  = "Tech ≤ 30%";
    g.coefficients = Vector::Zero(5);
    g.coefficients[0] = 1.0;
    g.coefficients[1] = 1.0;
    g.lower = 0.0;
    g.upper = 0.30;
    p.constraints.groups.push_back(g);

    // Soft enforcement (baseline)
    p.hard_group_constraints = false;
    MVOptimizer soft(p);
    auto rs = soft.optimize(data);

    // Hard enforcement
    p.hard_group_constraints = true;
    p.group_tolerance        = 1e-7;
    MVOptimizer hard(p);
    auto rh = hard.optimize(data);
    REQUIRE(rh.converged);
    const double sum_hard = rh.weights[0] + rh.weights[1];
    // Hard must satisfy the constraint to tolerance
    CHECK(sum_hard <= 0.30 + 1e-6);
    // Hard enforcement should never be less tight than soft
    const double sum_soft = rs.weights[0] + rs.weights[1];
    CHECK(sum_hard <= sum_soft + 1e-6);
}

// ── A10: additional pathological-input coverage ───────────────────────────────

TEST_CASE("Singular covariance (perfect correlation) still produces feasible weights",
          "[mvo][singular][a10]") {
    MarketData d;
    d.assets = { {"A","A",0.10,1e11,""}, {"B","B",0.10,1e11,""} };
    d.expected_returns = Vector(2); d.expected_returns << 0.10, 0.10;
    d.covariance = Matrix(2, 2);
    // Rank-1: Σ = σ·σ' for some σ
    d.covariance << 0.04, 0.04, 0.04, 0.04;

    MVOParameters p;
    p.constraints = PortfolioConstraints::longOnly(2);
    p.risk_aversion = 2.0;
    MVOptimizer opt(p);
    REQUIRE_NOTHROW(opt.optimize(d));
    auto r = opt.optimize(d);
    CHECK(r.weights.sum() == Approx(1.0).margin(1e-6));
    CHECK(r.weights[0] >= -1e-9);
    CHECK(r.weights[1] >= -1e-9);
}

// ── B6: transaction-cost model ───────────────────────────────────────────────

TEST_CASE("Quadratic transaction cost pulls weights toward current_weights",
          "[mvo][txcost][b6]") {
    auto data = fiveAssets();
    MVOParameters p;
    p.constraints = PortfolioConstraints::longOnly(5);
    p.constraints.current_weights = Vector::Constant(5, 0.20);
    p.risk_aversion = 2.0;

    // Baseline (no transaction cost)
    MVOptimizer base(p);
    auto r_base = base.optimize(data);
    const double turnover_base =
        0.5 * (r_base.weights - p.constraints.current_weights).cwiseAbs().sum();

    // Add a strong quadratic cost — should shrink turnover.
    p.quadratic_transaction_cost = Vector::Constant(5, 0.5);
    MVOptimizer with_tc(p);
    auto r_tc = with_tc.optimize(data);
    const double turnover_tc =
        0.5 * (r_tc.weights - p.constraints.current_weights).cwiseAbs().sum();
    CHECK(turnover_tc < turnover_base);
}

TEST_CASE("Linear transaction cost shrinks turnover via sign iteration",
          "[mvo][txcost][b6]") {
    auto data = fiveAssets();
    MVOParameters p;
    p.constraints = PortfolioConstraints::longOnly(5);
    p.constraints.current_weights = Vector::Constant(5, 0.20);
    p.risk_aversion = 2.0;

    MVOptimizer base(p);
    const double turnover_base =
        0.5 * (base.optimize(data).weights -
               p.constraints.current_weights).cwiseAbs().sum();

    // Sizable linear cost (e.g., 2% per unit traded) should compress turnover.
    p.linear_transaction_cost = Vector::Constant(5, 0.02);
    MVOptimizer with_tc(p);
    const double turnover_tc =
        0.5 * (with_tc.optimize(data).weights -
               p.constraints.current_weights).cwiseAbs().sum();
    CHECK(turnover_tc < turnover_base);
}

// ── B1: tracking-error constraint ────────────────────────────────────────────

TEST_CASE("Tracking-error constraint is honoured", "[mvo][te][b1]") {
    auto data = fiveAssets();
    // Equal-weight benchmark
    data.benchmark_weights = Vector::Constant(5, 0.20);

    MVOParameters p;
    p.constraints = PortfolioConstraints::longOnly(5);
    p.risk_aversion = 5.0;

    // First solve without TE — get the unconstrained TE for reference.
    MVOptimizer opt(p);
    auto r_unc = opt.optimize(data);
    const double te_unc = r_unc.metrics.tracking_error;

    // Now cap TE at half the unconstrained value.
    p.constraints.tracking_error_limit = 0.5 * te_unc;
    opt.setParameters(p);
    auto r_te = opt.optimize(data);
    REQUIRE(r_te.converged);
    CHECK(r_te.metrics.tracking_error <=
          p.constraints.tracking_error_limit + 1e-4);
    // Sanity: should still beat the benchmark on Sharpe a bit
    CHECK(r_te.metrics.sharpe_ratio > 0.0);
}

// ── B2: gross-exposure / leverage cap ────────────────────────────────────────

TEST_CASE("Gross-exposure cap reduces leverage vs unconstrained",
          "[mvo][leverage][b2]") {
    auto data = fiveAssets();
    MVOParameters p;
    // 130/30-style — must allow shorts
    p.constraints = PortfolioConstraints::withShorts(5, 1.0);
    p.constraints.budget = 1.0;
    p.risk_aversion = 3.0;

    // Unconstrained baseline (no leverage cap)
    MVOptimizer opt(p);
    const double gross_unc = opt.optimize(data).weights.cwiseAbs().sum();

    // With leverage cap — sign-iteration heuristic substantially reduces gross.
    p.constraints.gross_exposure_limit = 1.4;
    p.hard_group_constraints = true;
    p.group_tolerance = 1e-4;
    p.group_penalty = 1e4;
    opt.setParameters(p);
    auto r = opt.optimize(data);
    const double gross_cap = r.weights.cwiseAbs().sum();

    // Heuristic linearisation cannot guarantee |w|_1 ≤ L exactly, but it
    // must meaningfully reduce gross exposure compared to the unconstrained
    // solution. (Hard enforcement would require variable-splitting; see B2
    // notes in the source for the rationale.)
    CHECK(gross_cap < gross_unc);
    CHECK(gross_cap < 0.85 * gross_unc);
    CHECK(r.weights.sum() == Approx(1.0).margin(1e-3));
}

TEST_CASE("Extreme λ sweep does not crash or NaN out",
          "[mvo][stress][a10]") {
    auto data = fiveAssets();
    MVOParameters p;
    p.constraints = PortfolioConstraints::longOnly(5);
    MVOptimizer opt(p);
    for (double lam : { 1e-8, 1e-3, 1.0, 1e3, 1e6 }) {
        p.risk_aversion = lam;
        opt.setParameters(p);
        auto r = opt.optimize(data);
        REQUIRE(std::isfinite(r.weights.sum()));
        CHECK(r.weights.sum() == Approx(1.0).margin(1e-5));
        for (int i = 0; i < 5; ++i) {
            CHECK(std::isfinite(r.weights[i]));
            CHECK(r.weights[i] >= -1e-9);
        }
    }
}
