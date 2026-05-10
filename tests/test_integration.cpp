/**
 * @file test_integration.cpp
 * @brief End-to-end integration tests: read → optimise → write.
 */

#include <catch2/catch_all.hpp>
#include <portopt/portopt.hpp>

#include <filesystem>
#include <sstream>

using namespace portopt;
using Catch::Approx;

#ifndef PORTOPT_TEST_DATA_DIR
#define PORTOPT_TEST_DATA_DIR "tests/data"
#endif

static std::filesystem::path dataDir() {
    return std::filesystem::path(PORTOPT_TEST_DATA_DIR);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

static MarketData syntheticData(int n = 5) {
    // Reproducible synthetic market data
    MarketData d;
    d.assets.resize(n);
    std::vector<double> er  = {0.12, 0.10, 0.08, 0.09, 0.11};
    std::vector<double> var = {0.04, 0.03, 0.02, 0.025, 0.035};
    double corr = 0.2;

    for (int i = 0; i < n; ++i) {
        d.assets[i] = {std::string("A") + std::to_string(i+1),
                       "Asset " + std::to_string(i+1),
                       er[i % er.size()], 1e11};
    }

    d.expected_returns = Eigen::Map<Vector>(er.data(), std::min(n, 5));
    if (n > 5) {
        d.expected_returns.conservativeResize(n);
        for (int i = 5; i < n; ++i) d.expected_returns[i] = er[i % 5];
    }
    d.expected_returns.resize(n);
    for (int i = 0; i < n; ++i) d.expected_returns[i] = er[i % er.size()];

    // Build covariance with constant correlation
    d.covariance = Matrix(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double vi = var[i % 5], vj = var[j % 5];
            d.covariance(i, j) = (i == j) ? vi
                                           : corr * std::sqrt(vi * vj);
        }

    // Equal market weights
    d.market_weights = Vector::Constant(n, 1.0 / n);
    return d;
}

// ── Full MVO pipeline ─────────────────────────────────────────────────────────

TEST_CASE("Integration — MVO full pipeline", "[integration][mvo]") {
    auto data = syntheticData(5);

    MVOParameters params;
    params.risk_aversion = 3.0;
    params.frontier_points = 25;
    params.constraints = PortfolioConstraints::longOnly(5);

    MVOptimizer opt(params);
    auto result   = opt.optimize(data);
    auto frontier = opt.efficientFrontier(data);

    // ── Correctness checks ────────────────────────────────────────────────────
    REQUIRE(result.converged);
    CHECK(result.weights.sum() == Approx(1.0).epsilon(1e-7));
    CHECK(result.metrics.volatility > 0.0);
    REQUIRE(frontier.points.size() == 25);

    // ── JSON output ───────────────────────────────────────────────────────────
    std::ostringstream json_ss;
    io::WriterConfig cfg;
    cfg.format = io::OutputFormat::JSON;
    io::writeResult(result, json_ss, cfg);
    CHECK(!json_ss.str().empty());
    CHECK(json_ss.str().find("\"MVO\"") != std::string::npos);

    // ── CSV frontier output ───────────────────────────────────────────────────
    std::ostringstream csv_ss;
    cfg.format = io::OutputFormat::CSV;
    io::writeFrontier(frontier, csv_ss, cfg);
    CHECK(!csv_ss.str().empty());
    CHECK(csv_ss.str().find("risk_aversion") != std::string::npos);

    // ── Console output (smoke test) ───────────────────────────────────────────
    std::ostringstream console_ss;
    cfg.format = io::OutputFormat::Console;
    io::writeResult(result, console_ss, cfg);
    CHECK(!console_ss.str().empty());
}

// ── Full BL pipeline ──────────────────────────────────────────────────────────

TEST_CASE("Integration — Black-Litterman full pipeline", "[integration][bl]") {
    auto data = syntheticData(5);

    BlackLittermanParameters params;
    params.tau = 0.05;
    params.risk_aversion = 2.5;

    // View: Asset A1 will outperform A2 by 4%
    View v;
    v.description = "A1 outperforms A2";
    v.pick_vector = Vector(5);
    v.pick_vector << 1.0, -1.0, 0.0, 0.0, 0.0;
    v.expected_return = 0.04;
    v.confidence = 0.0005;
    params.views.push_back(v);

    params.mvo_params.risk_aversion = 2.5;
    params.mvo_params.constraints   = PortfolioConstraints::longOnly(5);
    params.mvo_params.frontier_points = 20;

    BlackLittermanOptimizer bl(params);
    auto result   = bl.optimize(data);
    auto frontier = bl.efficientFrontier(data);
    auto model    = bl.modelOutput(data);

    REQUIRE(result.converged);
    CHECK(result.weights.sum() == Approx(1.0).epsilon(1e-7));
    CHECK(result.method == "Black-Litterman");
    REQUIRE(frontier.points.size() == 20);

    // Model output sanity
    CHECK(model.prior_returns.size() == 5);
    CHECK(model.posterior_returns.size() == 5);

    // JSON BL model output
    auto json_str = io::blModelToJSON(model, data.assets);
    CHECK(json_str.find("prior_returns") != std::string::npos);
}

// ── File-based integration ────────────────────────────────────────────────────

TEST_CASE("Integration — JSON file round-trip", "[integration][file]") {
    auto path = dataDir() / "assets.json";
    if (!std::filesystem::exists(path)) SKIP("Test fixture not found");

    auto data = io::readMarketData(path);
    REQUIRE(data.assets.size() > 0);

    MVOParameters params;
    params.risk_aversion = 2.0;
    params.constraints = PortfolioConstraints::longOnly(
        static_cast<int>(data.assets.size()));

    MVOptimizer opt(params);
    auto result = opt.optimize(data);
    REQUIRE(result.converged);
}

TEST_CASE("Integration — CSV assets + covariance round-trip", "[integration][file]") {
    auto assets_path = dataDir() / "assets.csv";
    auto cov_path    = dataDir() / "covariance.csv";
    if (!std::filesystem::exists(assets_path) ||
        !std::filesystem::exists(cov_path)) SKIP("CSV test fixtures not found");

    auto data = io::readMarketDataFromCSV(assets_path, cov_path);
    REQUIRE(data.assets.size() > 0);
    CHECK(data.covariance.rows() == static_cast<int>(data.assets.size()));
}

// ── Numerical stability ───────────────────────────────────────────────────────

TEST_CASE("Integration — ill-conditioned covariance (near-singular)", "[integration][stability]") {
    // Two highly correlated assets
    MarketData d;
    d.assets = {{"X","",0.10,0}, {"Y","",0.10,0}, {"Z","",0.12,0}};
    d.expected_returns = Vector(3); d.expected_returns << 0.10, 0.10, 0.12;
    d.covariance = Matrix(3, 3);
    d.covariance <<
        0.04, 0.039, 0.005,
        0.039, 0.04,  0.005,
        0.005, 0.005, 0.03;

    MVOParameters params;
    params.risk_aversion = 2.0;
    params.constraints = PortfolioConstraints::longOnly(3);
    MVOptimizer opt(params);
    auto result = opt.optimize(d);

    // Should still produce a valid portfolio
    CHECK(result.weights.sum() == Approx(1.0).epsilon(1e-5));
    CHECK((result.weights.array() >= -1e-7).all());
}

TEST_CASE("Integration — single asset portfolio", "[integration][edge]") {
    MarketData d;
    d.assets = {{"ONLY","Only Asset",0.08,1e11}};
    d.expected_returns = Vector::Constant(1, 0.08);
    d.covariance = Matrix::Constant(1, 1, 0.04);

    MVOParameters params;
    params.risk_aversion = 1.0;
    params.constraints = PortfolioConstraints::longOnly(1);
    MVOptimizer opt(params);
    auto result = opt.optimize(d);

    REQUIRE(result.converged);
    CHECK(result.weights[0] == Approx(1.0).epsilon(1e-8));
}
