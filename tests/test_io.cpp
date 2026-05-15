/**
 * @file test_io.cpp
 * @brief Unit tests for the IO layer (readers and writers).
 */

#include <catch2/catch_all.hpp>
#include <portopt/io/reader.hpp>
#include <portopt/io/writer.hpp>

#include <sstream>
#include <fstream>
#include <filesystem>

using namespace portopt;
using namespace portopt::io;
using Catch::Approx;

#ifndef PORTOPT_TEST_DATA_DIR
#define PORTOPT_TEST_DATA_DIR "tests/data"
#endif

static std::filesystem::path dataDir() {
    return std::filesystem::path(PORTOPT_TEST_DATA_DIR);
}

// ── Format inference ──────────────────────────────────────────────────────────

TEST_CASE("inferFormat — extension detection", "[io][format]") {
    CHECK(inferFormat("assets.json") == Format::JSON);
    CHECK(inferFormat("assets.JSON") == Format::JSON);
    CHECK(inferFormat("data.csv")    == Format::CSV);
    CHECK(inferFormat("params.toml") == Format::TOML);
    CHECK_THROWS(inferFormat("unknown.xyz"));
}

// ── JSON market data reader ───────────────────────────────────────────────────

TEST_CASE("readMarketDataFromJSON — basic round-trip", "[io][reader][json]") {
    std::string json_str = R"({
        "assets": [
            {"ticker": "A", "name": "Asset A", "expected_return": 0.10, "market_cap": 1e11},
            {"ticker": "B", "name": "Asset B", "expected_return": 0.15, "market_cap": 2e11}
        ],
        "covariance": [[0.04, 0.01], [0.01, 0.09]],
        "market_weights": [0.6, 0.4]
    })";

    auto data = readMarketDataFromJSON(json_str);

    REQUIRE(data.assets.size() == 2);
    CHECK(data.assets[0].ticker == "A");
    CHECK(data.assets[1].ticker == "B");
    CHECK(data.expected_returns[0] == Approx(0.10));
    CHECK(data.expected_returns[1] == Approx(0.15));
    CHECK(data.covariance(0, 0) == Approx(0.04));
    CHECK(data.covariance(0, 1) == Approx(0.01));
    REQUIRE(data.market_weights.has_value());
    CHECK((*data.market_weights)[0] == Approx(0.6));
}

TEST_CASE("readMarketDataFromJSON — missing assets throws", "[io][reader][json]") {
    std::string json_str = R"({"covariance": [[0.04]]})";
    CHECK_THROWS_AS(readMarketDataFromJSON(json_str), std::runtime_error);
}

TEST_CASE("readMarketDataFromJSON — missing covariance throws", "[io][reader][json]") {
    std::string json_str = R"({"assets": [{"ticker": "A", "expected_return": 0.1}]})";
    CHECK_THROWS_AS(readMarketDataFromJSON(json_str), std::runtime_error);
}

TEST_CASE("readMarketDataFromJSON — covariance size mismatch throws", "[io][reader][json]") {
    std::string json_str = R"({
        "assets": [{"ticker": "A", "expected_return": 0.1},
                   {"ticker": "B", "expected_return": 0.2}],
        "covariance": [[0.04]]
    })";
    CHECK_THROWS_AS(readMarketDataFromJSON(json_str), std::runtime_error);
}

// ── File-based readers ────────────────────────────────────────────────────────

TEST_CASE("readMarketData — JSON file round-trip", "[io][reader][file]") {
    auto path = dataDir() / "assets.json";
    if (!std::filesystem::exists(path)) SKIP("Test data not found");

    auto data = readMarketData(path);
    CHECK(data.assets.size() > 0);
    CHECK(data.covariance.rows() == static_cast<int>(data.assets.size()));
}

TEST_CASE("readMVOParameters — JSON file", "[io][reader][params]") {
    auto path = dataDir() / "params.json";
    if (!std::filesystem::exists(path)) SKIP("Test data not found");

    auto params = readMVOParameters(path);
    CHECK(params.risk_aversion > 0.0);
    CHECK(params.frontier_points > 0);
}

TEST_CASE("readMVOParameters — TOML file", "[io][reader][params][toml]") {
    auto path = dataDir() / "params.toml";
    if (!std::filesystem::exists(path)) SKIP("Test data not found");

    auto params = readMVOParameters(path);
    CHECK(params.risk_aversion > 0.0);
}

// ── JSON writer ───────────────────────────────────────────────────────────────

TEST_CASE("resultToJSON — produces valid JSON", "[io][writer][json]") {
    OptimizationResult r;
    r.method    = "MVO";
    r.converged = true;
    r.weights   = Vector::Constant(3, 1.0 / 3.0);
    r.assets    = {{"A","",0.1,0}, {"B","",0.2,0}, {"C","",0.15,0}};
    r.metrics.expected_return = 0.15;
    r.metrics.volatility      = 0.18;
    r.metrics.sharpe_ratio    = 0.83;

    auto json_str = resultToJSON(r);
    CHECK(!json_str.empty());
    CHECK(json_str.find("MVO") != std::string::npos);
    CHECK(json_str.find("expected_return") != std::string::npos);
    CHECK(json_str.find("\"A\"") != std::string::npos);
}

TEST_CASE("resultToCSV — produces valid CSV", "[io][writer][csv]") {
    OptimizationResult r;
    r.method  = "MVO";
    r.weights = Vector::Constant(2, 0.5);
    r.assets  = {{"X","Asset X",0.1,0}, {"Y","Asset Y",0.2,0}};
    r.metrics.expected_return = 0.15;
    r.metrics.volatility      = 0.20;

    auto csv = resultToCSV(r);
    CHECK(csv.find("ticker,name,weight") != std::string::npos);
    CHECK(csv.find("X,Asset X") != std::string::npos);
}

// ── Console writer ────────────────────────────────────────────────────────────

TEST_CASE("writeResult — console output contains key fields", "[io][writer][console]") {
    OptimizationResult r;
    r.method    = "MVO";
    r.converged = true;
    r.weights   = Vector::Constant(2, 0.5);
    r.assets    = {{"AAPL","Apple",0.15,0}, {"GOOG","Alphabet",0.12,0}};
    r.metrics.expected_return = 0.135;
    r.metrics.volatility      = 0.17;
    r.metrics.sharpe_ratio    = 0.79;

    std::ostringstream ss;
    WriterConfig cfg;
    cfg.format = OutputFormat::Console;
    writeResult(r, ss, cfg);

    auto out = ss.str();
    CHECK(out.find("MVO") != std::string::npos);
    CHECK(out.find("AAPL") != std::string::npos);
    CHECK(out.find("Sharpe") != std::string::npos);
}

// ── Frontier writer ───────────────────────────────────────────────────────────

TEST_CASE("frontierToCSV — has correct columns", "[io][writer][frontier]") {
    EfficientFrontier ef;
    ef.method = "MVO";
    ef.assets = {{"A","",0.1,0}, {"B","",0.2,0}};

    EfficientFrontierPoint pt;
    pt.risk_aversion = 1.0;
    pt.weights = Vector::Constant(2, 0.5);
    pt.metrics.expected_return = 0.15;
    pt.metrics.volatility      = 0.18;
    pt.metrics.sharpe_ratio    = 0.83;
    ef.points.push_back(pt);

    auto csv = frontierToCSV(ef);
    CHECK(csv.find("risk_aversion") != std::string::npos);
    CHECK(csv.find("volatility") != std::string::npos);
    CHECK(csv.find(",A,") != std::string::npos || csv.find(",A") != std::string::npos);
}

// ── BL constraints lookup ────────────────────────────────────────────────────

TEST_CASE("BL constraints under [black_litterman] are honoured", "[io][bl][constraints]") {
    // Inline JSON with constraints under bl section
    std::string js = R"({
        "mvo": { "risk_aversion": 1.0 },
        "black_litterman": {
            "tau": 0.05,
            "risk_aversion": 2.5,
            "constraints": {
                "lower_bounds": [0.1, 0.1, 0.1],
                "upper_bounds": [0.5, 0.5, 0.5],
                "budget": 1.0
            }
        }
    })";
    auto tmp = std::filesystem::temp_directory_path() / "_portopt_bl_cstr.json";
    {
        std::ofstream f(tmp);
        f << js;
    }
    auto p = readBLParameters(tmp);
    REQUIRE(p.mvo_params.constraints.lower_bounds.size() == 3);
    CHECK(p.mvo_params.constraints.lower_bounds[0] == Approx(0.1));
    CHECK(p.mvo_params.constraints.upper_bounds[0] == Approx(0.5));
    std::filesystem::remove(tmp);
}

TEST_CASE("BL constraints groups override MVO groups in JSON",
          "[io][bl][constraints][groups]") {
    std::string js = R"({
        "mvo": {
            "constraints": {
                "groups": [
                    { "description": "mvo-group",
                      "coefficients": [1.0, 0.0, 0.0],
                      "lower": -1.0, "upper": 0.2 }
                ]
            }
        },
        "black_litterman": {
            "constraints": {
                "groups": [
                    { "description": "bl-group",
                      "coefficients": [0.0, 1.0, 0.0],
                      "lower": -1.0, "upper": 0.4 }
                ]
            }
        }
    })";

    auto tmp = std::filesystem::temp_directory_path() / "_portopt_bl_groups_override.json";
    { std::ofstream f(tmp); f << js; }

    auto p = readBLParameters(tmp);
    REQUIRE(p.mvo_params.constraints.groups.size() == 1);
    CHECK(p.mvo_params.constraints.groups[0].description == "bl-group");
    CHECK(p.mvo_params.constraints.groups[0].upper == Approx(0.4));
    std::filesystem::remove(tmp);
}

TEST_CASE("BL constraints groups override MVO groups in TOML",
          "[io][bl][constraints][groups][toml]") {
    std::string toml = R"(
[mvo.constraints]
[[mvo.constraints.groups]]
description = "mvo-group"
coefficients = [1.0, 0.0, 0.0]
lower = -1.0
upper = 0.2

[black_litterman.constraints]
[[black_litterman.constraints.groups]]
description = "bl-group"
coefficients = [0.0, 1.0, 0.0]
lower = -1.0
upper = 0.4
)";

    auto tmp = std::filesystem::temp_directory_path() / "_portopt_bl_groups_override.toml";
    { std::ofstream f(tmp); f << toml; }

    auto p = readBLParameters(tmp);
    REQUIRE(p.mvo_params.constraints.groups.size() == 1);
    CHECK(p.mvo_params.constraints.groups[0].description == "bl-group");
    CHECK(p.mvo_params.constraints.groups[0].upper == Approx(0.4));
    std::filesystem::remove(tmp);
}

TEST_CASE("MVO constraints support budget + groups + turnover",
          "[io][mvo][constraints]") {
    std::string js = R"({
        "mvo": {
            "risk_aversion": 2.0,
            "constraints": {
                "lower_bounds": [-0.5, -0.5, -0.5],
                "upper_bounds": [ 0.5,  0.5,  0.5],
                "budget": 0.0,
                "turnover_penalty": 0.1,
                "current_weights": [0.0, 0.0, 0.0],
                "groups": [
                    { "description": "first two",
                      "coefficients": [1.0, 1.0, 0.0],
                      "lower": 0.0, "upper": 0.3 }
                ]
            }
        }
    })";
    auto tmp = std::filesystem::temp_directory_path() / "_portopt_mvo_cstr.json";
    { std::ofstream f(tmp); f << js; }
    auto p = readMVOParameters(tmp);
    CHECK(p.constraints.budget == Approx(0.0));
    CHECK(p.constraints.turnover_penalty == Approx(0.1));
    REQUIRE(p.constraints.groups.size() == 1);
    CHECK(p.constraints.groups[0].upper == Approx(0.3));
    std::filesystem::remove(tmp);
}

// ── Returns CSV ──────────────────────────────────────────────────────────────

TEST_CASE("readReturnsCSV produces MarketData", "[io][returns]") {
    auto tmp = std::filesystem::temp_directory_path() / "_portopt_returns.csv";
    {
        std::ofstream f(tmp);
        f << "date,A,B\n";
        for (int t = 0; t < 100; ++t) {
            f << t << "," << 0.001 + 0.0001 * t << "," << -0.001 + 0.0002 * t << "\n";
        }
    }
    auto md = readReturnsCSV(tmp, 252.0, "ledoit-wolf");
    CHECK(md.assets.size() == 2);
    CHECK(md.covariance.rows() == 2);
    std::filesystem::remove(tmp);
}

// ── Benchmark / risk-free in JSON ────────────────────────────────────────────

TEST_CASE("JSON market data includes benchmark and rf", "[io][json][benchmark]") {
    std::string j = R"({
        "assets": [
            {"ticker": "A", "expected_return": 0.10},
            {"ticker": "B", "expected_return": 0.15}
        ],
        "covariance": [[0.04, 0.01], [0.01, 0.09]],
        "market_weights":    [0.5, 0.5],
        "benchmark_weights": [0.4, 0.6],
        "risk_free_rate": 0.03
    })";
    auto md = readMarketDataFromJSON(j);
    REQUIRE(md.benchmark_weights.has_value());
    CHECK((*md.benchmark_weights)[0] == Approx(0.4));
    CHECK(md.risk_free_rate == Approx(0.03));
}

TEST_CASE("frontierToJSON — serialises correctly", "[io][writer][frontier]") {
    EfficientFrontier ef;
    ef.method = "Black-Litterman";
    ef.assets = {{"X","",0.1,0}};

    EfficientFrontierPoint pt;
    pt.risk_aversion = 2.5;
    pt.weights = Vector::Ones(1);
    pt.metrics.expected_return = 0.10;
    pt.metrics.volatility      = 0.15;
    ef.points.push_back(pt);

    auto j = frontierToJSON(ef);
    CHECK(j.find("Black-Litterman") != std::string::npos);
    CHECK(j.find("2.5") != std::string::npos);
}
