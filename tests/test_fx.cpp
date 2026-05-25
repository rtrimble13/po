/**
 * @file test_fx.cpp
 * @brief Tests for the multi-currency helpers (B13).
 */

#include <catch2/catch_all.hpp>
#include <portopt/fx.hpp>

using namespace portopt;
using Catch::Approx;

namespace {

AssetUniverse multiCurrencyAssets() {
    AssetUniverse u(4);
    u[0] = {"AAPL", "Apple",    0.10, 1e12, "Tech",     "USD"};
    u[1] = {"SAP",  "SAP SE",   0.08, 1e11, "Tech",     "EUR"};
    u[2] = {"TM",   "Toyota",   0.06, 1e11, "Auto",     "JPY"};
    u[3] = {"BRK",  "Berkshire",0.07, 1e11, "Holdings", "USD"};
    return u;
}

} // namespace

TEST_CASE("currencyExposure sums weights by currency code",
          "[fx][b13]") {
    auto u = multiCurrencyAssets();
    Vector w(4); w << 0.3, 0.2, 0.15, 0.35;
    const auto exp = fx::currencyExposure(u, w);
    // Three currencies present: EUR, JPY, USD
    REQUIRE(exp.currency.size() == 3);
    // Lexicographic order
    CHECK(exp.currency[0] == "EUR");
    CHECK(exp.currency[1] == "JPY");
    CHECK(exp.currency[2] == "USD");
    CHECK(exp.weight[0] == Approx(0.20).margin(1e-12));
    CHECK(exp.weight[1] == Approx(0.15).margin(1e-12));
    CHECK(exp.weight[2] == Approx(0.65).margin(1e-12));
    CHECK(exp.weight.sum() == Approx(1.0).margin(1e-12));
}

TEST_CASE("convertExpectedReturns applies FX uplift",
          "[fx][b13]") {
    auto u = multiCurrencyAssets();
    Vector mu_local(4); mu_local << 0.10, 0.08, 0.06, 0.07;
    std::map<std::string, double> fx_returns = {
        {"USD", 0.0},
        {"EUR", 0.03},
        {"JPY",-0.02}
    };

    auto mu_unhedged = fx::convertExpectedReturns(u, mu_local, fx_returns, 0.0);
    // USD-quoted assets unchanged
    CHECK(mu_unhedged[0] == Approx(0.10));
    CHECK(mu_unhedged[3] == Approx(0.07));
    // EUR: (1 + 0.08)(1 + 0.03) - 1 = 0.1124
    CHECK(mu_unhedged[1] == Approx(0.1124).margin(1e-6));
    // JPY: (1 + 0.06)(1 - 0.02) - 1 = 0.0388
    CHECK(mu_unhedged[2] == Approx(0.0388).margin(1e-6));

    // Fully hedged → FX contribution wiped out, equal to local
    auto mu_hedged = fx::convertExpectedReturns(u, mu_local, fx_returns, 1.0);
    for (int i = 0; i < 4; ++i)
        CHECK(mu_hedged[i] == Approx(mu_local[i]).margin(1e-12));
}

TEST_CASE("currencyExposureConstraint produces a sparse mask",
          "[fx][b13]") {
    auto u = multiCurrencyAssets();
    const auto g = fx::currencyExposureConstraint(u, "USD", 0.2, 0.7);
    REQUIRE(g.coefficients.size() == 4);
    CHECK(g.coefficients[0] == 1.0);    // AAPL  USD
    CHECK(g.coefficients[1] == 0.0);    // SAP   EUR
    CHECK(g.coefficients[2] == 0.0);    // TM    JPY
    CHECK(g.coefficients[3] == 1.0);    // BRK   USD
    CHECK(g.lower == 0.2);
    CHECK(g.upper == 0.7);
}
