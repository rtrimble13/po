/**
 * @file test_portfolios.cpp
 * @brief Tests for the convenience portfolio constructors (B15).
 */

#include <catch2/catch_all.hpp>
#include <portopt/portfolios.hpp>

using namespace portopt;
using namespace portopt::portfolios;
using Catch::Approx;

TEST_CASE("equalWeight gives 1/n on every asset", "[portfolios][equal_weight]") {
    auto w = equalWeight(5);
    CHECK(w.size() == 5);
    CHECK(w.sum() == Approx(1.0).margin(1e-12));
    for (int i = 0; i < 5; ++i)
        CHECK(w[i] == Approx(0.2).margin(1e-12));
    CHECK_THROWS(equalWeight(0));
    CHECK_THROWS(equalWeight(-1));
}

TEST_CASE("inverseVariance is biased toward low-variance assets",
          "[portfolios][inv_var]") {
    Matrix S(3, 3);
    S << 0.04, 0.00, 0.00,
         0.00, 0.01, 0.00,   // lowest variance — should get the most weight
         0.00, 0.00, 0.09;
    auto w = inverseVariance(S);
    CHECK(w.size() == 3);
    CHECK(w.sum() == Approx(1.0).margin(1e-12));
    CHECK(w[1] > w[0]);
    CHECK(w[0] > w[2]);
}

TEST_CASE("inverseVolatility lies between equal-weight and inverse-variance",
          "[portfolios][inv_vol]") {
    Matrix S(3, 3);
    S << 0.04, 0.00, 0.00,
         0.00, 0.01, 0.00,
         0.00, 0.00, 0.09;
    auto w_eq  = equalWeight(3);
    auto w_iv  = inverseVolatility(S);
    auto w_ivr = inverseVariance(S);
    CHECK(w_iv.sum() == Approx(1.0).margin(1e-12));
    // Inverse-vol concentrates less than inverse-variance on the low-vol
    // asset (asset 1) — but still more than equal-weight.
    CHECK(w_iv[1] > w_eq[1]);
    CHECK(w_iv[1] < w_ivr[1]);
}

TEST_CASE("marketCapWeighted normalises caps to weights",
          "[portfolios][market_cap]") {
    AssetUniverse u = {
        {"A", "", 0.0, 100.0},
        {"B", "", 0.0, 300.0},
        {"C", "", 0.0, 100.0}
    };
    auto w = marketCapWeighted(u);
    CHECK(w.sum() == Approx(1.0).margin(1e-12));
    CHECK(w[0] == Approx(0.2).margin(1e-12));
    CHECK(w[1] == Approx(0.6).margin(1e-12));
    CHECK(w[2] == Approx(0.2).margin(1e-12));
}

TEST_CASE("marketCapWeighted rejects zero total cap",
          "[portfolios][market_cap]") {
    AssetUniverse u = {{"A", "", 0.0, 0.0}};
    CHECK_THROWS(marketCapWeighted(u));
}

// ── B3: Equal Risk Contribution ───────────────────────────────────────────────

TEST_CASE("equalRiskContribution equalises risk contributions on diagonal Σ",
          "[portfolios][erc]") {
    // Diagonal covariance: ERC reduces to inverse-volatility.
    Matrix S(3, 3);
    S << 0.04, 0.00, 0.00,
         0.00, 0.16, 0.00,
         0.00, 0.00, 0.01;
    auto w = equalRiskContribution(S);
    CHECK(w.sum() == Approx(1.0).margin(1e-9));

    // Risk contributions w_i (Σw)_i should be nearly equal.
    Vector rc(3);
    for (int i = 0; i < 3; ++i) rc[i] = w[i] * (S.row(i).dot(w));
    const double mean_rc = rc.mean();
    for (int i = 0; i < 3; ++i)
        CHECK(rc[i] == Approx(mean_rc).margin(1e-6));

    // Reference: ERC on diagonal Σ matches inverse-vol weights.
    auto w_iv = inverseVolatility(S);
    for (int i = 0; i < 3; ++i)
        CHECK(w[i] == Approx(w_iv[i]).margin(1e-6));
}

TEST_CASE("equalRiskContribution handles correlated assets",
          "[portfolios][erc]") {
    Matrix S(3, 3);
    S <<  0.04,  0.01,  0.00,
          0.01,  0.09, -0.005,
          0.00, -0.005, 0.01;
    auto w = equalRiskContribution(S);
    REQUIRE(w.sum() == Approx(1.0).margin(1e-9));
    for (int i = 0; i < 3; ++i)
        CHECK(w[i] > 0.0);

    Vector rc(3);
    for (int i = 0; i < 3; ++i) rc[i] = w[i] * (S.row(i).dot(w));
    const double max_rc = rc.maxCoeff();
    const double min_rc = rc.minCoeff();
    CHECK((max_rc - min_rc) / max_rc < 1e-5);   // equalised to ~1e-5
}

TEST_CASE("equalRiskContribution rejects malformed inputs",
          "[portfolios][erc]") {
    Matrix S = Matrix::Zero(3, 3);    // diag is zero → undefined ERC
    CHECK_THROWS(equalRiskContribution(S));
    Matrix nonsquare = Matrix::Zero(3, 4);
    CHECK_THROWS(equalRiskContribution(nonsquare));
}
