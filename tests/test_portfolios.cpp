/**
 * @file test_portfolios.cpp
 * @brief Tests for the convenience portfolio constructors (B15).
 */

#include <catch2/catch_all.hpp>
#include <portopt/portfolios.hpp>

#include <random>

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

// ── B4: Hierarchical Risk Parity ─────────────────────────────────────────────

TEST_CASE("hierarchicalRiskParity sums to 1 and is long-only",
          "[portfolios][hrp][b4]") {
    Matrix S(4, 4);
    S << 0.04, 0.02, 0.005, 0.001,
         0.02, 0.05, 0.006, 0.002,
         0.005,0.006,0.03, 0.015,
         0.001,0.002,0.015,0.025;
    auto w = hierarchicalRiskParity(S);
    REQUIRE(w.size() == 4);
    CHECK(w.sum() == Approx(1.0).margin(1e-9));
    for (int i = 0; i < 4; ++i) {
        CHECK(w[i] > 0.0);
        CHECK(w[i] < 1.0);
    }
}

TEST_CASE("hierarchicalRiskParity prefers low-variance assets in equal-corr Σ",
          "[portfolios][hrp][b4]") {
    // Identical correlation, different variances — lowest-vol should win.
    Matrix S(3, 3);
    S << 0.01, 0.0, 0.0,
         0.0,  0.04, 0.0,
         0.0,  0.0,  0.09;
    auto w = hierarchicalRiskParity(S);
    CHECK(w[0] > w[1]);
    CHECK(w[1] > w[2]);
}

// ── B15: Maximum diversification ─────────────────────────────────────────────

TEST_CASE("maximumDiversification returns long-only normalised weights",
          "[portfolios][maxdiv]") {
    Matrix S(3, 3);
    S << 0.04, 0.01, 0.005,
         0.01, 0.05, 0.003,
         0.005,0.003,0.02;
    auto w = maximumDiversification(S);
    CHECK(w.sum() == Approx(1.0).margin(1e-9));
    for (int i = 0; i < 3; ++i) CHECK(w[i] >= 0.0);
}

// ── B15: Resampled MVO ───────────────────────────────────────────────────────

TEST_CASE("resampledMVO returns valid long-only weights",
          "[portfolios][michaud]") {
    Vector mu(3); mu << 0.10, 0.08, 0.06;
    Matrix S(3, 3);
    S << 0.04, 0.01, 0.005,
         0.01, 0.05, 0.003,
         0.005,0.003,0.02;
    auto w = resampledMVO(mu, S,
                          /*risk_aversion=*/2.0,
                          /*n_samples=*/120,
                          /*n_resamples=*/30,
                          /*seed=*/123u);
    REQUIRE(w.size() == 3);
    CHECK(w.sum() == Approx(1.0).margin(1e-9));
    for (int i = 0; i < 3; ++i) CHECK(w[i] >= 0.0);
}

// ── B5: Minimum-CVaR ─────────────────────────────────────────────────────────

TEST_CASE("realisedCVaR matches manual tail mean", "[portfolios][cvar][b5]") {
    Matrix R(10, 2);
    R <<  0.02,  0.03,
          0.01,  0.02,
          0.00,  0.01,
         -0.01,  0.00,
         -0.02, -0.01,
         -0.05, -0.04,
          0.04,  0.05,
          0.03,  0.04,
          0.05,  0.06,
         -0.10, -0.08;
    Vector w(2); w << 0.5, 0.5;

    // L = -r_t'w; sorted ascending in losses, tail 90% = top 10% of loss.
    // T=10, alpha=0.9, cut_idx = 9 (worst), so CVaR = max loss only.
    const double c = realisedCVaR(R, w, 0.9);
    const double worst_loss = -(R.row(9) * w).value();
    CHECK(c == Approx(worst_loss).margin(1e-12));
}

TEST_CASE("minimumCVaR returns budget-respecting long-only weights",
          "[portfolios][cvar][b5]") {
    Matrix R(200, 3);
    std::mt19937 rng(7);
    std::normal_distribution<double> nd(0.0, 0.02);
    std::normal_distribution<double> hd(0.0, 0.04);
    for (int t = 0; t < 200; ++t) {
        R(t, 0) = nd(rng);
        R(t, 1) = nd(rng);
        R(t, 2) = hd(rng);    // riskier
    }
    auto w = minimumCVaR(R, 0.95);
    REQUIRE(w.size() == 3);
    CHECK(w.sum() == Approx(1.0).margin(1e-3));
    for (int i = 0; i < 3; ++i) {
        CHECK(w[i] >= -1e-9);
        CHECK(w[i] <= 1.0 + 1e-9);
    }
    // Riskier asset (3rd) should have small weight in min-CVaR.
    CHECK(w[2] < std::max(w[0], w[1]));
}
