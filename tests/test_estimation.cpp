/**
 * @file test_estimation.cpp
 * @brief Tests for the covariance / mean estimators (sample, Ledoit-Wolf, OAS).
 */

#include <catch2/catch_all.hpp>
#include <portopt/estimation.hpp>

#include <random>

using namespace portopt;
using namespace portopt::estimation;
using Catch::Approx;

namespace {

Matrix syntheticReturns(int T, int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, 0.01);
    Matrix R(T, n);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < n; ++i)
            R(t, i) = nd(rng);
    return R;
}

} // namespace

TEST_CASE("sampleMean is annualised correctly", "[estimation][mean]") {
    Matrix R = syntheticReturns(1000, 3, 1);
    Vector mu = sampleMean(R, 252.0);
    Vector mu_raw = R.colwise().mean().transpose();
    CHECK((mu - 252.0 * mu_raw).norm() < 1e-12);
}

TEST_CASE("sampleCovariance is symmetric and PSD", "[estimation][cov]") {
    Matrix R = syntheticReturns(500, 4, 2);
    Matrix S = sampleCovariance(R, true, 252.0);
    CHECK((S - S.transpose()).norm() < 1e-10);
    Eigen::SelfAdjointEigenSolver<Matrix> es(S);
    CHECK(es.eigenvalues().minCoeff() > -1e-10);
}

TEST_CASE("linearShrinkage(delta=0) == sample", "[estimation][shrinkage]") {
    Matrix R = syntheticReturns(300, 5, 3);
    Matrix S = sampleCovariance(R);
    Matrix Z = linearShrinkage(S, 0.0);
    CHECK((Z - S).norm() < 1e-12);
}

TEST_CASE("linearShrinkage(delta=1) == target", "[estimation][shrinkage]") {
    Matrix R = syntheticReturns(300, 5, 4);
    Matrix S = sampleCovariance(R);
    Matrix Z = linearShrinkage(S, 1.0);
    // Default target = diag(S)
    Matrix expected = Matrix::Zero(5, 5);
    expected.diagonal() = S.diagonal();
    CHECK((Z - expected).norm() < 1e-12);
}

TEST_CASE("Ledoit-Wolf returns δ ∈ [0, 1]", "[estimation][lw]") {
    Matrix R = syntheticReturns(200, 6, 5);
    double delta = -1.0;
    Matrix S = ledoitWolfShrinkage(R, 252.0, &delta);
    CHECK(delta >= 0.0);
    CHECK(delta <= 1.0);
    CHECK(S.rows() == 6);
    CHECK(S.cols() == 6);
    CHECK((S - S.transpose()).norm() < 1e-10);
}

TEST_CASE("OAS returns δ ∈ [0, 1]", "[estimation][oas]") {
    Matrix R = syntheticReturns(50, 10, 6);  // T < n: OAS shines here
    double delta = -1.0;
    Matrix S = oasShrinkage(R, 252.0, &delta);
    CHECK(delta >= 0.0);
    CHECK(delta <= 1.0);
    Eigen::SelfAdjointEigenSolver<Matrix> es(S);
    CHECK(es.eigenvalues().minCoeff() > -1e-10);
}

TEST_CASE("fromReturns builds valid MarketData with shrinkage",
          "[estimation][from_returns]") {
    std::vector<std::string> tickers = {"A", "B", "C"};
    Matrix R = syntheticReturns(252, 3, 7);

    auto md = fromReturns(tickers, R, 252.0, "ledoit-wolf");
    REQUIRE(md.assets.size() == 3);
    CHECK(md.expected_returns.size() == 3);
    CHECK(md.covariance.rows() == 3);
    CHECK(md.assets[0].ticker == "A");
    CHECK(md.assets[0].expected_return ==
          Approx(md.expected_returns[0]).epsilon(1e-12));
}

TEST_CASE("fromReturns rejects unknown shrinkage", "[estimation][from_returns]") {
    Matrix R = syntheticReturns(50, 2, 8);
    CHECK_THROWS(fromReturns({"A","B"}, R, 252.0, "magic"));
}

// ── A8: Missing-data (NaN / Inf) rejection ────────────────────────────────────

TEST_CASE("sampleCovariance rejects NaN with actionable message",
          "[estimation][nan]") {
    Matrix R = syntheticReturns(50, 3, 11);
    R(5, 1) = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(sampleCovariance(R), std::invalid_argument);
}

TEST_CASE("Shrinkage estimators reject Inf inputs", "[estimation][nan]") {
    Matrix R = syntheticReturns(50, 3, 12);
    R(0, 0) = std::numeric_limits<double>::infinity();
    CHECK_THROWS_AS(ledoitWolfShrinkage(R, 252.0), std::invalid_argument);
    CHECK_THROWS_AS(oasShrinkage(R, 252.0), std::invalid_argument);
}

// ── A6: Ledoit-Wolf δ is stable across sample sizes ───────────────────────────
//
// Without an external reference here we anchor on internal sanity:
// (1) δ̂ is deterministic for a fixed seed,
// (2) it lies in [0, 1],
// (3) it shrinks toward 1 as T decreases (more noise → more shrinkage).

// ── B14: EWMA covariance ──────────────────────────────────────────────────────

TEST_CASE("EWMA covariance is symmetric and PSD", "[estimation][ewma]") {
    Matrix R = syntheticReturns(500, 4, 17);
    Matrix S = ewmaCovariance(R, 0.94, 252.0);
    REQUIRE(S.rows() == 4);
    REQUIRE(S.cols() == 4);
    CHECK((S - S.transpose()).cwiseAbs().maxCoeff() < 1e-12);
    Eigen::SelfAdjointEigenSolver<Matrix> es(S);
    CHECK(es.eigenvalues().minCoeff() > -1e-10);
}

TEST_CASE("EWMA covariance weights recent observations more heavily",
          "[estimation][ewma]") {
    // Build a returns matrix where only the last K rows have variance,
    // earlier rows are zero. A lower λ (more aggressive decay) should
    // produce a larger covariance than a higher λ — the recent vol
    // dominates more.
    const int T = 500, n = 2;
    Matrix R = Matrix::Zero(T, n);
    std::mt19937 rng(99);
    std::normal_distribution<double> nd(0.0, 0.02);
    for (int t = T - 20; t < T; ++t)
        for (int i = 0; i < n; ++i) R(t, i) = nd(rng);

    Matrix S_fast = ewmaCovariance(R, 0.80, 1.0);  // fast decay
    Matrix S_slow = ewmaCovariance(R, 0.99, 1.0);  // slow decay
    // Sum of diagonals (total variance) should be larger under fast decay
    CHECK(S_fast.diagonal().sum() > S_slow.diagonal().sum());
}

TEST_CASE("EWMA rejects bad lambda", "[estimation][ewma]") {
    Matrix R = syntheticReturns(100, 2, 19);
    CHECK_THROWS(ewmaCovariance(R, 0.0, 1.0));
    CHECK_THROWS(ewmaCovariance(R, 1.0, 1.0));
    CHECK_THROWS(ewmaCovariance(R, -0.1, 1.0));
}

TEST_CASE("Ledoit-Wolf δ̂ grows as T shrinks", "[estimation][lw]") {
    const int n = 5;
    Matrix R_large = syntheticReturns(2000, n, 13);
    Matrix R_small = syntheticReturns(50,   n, 13);

    double d_large = 0.0, d_small = 0.0;
    (void)ledoitWolfShrinkage(R_large, 252.0, &d_large);
    (void)ledoitWolfShrinkage(R_small, 252.0, &d_small);

    CHECK(d_large >= 0.0);
    CHECK(d_large <= 1.0);
    CHECK(d_small >= 0.0);
    CHECK(d_small <= 1.0);
    CHECK(d_small > d_large);   // less data ⇒ more shrinkage
}
