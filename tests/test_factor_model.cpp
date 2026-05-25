/**
 * @file test_factor_model.cpp
 * @brief Tests for the factor risk model (B7) and factor-neutrality
 *        constraints (B8).
 */

#include <catch2/catch_all.hpp>
#include <portopt/factor_model.hpp>
#include <portopt/mvo.hpp>

using namespace portopt;
using Catch::Approx;

namespace {

FactorRiskModel twoFactorThreeAsset() {
    FactorRiskModel m;
    m.loadings = Matrix(3, 2);
    m.loadings << 1.0, 0.5,
                  1.0, 0.2,
                  1.0, -0.3;
    m.factor_covariance = Matrix(2, 2);
    m.factor_covariance << 0.04, 0.0,
                           0.0,  0.02;
    m.specific_variance = Vector(3);
    m.specific_variance << 0.01, 0.01, 0.02;
    m.factor_names = { "market", "value" };
    return m;
}

} // namespace

TEST_CASE("FactorRiskModel reconstructs Σ = BΩB' + D",
          "[factor_model][b7]") {
    auto m = twoFactorThreeAsset();
    const Matrix S = m.assetCovariance();
    REQUIRE(S.rows() == 3);
    REQUIRE(S.cols() == 3);
    // Symmetric
    CHECK((S - S.transpose()).norm() < 1e-12);
    // Diagonal must equal B Ω B' diag + specific variance.
    const Matrix BOB = m.loadings * m.factor_covariance * m.loadings.transpose();
    for (int i = 0; i < 3; ++i)
        CHECK(S(i, i) == Approx(BOB(i, i) + m.specific_variance[i]).margin(1e-12));
}

TEST_CASE("FactorRiskModel rejects malformed inputs", "[factor_model][validate]") {
    FactorRiskModel m;
    m.loadings = Matrix(3, 2);
    m.loadings.setZero();
    // Wrong-sized factor_covariance
    m.factor_covariance = Matrix(3, 3);
    m.factor_covariance.setIdentity();
    m.specific_variance = Vector::Zero(3);
    CHECK_THROWS(m.validate());

    // Negative specific variance
    auto m2 = twoFactorThreeAsset();
    m2.specific_variance[0] = -0.01;
    CHECK_THROWS(m2.validate());

    // Mismatched names
    auto m3 = twoFactorThreeAsset();
    m3.factor_names.push_back("extra");
    CHECK_THROWS(m3.validate());
}

TEST_CASE("decomposeRisk: systematic + specific = total", "[factor_model][b7]") {
    auto m = twoFactorThreeAsset();
    Vector w(3); w << 0.5, 0.3, 0.2;
    const auto d = decomposeRisk(m, w);
    CHECK(d.total_variance ==
          Approx(d.systematic_variance + d.specific_variance).margin(1e-12));
    // Compare to direct Σw computation
    const Matrix S = m.assetCovariance();
    CHECK(d.total_variance == Approx(w.dot(S * w)).margin(1e-10));
    // Per-factor variance sum equals systematic
    CHECK(d.factor_variance.sum() ==
          Approx(d.systematic_variance).margin(1e-12));
}

TEST_CASE("factorNeutralConstraint pins net factor loading",
          "[factor_model][b8]") {
    auto m = twoFactorThreeAsset();
    // Constrain net value loading to [-0.01, 0.01].
    auto g = factorNeutralConstraint(m, /*factor_index=*/1, -0.01, 0.01);
    REQUIRE(g.coefficients.size() == 3);
    CHECK(g.coefficients[0] == Approx(0.5));
    CHECK(g.coefficients[1] == Approx(0.2));
    CHECK(g.coefficients[2] == Approx(-0.3));
    CHECK(g.lower == Approx(-0.01));
    CHECK(g.upper == Approx(0.01));
    CHECK(g.description.find("value") != std::string::npos);
}

TEST_CASE("MVO honours a factor-neutral constraint via group enforcement",
          "[factor_model][b8][mvo]") {
    auto fm = twoFactorThreeAsset();
    MarketData d;
    d.assets = { {"A","A",0.10,1e10,""},
                 {"B","B",0.08,1e10,""},
                 {"C","C",0.06,1e10,""} };
    d.expected_returns = Vector(3); d.expected_returns << 0.10, 0.08, 0.06;
    d.covariance = fm.assetCovariance();

    MVOParameters p;
    p.constraints = PortfolioConstraints::longOnly(3);
    p.constraints.groups.push_back(
        factorNeutralConstraint(fm, /*factor_index=*/1, -0.02, 0.02));
    p.hard_group_constraints = true;
    p.group_tolerance = 1e-5;
    p.risk_aversion = 2.0;

    MVOptimizer opt(p);
    auto r = opt.optimize(d);
    REQUIRE(r.converged);
    // Verify the actual factor exposure is inside the bound.
    const Vector exposures = fm.loadings.transpose() * r.weights;
    CHECK(std::abs(exposures[1]) <= 0.02 + 1e-4);
}
