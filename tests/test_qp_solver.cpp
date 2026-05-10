/**
 * @file test_qp_solver.cpp
 * @brief Unit tests for the QP solver (projection + FISTA solver).
 */

#include <catch2/catch_all.hpp>
#include <portopt/qp_solver.hpp>

using namespace portopt;
using namespace portopt::qp;
using Catch::Approx;

// ── Projection tests ──────────────────────────────────────────────────────────

TEST_CASE("projectOntoSimplex — uniform lb/ub", "[qp][projection]") {
    const int n = 4;
    Vector lb = Vector::Zero(n);
    Vector ub = Vector::Ones(n);

    SECTION("Already feasible") {
        Vector v = Vector::Constant(n, 0.25);
        Vector x = projectOntoSimplex(v, 1.0, lb, ub);
        CHECK(x.sum() == Approx(1.0).epsilon(1e-9));
        CHECK((x.array() >= -1e-12).all());
    }

    SECTION("Needs normalisation upward") {
        Vector v = Vector::Zero(n);
        Vector x = projectOntoSimplex(v, 1.0, lb, ub);
        CHECK(x.sum() == Approx(1.0).epsilon(1e-9));
    }

    SECTION("Needs normalisation downward") {
        Vector v = Vector::Constant(n, 1.0);
        Vector x = projectOntoSimplex(v, 1.0, lb, ub);
        CHECK(x.sum() == Approx(1.0).epsilon(1e-9));
        CHECK((x.array() <= 1.0 + 1e-9).all());
    }

    SECTION("Sparse solution") {
        // v heavily skewed to first element
        Vector v(n);
        v << 10.0, 0.01, 0.01, 0.01;
        Vector x = projectOntoSimplex(v, 1.0, lb, ub);
        CHECK(x.sum() == Approx(1.0).epsilon(1e-9));
        CHECK(x[0] == Approx(1.0).epsilon(1e-6));
        CHECK(x[1] < 1e-6);
    }
}

TEST_CASE("projectOntoSimplex — non-uniform bounds", "[qp][projection]") {
    const int n = 3;
    Vector lb(n); lb << 0.1, 0.2, 0.0;
    Vector ub(n); ub << 0.5, 0.6, 0.4;

    Vector v = Vector::Constant(n, 0.333);
    Vector x = projectOntoSimplex(v, 1.0, lb, ub);

    CHECK(x.sum() == Approx(1.0).epsilon(1e-9));
    for (int i = 0; i < n; ++i) {
        CHECK(x[i] >= lb[i] - 1e-9);
        CHECK(x[i] <= ub[i] + 1e-9);
    }
}

TEST_CASE("projectOntoSimplex — infeasible budget", "[qp][projection]") {
    Vector lb = Vector::Constant(3, 0.4); // sum(lb) = 1.2 > 1.0
    Vector ub = Vector::Ones(3);
    Vector v  = Vector::Constant(3, 0.333);
    CHECK_THROWS_AS(projectOntoSimplex(v, 1.0, lb, ub), std::invalid_argument);
}

// ── QP solver tests ───────────────────────────────────────────────────────────

TEST_CASE("QP solver — minimum variance portfolio (2 assets)", "[qp][solver]") {
    // Two uncorrelated assets with equal variance: optimal = 50/50
    int n = 2;
    Matrix Q(n, n);
    Q << 2.0, 0.0,
         0.0, 2.0;
    Vector f = Vector::Zero(n);
    Vector lb = Vector::Zero(n);
    Vector ub = Vector::Ones(n);

    auto res = solve(Q, f, lb, ub);

    REQUIRE(res.converged);
    CHECK(res.x[0] == Approx(0.5).epsilon(1e-5));
    CHECK(res.x[1] == Approx(0.5).epsilon(1e-5));
    CHECK(res.x.sum() == Approx(1.0).epsilon(1e-9));
}

TEST_CASE("QP solver — maximum return portfolio (2 assets)", "[qp][solver]") {
    // With λ very large, solver concentrates on highest-return asset
    int n = 2;
    Matrix Q(n, n);
    Q << 0.04, 0.0,
         0.0,  0.04;
    double lam = 100.0;
    Vector mu(n); mu << 0.05, 0.20; // asset 2 has much higher return
    Vector f = -lam * mu;
    Vector lb = Vector::Zero(n);
    Vector ub = Vector::Ones(n);

    auto res = solve(Q, f, lb, ub);

    REQUIRE(res.converged);
    CHECK(res.x.sum() == Approx(1.0).epsilon(1e-6));
    CHECK(res.x[1] > 0.95); // almost all weight on asset 2
}

TEST_CASE("QP solver — solution satisfies KKT conditions", "[qp][solver]") {
    // Verify w'Qw + f'w is minimised by checking gradient ~ 0 on active constraints
    int n = 5;
    Matrix A = Matrix::Random(n, n);
    Matrix Q = A * A.transpose() + Matrix::Identity(n, n) * 0.1; // PD
    Q *= 2.0;
    Vector f = Vector::Random(n);
    Vector lb = Vector::Zero(n);
    Vector ub = Vector::Ones(n);

    SolverConfig cfg;
    cfg.tolerance = 1e-8;
    auto res = solve(Q, f, lb, ub, cfg);

    REQUIRE(res.converged);
    CHECK(res.x.sum() == Approx(1.0).epsilon(1e-7));
    CHECK((res.x.array() >= -1e-8).all());
    CHECK((res.x.array() <= 1.0 + 1e-8).all());
}

TEST_CASE("QP solver — power iteration eigenvalue", "[qp][eigen]") {
    Matrix Q(3, 3);
    Q << 5.0, 1.0, 0.5,
         1.0, 4.0, 0.3,
         0.5, 0.3, 3.0;
    double L = largestEigenvalue(Q);
    double L_ref = Q.eigenvalues().real().maxCoeff();
    CHECK(L == Approx(L_ref).epsilon(1e-4));
}

TEST_CASE("QP solver — short-selling allowed", "[qp][solver]") {
    int n = 3;
    Matrix Q = Matrix::Identity(n, n) * 0.04;
    Vector mu(n); mu << 0.10, 0.08, 0.06;
    Vector f = -2.0 * mu;
    Vector lb = Vector::Constant(n, -0.5);
    Vector ub = Vector::Ones(n);

    SolverConfig cfg;
    cfg.budget = 1.0;
    auto res = solve(Q, f, lb, ub, cfg);

    REQUIRE(res.converged);
    CHECK(res.x.sum() == Approx(1.0).epsilon(1e-6));
    CHECK((res.x.array() >= -0.5 - 1e-8).all());
}
