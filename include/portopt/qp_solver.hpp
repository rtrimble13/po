#pragma once
/**
 * @file qp_solver.hpp
 * @brief Quadratic programming solver for portfolio weight optimisation.
 *
 * Solves the box-and-simplex constrained QP:
 *
 *   min   0.5 x'Qx + f'x
 *   s.t.  sum(x) = S          (budget constraint, default S = 1)
 *         lb_i <= x_i <= ub_i  (per-asset bounds)
 *
 * Method: Nesterov-accelerated projected gradient descent.
 * Projection onto the generalised simplex with box constraints is performed
 * in O(n log n) via bisection on the dual Lagrange multiplier.
 */

#include "types.hpp"

namespace portopt {
namespace qp {

/// Solver configuration.
struct SolverConfig {
    int    max_iterations{10000};
    double tolerance{1e-9};        ///< Primal / dual residual convergence tol
    double budget{1.0};            ///< Budget constraint (sum of weights)
    bool   use_nesterov{true};     ///< Enable momentum acceleration
};

/// Raw result from the QP solver.
struct SolverResult {
    Vector x;
    double objective{0.0};
    int    iterations{0};
    bool   converged{false};
    double primal_residual{0.0};   ///< ||x_{k+1} - x_k||
};

/**
 * @brief Project vector @p v onto { x : sum(x)=s, lb<=x<=ub }.
 *
 * Uses bisection on the KKT multiplier; guaranteed O(n log(1/ε)).
 *
 * @param v   Input vector to project
 * @param s   Required sum (budget)
 * @param lb  Element-wise lower bounds
 * @param ub  Element-wise upper bounds
 * @return    Projected vector satisfying all constraints
 */
Vector projectOntoSimplex(const Vector& v, double s,
                          const Vector& lb, const Vector& ub);

/**
 * @brief Solve the box-simplex QP.
 *
 * @param Q      Symmetric positive semi-definite cost matrix (n×n)
 * @param f      Linear cost vector (n×1)
 * @param lb     Lower bounds (n×1)
 * @param ub     Upper bounds (n×1)
 * @param cfg    Solver configuration
 * @return       SolverResult with optimal weights and diagnostics
 */
SolverResult solve(const Matrix&       Q,
                   const Vector&       f,
                   const Vector&       lb,
                   const Vector&       ub,
                   const SolverConfig& cfg = {});

/**
 * @brief Compute the largest eigenvalue of @p M via power iteration.
 *
 * Used internally to set the gradient step size (Lipschitz constant).
 */
double largestEigenvalue(const Matrix& M, int max_iter = 200,
                         double tol = 1e-10);

} // namespace qp
} // namespace portopt
