#include <portopt/qp_solver.hpp>
#include <portopt/logging.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace portopt {
namespace qp {

// ── Simplex projection ────────────────────────────────────────────────────────
//
// Project v onto { x : sum(x) = s,  lb_i ≤ x_i ≤ ub_i }.
//
// The KKT system reduces to finding θ such that
//     x_i(θ) = clip(v_i − θ, lb_i, ub_i)
// satisfies  Σ x_i(θ) = s.  f(θ) := Σ x_i(θ) − s is monotone non-increasing in θ.
//
// A tight bracket is θ ∈ [min(v − ub), max(v − lb)] — at the lower endpoint
// every x_i is at its upper bound; at the upper endpoint every x_i is at its
// lower bound. No widening loop is needed.

Vector projectOntoSimplex(const Vector& v, double s,
                          const Vector& lb, const Vector& ub) {
    const int n = static_cast<int>(v.size());
    if (lb.size() != n || ub.size() != n)
        throw std::invalid_argument("projectOntoSimplex: dimension mismatch");

    // Verify feasibility: lb_sum <= s <= ub_sum
    const double lb_sum = lb.sum();
    const double ub_sum = ub.sum();
    if (s < lb_sum - 1e-9 || s > ub_sum + 1e-9)
        throw std::invalid_argument(
            "projectOntoSimplex: infeasible budget (s=" + std::to_string(s) +
            " not in [" + std::to_string(lb_sum) + ", " + std::to_string(ub_sum) + "])");

    auto x_of_theta = [&](double theta) {
        Vector x(n);
        for (int i = 0; i < n; ++i)
            x[i] = std::max(lb[i], std::min(ub[i], v[i] - theta));
        return x;
    };
    auto f_of_theta = [&](double theta) {
        double total = 0.0;
        for (int i = 0; i < n; ++i)
            total += std::max(lb[i], std::min(ub[i], v[i] - theta));
        return total - s;
    };

    // Tight bracket — at lo: every x_i is at ub (max possible sum)
    //                 at hi: every x_i is at lb (min possible sum)
    double lo = (v - ub).minCoeff();
    double hi = (v - lb).maxCoeff();
    // Guard against degenerate equality (e.g. lb == ub for all i)
    if (hi - lo < 1e-15) {
        return x_of_theta(0.5 * (lo + hi));
    }

    // Bisect to required precision
    for (int iter = 0; iter < 200; ++iter) {
        const double mid = 0.5 * (lo + hi);
        const double fmid = f_of_theta(mid);
        if (std::abs(fmid) < 1e-12) { lo = hi = mid; break; }
        if (fmid > 0) lo = mid;
        else          hi = mid;
        if (hi - lo < 1e-14 * std::max(1.0, std::abs(mid))) break;
    }

    return x_of_theta(0.5 * (lo + hi));
}

// ── Largest eigenvalue via power iteration ────────────────────────────────────

double largestEigenvalue(const Matrix& M, int max_iter, double tol) {
    const int n = static_cast<int>(M.rows());
    Vector b = Vector::Ones(n).normalized();
    double lambda = 0.0;

    for (int i = 0; i < max_iter; ++i) {
        Vector Ab = M * b;
        const double norm_Ab = Ab.norm();
        if (norm_Ab < 1e-30) return 0.0;
        b = Ab / norm_Ab;
        // Rayleigh quotient on normalised iterate
        const double lambda_new = b.dot(M * b);
        if (std::abs(lambda_new - lambda) <
            tol * std::max(1.0, std::abs(lambda_new)))
            return lambda_new;
        lambda = lambda_new;
    }
    return lambda;
}

// ── Group constraint contribution to gradient ─────────────────────────────────
//
// For each group g with bounds [lo_g, hi_g] and coefficient a_g:
//   violation_hi = max(0, a_g'x - hi_g)
//   violation_lo = max(0, lo_g - a_g'x)
//   penalty = 0.5 * κ * (violation_hi² + violation_lo²)
//   gradient contribution = κ * (violation_hi - violation_lo) * a_g

static Vector groupPenaltyGradient(const Vector& x,
                                   const std::vector<GroupConstraint>& groups,
                                   double kappa) {
    Vector g_total = Vector::Zero(x.size());
    if (kappa <= 0.0 || groups.empty()) return g_total;
    for (const auto& g : groups) {
        const double ax = g.coefficients.dot(x);
        double scale = 0.0;
        if (ax > g.upper) scale =  (ax - g.upper);
        if (ax < g.lower) scale -= (g.lower - ax);
        if (scale != 0.0)
            g_total.noalias() += (kappa * scale) * g.coefficients;
    }
    return g_total;
}

static double groupPenaltyValue(const Vector& x,
                                const std::vector<GroupConstraint>& groups,
                                double kappa) {
    if (kappa <= 0.0 || groups.empty()) return 0.0;
    double total = 0.0;
    for (const auto& g : groups) {
        const double ax = g.coefficients.dot(x);
        if (ax > g.upper) total += (ax - g.upper) * (ax - g.upper);
        if (ax < g.lower) total += (g.lower - ax) * (g.lower - ax);
    }
    return 0.5 * kappa * total;
}

// ── Core projected-gradient driver ───────────────────────────────────────────

static SolverResult solveImpl(const Matrix&                        Q,
                              const Vector&                        f,
                              const Vector&                        lb,
                              const Vector&                        ub,
                              const std::vector<GroupConstraint>&  groups,
                              double                               group_penalty,
                              const SolverConfig&                  cfg) {
    const int n = static_cast<int>(Q.rows());

    if (Q.cols() != n || f.size() != n || lb.size() != n || ub.size() != n)
        throw std::invalid_argument("QP solver: inconsistent dimensions");

    log::debug("QP solve: n={} budget={:.4f} groups={}",
               n, cfg.budget, groups.size());

    // Lipschitz constant L = largest eigenvalue of Q, augmented by group-penalty
    // term (κ * Σ a_g a_g').  Cheap upper bound: κ * Σ ||a_g||².
    double L = largestEigenvalue(Q);
    if (group_penalty > 0.0) {
        double bound = 0.0;
        for (const auto& g : groups)
            bound += g.coefficients.squaredNorm();
        L += group_penalty * bound;
    }
    if (L < 1e-12) L = 1.0;  // pure linear cost — gradient step still well-defined

    // Initialise: warm-start, equal-weight, or projected origin
    Vector w0;
    if (cfg.warm_start.size() == n)
        w0 = cfg.warm_start;
    else
        w0 = Vector::Constant(n, cfg.budget / static_cast<double>(n));
    Vector w = projectOntoSimplex(w0, cfg.budget, lb, ub);

    SolverResult result;
    result.converged = false;

    auto gradient = [&](const Vector& x) {
        Vector g = Q * x + f;
        if (group_penalty > 0.0 && !groups.empty())
            g.noalias() += groupPenaltyGradient(x, groups, group_penalty);
        return g;
    };

    if (!cfg.use_nesterov) {
        // Plain projected gradient
        for (int iter = 0; iter < cfg.max_iterations; ++iter) {
            const Vector grad = gradient(w);
            const Vector w_new = projectOntoSimplex(w - grad / L, cfg.budget, lb, ub);
            const double residual = (w_new - w).norm();
            w = w_new;

            if (residual < cfg.tolerance) {
                result.converged       = true;
                result.iterations      = iter + 1;
                result.primal_residual = residual;
                break;
            }
            if (iter == cfg.max_iterations - 1) {
                result.iterations      = cfg.max_iterations;
                result.primal_residual = residual;
            }
        }
    } else {
        // Nesterov-accelerated projected gradient (FISTA)
        Vector y = w;
        double t = 1.0;

        for (int iter = 0; iter < cfg.max_iterations; ++iter) {
            const Vector grad = gradient(y);
            const Vector w_new = projectOntoSimplex(y - grad / L, cfg.budget, lb, ub);
            const double t_new = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * t * t));
            y = w_new + ((t - 1.0) / t_new) * (w_new - w);

            const double residual = (w_new - w).norm();
            w = w_new;
            t = t_new;

            if (iter % 200 == 0)
                log::trace("QP iter={} residual={:.2e}", iter, residual);

            if (residual < cfg.tolerance) {
                result.converged       = true;
                result.iterations      = iter + 1;
                result.primal_residual = residual;
                break;
            }
            if (iter == cfg.max_iterations - 1) {
                result.iterations      = cfg.max_iterations;
                result.primal_residual = residual;
            }
        }
    }

    if (!result.converged)
        log::warn("QP solver did not converge in {} iterations (residual={:.2e})",
                  cfg.max_iterations, result.primal_residual);

    result.x         = w;
    result.gradient  = gradient(w);
    result.objective = 0.5 * w.dot(Q * w) + f.dot(w)
                     + groupPenaltyValue(w, groups, group_penalty);

    log::debug("QP done: obj={:.6f} converged={} iters={}",
               result.objective, result.converged, result.iterations);

    return result;
}

SolverResult solve(const Matrix&       Q,
                   const Vector&       f,
                   const Vector&       lb,
                   const Vector&       ub,
                   const SolverConfig& cfg) {
    return solveImpl(Q, f, lb, ub, {}, 0.0, cfg);
}

SolverResult solveWithGroups(const Matrix&                        Q,
                             const Vector&                        f,
                             const Vector&                        lb,
                             const Vector&                        ub,
                             const std::vector<GroupConstraint>&  groups,
                             double                               group_penalty,
                             const SolverConfig&                  cfg) {
    return solveImpl(Q, f, lb, ub, groups, group_penalty, cfg);
}

} // namespace qp
} // namespace portopt
