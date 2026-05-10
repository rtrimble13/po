#include <portopt/qp_solver.hpp>
#include <portopt/logging.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace portopt {
namespace qp {

// ── Simplex projection ────────────────────────────────────────────────────────

Vector projectOntoSimplex(const Vector& v, double s,
                          const Vector& lb, const Vector& ub) {
    const int n = static_cast<int>(v.size());
    if (lb.size() != n || ub.size() != n)
        throw std::invalid_argument("projectOntoSimplex: dimension mismatch");

    // Verify feasibility: lb_sum <= s <= ub_sum
    double lb_sum = lb.sum();
    double ub_sum = ub.sum();
    if (s < lb_sum - 1e-12 || s > ub_sum + 1e-12)
        throw std::invalid_argument("projectOntoSimplex: infeasible budget");

    // Clamp to box first
    Vector x = v.cwiseMax(lb).cwiseMin(ub);

    // If already sums to s, done
    double residual = x.sum() - s;
    if (std::abs(residual) < 1e-12)
        return x;

    // Bisect on theta: x_i(theta) = clip(v_i - theta, lb_i, ub_i)
    // f(theta) = sum(x_i(theta)) - s  is monotone decreasing
    auto f = [&](double theta) {
        double total = 0.0;
        for (int i = 0; i < n; ++i)
            total += std::max(lb[i], std::min(ub[i], v[i] - theta));
        return total - s;
    };

    // Bracket: find theta such that f changes sign
    double lo = v.minCoeff() - ub.maxCoeff() - 1.0;
    double hi = v.maxCoeff() - lb.minCoeff() + 1.0;

    // Safety: widen until bracketed
    for (int k = 0; k < 100 && f(lo) < 0; ++k) lo -= std::abs(lo) + 1.0;
    for (int k = 0; k < 100 && f(hi) > 0; ++k) hi += std::abs(hi) + 1.0;

    // Bisection
    for (int iter = 0; iter < 200; ++iter) {
        double mid = 0.5 * (lo + hi);
        double fmid = f(mid);
        if (std::abs(fmid) < 1e-12) { lo = hi = mid; break; }
        if (fmid > 0) lo = mid;
        else          hi = mid;
    }

    double theta = 0.5 * (lo + hi);
    for (int i = 0; i < n; ++i)
        x[i] = std::max(lb[i], std::min(ub[i], v[i] - theta));

    return x;
}

// ── Largest eigenvalue via power iteration ────────────────────────────────────

double largestEigenvalue(const Matrix& M, int max_iter, double tol) {
    const int n = static_cast<int>(M.rows());
    Vector b = Vector::Ones(n).normalized();
    double lambda = 0.0;

    for (int i = 0; i < max_iter; ++i) {
        Vector Ab = M * b;
        double lambda_new = b.dot(Ab);
        b = Ab.normalized();
        if (std::abs(lambda_new - lambda) < tol * std::max(1.0, std::abs(lambda_new)))
            return lambda_new;
        lambda = lambda_new;
    }
    return lambda;
}

// ── Main QP solver ────────────────────────────────────────────────────────────

SolverResult solve(const Matrix&       Q,
                   const Vector&       f,
                   const Vector&       lb,
                   const Vector&       ub,
                   const SolverConfig& cfg) {
    const int n = static_cast<int>(Q.rows());

    if (Q.cols() != n || f.size() != n || lb.size() != n || ub.size() != n)
        throw std::invalid_argument("QP solver: inconsistent dimensions");

    log::debug("QP solve: n={} budget={:.4f}", n, cfg.budget);

    // Lipschitz constant L = largest eigenvalue of Q
    double L = largestEigenvalue(Q);
    if (L < 1e-12) {
        // Q is (near) zero — minimise f'x subject to constraints
        // Optimal: push weight to lowest-f asset within simplex
        Vector x = Vector::Zero(n);
        int idx;
        f.minCoeff(&idx);
        x = projectOntoSimplex(
            Vector::Constant(n, cfg.budget / n), cfg.budget, lb, ub);
        SolverResult res;
        res.x = x;
        res.converged = true;
        res.iterations = 0;
        res.objective = 0.5 * x.dot(Q * x) + f.dot(x);
        return res;
    }

    // Initialise: equal-weight portfolio projected onto feasible set
    Vector w = projectOntoSimplex(
        Vector::Constant(n, cfg.budget / n), cfg.budget, lb, ub);

    SolverResult result;
    result.converged = false;

    if (!cfg.use_nesterov) {
        // Plain projected gradient
        for (int iter = 0; iter < cfg.max_iterations; ++iter) {
            Vector grad = Q * w + f;
            Vector w_new = projectOntoSimplex(w - grad / L, cfg.budget, lb, ub);
            double residual = (w_new - w).norm();
            w = w_new;

            if (residual < cfg.tolerance) {
                result.converged  = true;
                result.iterations = iter + 1;
                result.primal_residual = residual;
                break;
            }
        }
    } else {
        // Nesterov-accelerated projected gradient (FISTA)
        Vector y = w;
        double t = 1.0;

        for (int iter = 0; iter < cfg.max_iterations; ++iter) {
            Vector grad = Q * y + f;
            Vector w_new = projectOntoSimplex(y - grad / L, cfg.budget, lb, ub);
            double t_new = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * t * t));
            y = w_new + ((t - 1.0) / t_new) * (w_new - w);

            double residual = (w_new - w).norm();
            w = w_new;
            t = t_new;

            if (iter % 100 == 0)
                log::trace("QP iter={} residual={:.2e}", iter, residual);

            if (residual < cfg.tolerance) {
                result.converged  = true;
                result.iterations = iter + 1;
                result.primal_residual = residual;
                break;
            }
        }
    }

    if (!result.converged)
        log::warn("QP solver did not converge in {} iterations", cfg.max_iterations);

    result.x = w;
    result.objective = 0.5 * w.dot(Q * w) + f.dot(w);

    log::debug("QP done: obj={:.6f} converged={} iters={}",
               result.objective, result.converged, result.iterations);

    return result;
}

} // namespace qp
} // namespace portopt
