#include <portopt/mvo.hpp>
#include <portopt/logging.hpp>

#include <cmath>
#include <stdexcept>

namespace portopt {

MVOptimizer::MVOptimizer(MVOParameters params)
    : params_(std::move(params)) {
    solver_cfg_.tolerance      = 1e-9;
    solver_cfg_.max_iterations = 20000;
    solver_cfg_.use_nesterov   = true;
}

// ── Validation ────────────────────────────────────────────────────────────────

void MVOptimizer::validateData(const MarketData& data) const {
    const int n = static_cast<int>(data.assets.size());
    if (n == 0)
        throw std::invalid_argument("MVO: asset universe is empty");
    if (data.expected_returns.size() != n)
        throw std::invalid_argument("MVO: expected_returns size mismatch");
    if (data.covariance.rows() != n || data.covariance.cols() != n)
        throw std::invalid_argument("MVO: covariance matrix size mismatch");
    // Symmetry check (loose)
    if ((data.covariance - data.covariance.transpose()).norm() > 1e-6 * data.covariance.norm())
        throw std::invalid_argument("MVO: covariance matrix is not symmetric");
}

// ── Portfolio metrics ─────────────────────────────────────────────────────────

PortfolioMetrics MVOptimizer::computeMetrics(const Vector& w,
                                             const Vector& mu,
                                             const Matrix& sigma) {
    PortfolioMetrics m;
    m.expected_return = w.dot(mu);
    m.variance        = w.dot(sigma * w);
    m.volatility      = std::sqrt(std::max(0.0, m.variance));
    m.sharpe_ratio    = (m.volatility > 1e-12)
                        ? m.expected_return / m.volatility : 0.0;
    return m;
}

// ── Core single-λ solve ───────────────────────────────────────────────────────

OptimizationResult MVOptimizer::optimizeFor(const MarketData& data,
                                            double risk_aversion) {
    const int n = static_cast<int>(data.assets.size());

    // Ensure constraints are initialised
    PortfolioConstraints cons = params_.constraints;
    if (cons.lower_bounds.size() != n || cons.upper_bounds.size() != n)
        cons = PortfolioConstraints::longOnly(n);
    cons.validate(n);

    // QP: min  w'Σw - λ μ'w  ≡  min  0.5 w'(2Σ)w + (-λμ)'w
    Matrix Q = 2.0 * data.covariance;
    Vector f = -risk_aversion * data.expected_returns;

    solver_cfg_.budget = 1.0;
    qp::SolverResult qp = qp::solve(Q, f,
                                    cons.lower_bounds, cons.upper_bounds,
                                    solver_cfg_);

    OptimizationResult result;
    result.weights    = qp.x;
    result.assets     = data.assets;
    result.converged  = qp.converged;
    result.iterations = qp.iterations;
    result.method     = "MVO";
    result.metrics    = computeMetrics(qp.x,
                                       data.expected_returns,
                                       data.covariance);
    if (!qp.converged)
        result.status_message = "Solver did not converge; result may be approximate";

    return result;
}

// ── Public interface ──────────────────────────────────────────────────────────

OptimizationResult MVOptimizer::optimize(const MarketData& data) {
    validateData(data);
    const int n = static_cast<int>(data.assets.size());
    log::info("MVO optimize: n={} assets, lambda={:.4f}", n, params_.risk_aversion);

    auto result = optimizeFor(data, params_.risk_aversion);

    log::info("MVO result: return={:.4f} vol={:.4f} sharpe={:.4f} converged={}",
              result.metrics.expected_return,
              result.metrics.volatility,
              result.metrics.sharpe_ratio,
              result.converged);
    return result;
}

EfficientFrontier MVOptimizer::efficientFrontier(const MarketData& data) {
    validateData(data);
    const int n        = static_cast<int>(data.assets.size());
    const int pts      = params_.frontier_points;
    const double lo    = params_.min_risk_aversion;
    const double hi    = params_.max_risk_aversion;

    log::info("MVO efficient frontier: n={} assets, {} points, lambda=[{:.3f},{:.3f}]",
              n, pts, lo, hi);

    EfficientFrontier frontier;
    frontier.assets = data.assets;
    frontier.method = "MVO";
    frontier.points.reserve(pts);

    // Logarithmic sweep: more resolution at low-risk end
    for (int i = 0; i < pts; ++i) {
        double t      = static_cast<double>(i) / std::max(pts - 1, 1);
        double lambda = lo * std::pow(hi / lo, t);

        auto r = optimizeFor(data, lambda);

        EfficientFrontierPoint pt;
        pt.risk_aversion = lambda;
        pt.weights       = r.weights;
        pt.metrics       = r.metrics;
        frontier.points.push_back(std::move(pt));

        log::debug("Frontier pt {}/{}: lambda={:.3f} ret={:.4f} vol={:.4f}",
                   i + 1, pts, lambda,
                   frontier.points.back().metrics.expected_return,
                   frontier.points.back().metrics.volatility);
    }

    return frontier;
}

} // namespace portopt
