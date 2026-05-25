#include <portopt/mvo.hpp>
#include <portopt/portopt.hpp>   // VERSION_STRING for audit trail
#include <portopt/logging.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace portopt {

MVOptimizer::MVOptimizer(MVOParameters params)
    : params_(std::move(params)) {
    solver_cfg_.tolerance      = 1e-9;
    solver_cfg_.max_iterations = 20000;
    solver_cfg_.use_nesterov   = true;
}

// Resolve the effective risk-free rate: an explicit non-zero
// MVOParameters value overrides whatever is on the MarketData.
// A zero in the params field is treated as "not set" — the data value
// then applies. Used by both `optimizeFor` and `maxSharpePortfolio`
// so the two call sites cannot drift.
static double effectiveRiskFreeRate(const MVOParameters& params,
                                    const MarketData& data) {
    return (params.risk_free_rate != 0.0) ? params.risk_free_rate
                                          : data.risk_free_rate;
}

// ── Numerical tolerances (A9) ────────────────────────────────────────────────
// Single source of truth for "near a bound" / "approximately equal"
// checks throughout the MVO pipeline. Absolute values are appropriate
// for weights — which live in [-1, 1] for any realistic portfolio —
// so we don't scale these by ‖w‖. Group / linear tolerances scale with
// the magnitude of the coefficient vector via the caller.
namespace tol {
    constexpr double kActiveBound  = 1e-6;   ///< |w_i − bound| < kActiveBound
    constexpr double kFeasibility  = 1e-9;   ///< budget / equality slack
    constexpr double kPSD          = 1e-6;   ///< relative eigenvalue floor on Σ
}

// ── Audit-trail hashing (B16) ────────────────────────────────────────────────
// Stable, version-independent 64-bit FNV-1a over a canonicalised
// stringification of inputs / params. Used to stamp every
// OptimizationResult so two runs with identical inputs share an
// identical hash, and any drift surfaces immediately.
static std::string fnv1aHex(const std::string& s) {
    std::uint64_t h = 0xcbf29ce484222325ULL;
    for (char c : s) {
        h ^= static_cast<std::uint8_t>(c);
        h *= 0x100000001b3ULL;
    }
    char buf[20];
    std::snprintf(buf, sizeof(buf), "%016llx",
                  static_cast<unsigned long long>(h));
    return std::string(buf);
}

static void writeVec(std::ostringstream& os, const Vector& v) {
    os << '[' << v.size() << ':';
    os.precision(16);
    for (int i = 0; i < v.size(); ++i) os << v[i] << ',';
    os << ']';
}

static void writeMat(std::ostringstream& os, const Matrix& M) {
    os << '{' << M.rows() << 'x' << M.cols() << ':';
    os.precision(16);
    for (int i = 0; i < M.rows(); ++i)
        for (int j = 0; j < M.cols(); ++j) os << M(i, j) << ',';
    os << '}';
}

static std::string inputHash(const MarketData& data) {
    std::ostringstream os;
    writeVec(os, data.expected_returns);
    writeMat(os, data.covariance);
    if (data.market_weights)    { os << "mw="; writeVec(os, *data.market_weights); }
    if (data.benchmark_weights) { os << "bm="; writeVec(os, *data.benchmark_weights); }
    os.precision(16);
    os << "rf=" << data.risk_free_rate;
    return fnv1aHex(os.str());
}

static std::string paramsHash(const MVOParameters& p) {
    std::ostringstream os;
    os.precision(16);
    os << "lam=" << p.risk_aversion
       << " rf="  << p.risk_free_rate
       << " grp_pen=" << p.group_penalty
       << " budget=" << p.constraints.budget
       << " kappa="  << p.constraints.turnover_penalty;
    writeVec(os, p.constraints.lower_bounds);
    writeVec(os, p.constraints.upper_bounds);
    if (p.constraints.current_weights.size() > 0)
        writeVec(os, p.constraints.current_weights);
    for (const auto& g : p.constraints.groups) {
        os << "[" << g.lower << ',' << g.upper << ']';
        writeVec(os, g.coefficients);
    }
    return fnv1aHex(os.str());
}

// ── Validation ────────────────────────────────────────────────────────────────

void MVOptimizer::validateMarketData(const MarketData& data) {
    const int n = static_cast<int>(data.assets.size());
    if (n == 0)
        throw std::invalid_argument("MVO: asset universe is empty");
    if (data.expected_returns.size() != n)
        throw std::invalid_argument(
            "MVO: expected_returns size mismatch (got " +
            std::to_string(data.expected_returns.size()) +
            ", expected " + std::to_string(n) + ")");
    if (data.covariance.rows() != n || data.covariance.cols() != n)
        throw std::invalid_argument(
            "MVO: covariance matrix size mismatch (got " +
            std::to_string(data.covariance.rows()) + "x" +
            std::to_string(data.covariance.cols()) +
            ", expected " + std::to_string(n) + "x" + std::to_string(n) + ")");

    // Symmetry check (relative tolerance)
    const double sym_err = (data.covariance - data.covariance.transpose()).norm();
    if (sym_err > 1e-6 * std::max(1.0, data.covariance.norm()))
        throw std::invalid_argument("MVO: covariance matrix is not symmetric");

    // Diagonal must be non-negative (variances)
    for (int i = 0; i < n; ++i) {
        if (data.covariance(i, i) < -1e-10)
            throw std::invalid_argument(
                "MVO: covariance has negative diagonal at index " + std::to_string(i));
    }

    // PSD check via LDLT (fast for symmetric matrices)
    Eigen::LDLT<Matrix> ldlt(0.5 * (data.covariance + data.covariance.transpose()));
    if (ldlt.info() != Eigen::Success) {
        log::warn("MVO: covariance LDLT decomposition failed; "
                  "matrix may be indefinite — proceeding with caution");
    } else {
        const Vector d = ldlt.vectorD();
        const double min_d = d.minCoeff();
        const double max_d = d.maxCoeff();
        if (min_d < -1e-8 * std::max(1.0, std::abs(max_d))) {
            log::warn("MVO: covariance is not positive semi-definite "
                      "(min eigval ≈ {:.3e}); consider shrinkage or regularisation",
                      min_d);
        }
    }

    if (data.benchmark_weights.has_value() &&
        data.benchmark_weights->size() != n)
        throw std::invalid_argument("MVO: benchmark_weights size mismatch");
}

void MVOptimizer::validateData(const MarketData& data) const {
    validateMarketData(data);
}

// ── Portfolio metrics ─────────────────────────────────────────────────────────

PortfolioMetrics MVOptimizer::computeMetrics(const Vector& w,
                                             const Vector& mu,
                                             const Matrix& sigma,
                                             double risk_free_rate) {
    PortfolioMetrics m;
    m.expected_return = w.dot(mu);
    const Vector sigma_w = sigma * w;
    m.variance        = w.dot(sigma_w);
    m.volatility      = std::sqrt(std::max(0.0, m.variance));

    if (m.volatility > 1e-12) {
        m.sharpe_ratio = (m.expected_return - risk_free_rate) / m.volatility;
    } else {
        m.sharpe_ratio = std::numeric_limits<double>::quiet_NaN();
    }

    // Risk contributions: RC_i = w_i (Σw)_i / σ
    const int n = static_cast<int>(w.size());
    m.risk_contribution = Vector::Zero(n);
    if (m.volatility > 1e-12) {
        for (int i = 0; i < n; ++i)
            m.risk_contribution[i] = w[i] * sigma_w[i] / m.volatility;
    }

    // Diversification ratio: (Σ |w_i| σ_i) / σ_p
    if (m.volatility > 1e-12) {
        double weighted_vol = 0.0;
        for (int i = 0; i < n; ++i)
            weighted_vol += std::abs(w[i]) * std::sqrt(std::max(0.0, sigma(i, i)));
        m.diversification_ratio = weighted_vol / m.volatility;
    }

    // Effective number of assets (Herfindahl)
    const double sumsq = w.squaredNorm();
    m.effective_n_assets = (sumsq > 1e-15) ? 1.0 / sumsq
                                            : std::numeric_limits<double>::quiet_NaN();

    return m;
}

void MVOptimizer::augmentBenchmarkMetrics(PortfolioMetrics& metrics,
                                          const Vector& w,
                                          const Vector& mu,
                                          const Matrix& sigma,
                                          const Vector& benchmark,
                                          double risk_free_rate) {
    if (benchmark.size() != w.size()) return;
    const Vector active = w - benchmark;

    const double te_var = active.dot(sigma * active);
    metrics.tracking_error = std::sqrt(std::max(0.0, te_var));

    const double active_ret = active.dot(mu);
    metrics.information_ratio = (metrics.tracking_error > 1e-12)
        ? active_ret / metrics.tracking_error
        : std::numeric_limits<double>::quiet_NaN();

    metrics.active_share = 0.5 * active.cwiseAbs().sum();

    // Beta to benchmark: cov(w'r, b'r) / var(b'r) = w'Σb / b'Σb
    const double bm_var = benchmark.dot(sigma * benchmark);
    metrics.beta_to_benchmark = (bm_var > 1e-12)
        ? w.dot(sigma * benchmark) / bm_var
        : std::numeric_limits<double>::quiet_NaN();

    (void)risk_free_rate;
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

    // QP: min  w'Σw - λμ'w  ≡  min  0.5 w'(2Σ)w + (-λμ)'w
    Matrix Q = 2.0 * data.covariance;
    Vector f = -risk_aversion * data.expected_returns;

    // Turnover penalty: + κ‖w − w₀‖²  →  Q += 2κI,  f += -2κ w₀
    if (cons.turnover_penalty > 0.0 && cons.current_weights.size() == n) {
        const double k2 = 2.0 * cons.turnover_penalty;
        Q.diagonal().array() += k2;
        f.noalias() -= k2 * cons.current_weights;
    }

    solver_cfg_.budget = cons.budget;

    const auto t0 = std::chrono::steady_clock::now();
    qp::SolverResult qp = qp::solveWithGroups(
        Q, f, cons.lower_bounds, cons.upper_bounds,
        cons.groups, params_.group_penalty, solver_cfg_);
    const auto t1 = std::chrono::steady_clock::now();

    OptimizationResult result;
    result.weights    = qp.x;
    result.assets     = data.assets;
    result.converged  = qp.converged;
    result.iterations = qp.iterations;
    result.method     = "MVO";
    result.solve_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.gradient_at_optimum = qp.gradient;
    result.primal_residual = qp.primal_residual;
    result.kkt_residual    = qp.kkt_residual;
    result.dual_estimate   = qp.dual_estimate;
    result.library_version = VERSION_STRING;
    result.input_hash      = inputHash(data);
    result.params_hash     = paramsHash(params_);

    const double rf = effectiveRiskFreeRate(params_, data);
    result.metrics = computeMetrics(qp.x, data.expected_returns,
                                    data.covariance, rf);

    // Turnover diagnostic
    if (cons.current_weights.size() == n) {
        result.metrics.turnover = 0.5 * (qp.x - cons.current_weights).cwiseAbs().sum();
    }

    // Benchmark-relative metrics
    if (data.benchmark_weights.has_value() &&
        data.benchmark_weights->size() == n) {
        augmentBenchmarkMetrics(result.metrics, qp.x, data.expected_returns,
                                data.covariance, *data.benchmark_weights, rf);
    }

    // Active constraints
    for (int i = 0; i < n; ++i) {
        if (std::abs(qp.x[i] - cons.lower_bounds[i]) < tol::kActiveBound)
            result.active_lower_bounds.push_back(i);
        if (std::abs(qp.x[i] - cons.upper_bounds[i]) < tol::kActiveBound)
            result.active_upper_bounds.push_back(i);
    }

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

    if (pts <= 0)
        throw std::invalid_argument("efficientFrontier: frontier_points must be > 0");
    if (lo <= 0.0)
        throw std::invalid_argument(
            "efficientFrontier: min_risk_aversion must be > 0 (got " +
            std::to_string(lo) + ")");
    if (hi <= lo)
        throw std::invalid_argument(
            "efficientFrontier: max_risk_aversion (" + std::to_string(hi) +
            ") must be > min_risk_aversion (" + std::to_string(lo) + ")");

    log::info("MVO efficient frontier: n={} assets, {} points, lambda=[{:.3f},{:.3f}]",
              n, pts, lo, hi);

    EfficientFrontier frontier;
    frontier.assets = data.assets;
    frontier.method = "MVO";
    frontier.points.reserve(static_cast<std::size_t>(pts));

    // Logarithmic sweep: more resolution at low-risk end
    for (int i = 0; i < pts; ++i) {
        const double t = (pts == 1) ? 0.0
                                     : static_cast<double>(i) /
                                       static_cast<double>(pts - 1);
        const double lambda = lo * std::pow(hi / lo, t);

        auto r = optimizeFor(data, lambda);

        EfficientFrontierPoint pt;
        pt.risk_aversion = lambda;
        pt.weights       = r.weights;
        pt.metrics       = r.metrics;
        frontier.points.push_back(std::move(pt));
    }

    return frontier;
}

// ── PM-friendly portfolio constructors ───────────────────────────────────────

OptimizationResult MVOptimizer::minVariancePortfolio(const MarketData& data) {
    validateData(data);
    // λ very small → variance dominates the objective
    auto r = optimizeFor(data, 1e-6);
    r.status_message = "Minimum-variance portfolio (λ → 0)";
    return r;
}

OptimizationResult MVOptimizer::maxSharpePortfolio(const MarketData& data) {
    validateData(data);

    // Sweep λ and pick the highest-Sharpe point, refine via golden-section.
    // This is robust to non-convex behaviour of Sharpe vs λ that arises
    // when bounds are tight.
    const double rf = effectiveRiskFreeRate(params_, data);

    auto sharpe_at = [&](double lam) {
        auto r = optimizeFor(data, lam);
        if (r.metrics.volatility < 1e-12) return -1e30;
        return (r.metrics.expected_return - rf) / r.metrics.volatility;
    };

    // Coarse scan
    const int N = 25;
    const double lo = std::max(params_.min_risk_aversion, 1e-3);
    const double hi = std::max(params_.max_risk_aversion, lo * 100.0);
    double best_lam = lo;
    double best_sh  = -1e30;
    for (int i = 0; i < N; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(N - 1);
        const double lam = lo * std::pow(hi / lo, t);
        const double sh = sharpe_at(lam);
        if (sh > best_sh) { best_sh = sh; best_lam = lam; }
    }

    // Golden-section refine in log-λ around best
    const double phi = 0.6180339887498949;
    double a = std::log(best_lam) - std::log(hi / lo) / static_cast<double>(N - 1);
    double b = std::log(best_lam) + std::log(hi / lo) / static_cast<double>(N - 1);
    a = std::max(a, std::log(lo));
    b = std::min(b, std::log(hi));
    double c = b - phi * (b - a);
    double d = a + phi * (b - a);
    double fc = sharpe_at(std::exp(c));
    double fd = sharpe_at(std::exp(d));
    for (int iter = 0; iter < 25; ++iter) {
        if (fc > fd) {
            b = d; d = c; fd = fc;
            c = b - phi * (b - a);
            fc = sharpe_at(std::exp(c));
        } else {
            a = c; c = d; fc = fd;
            d = a + phi * (b - a);
            fd = sharpe_at(std::exp(d));
        }
        if (std::abs(b - a) < 1e-4) break;
    }
    const double lam_opt = std::exp(0.5 * (a + b));
    auto r = optimizeFor(data, lam_opt);
    r.status_message = "Maximum-Sharpe (tangency) portfolio";
    return r;
}

// ── Frontier-based target search (A4) ─────────────────────────────────────────
//
// Solve `metric(w*(λ)) = target` robustly under constraints that can make
// the λ → metric map non-monotonic. Approach:
//   1. Scan a fine log-λ grid, recording (λ_k, m_k).
//   2. Find every segment [λ_k, λ_{k+1}] where target lies between m_k and
//      m_{k+1} (a "bracket"). When more than one bracket exists, prefer
//      the segment whose midpoint metric is closest to target.
//   3. Bisect inside the chosen segment using the actual sign of the
//      metric difference at the endpoints (no monotonicity assumption).
//   4. Report multi-bracket ambiguity in status_message so the caller can
//      decide whether to trust the answer.

struct TargetSearchResult {
    double lambda;
    int    brackets;     // how many sign-change segments matched
    double m_min;
    double m_max;
};

template <typename MetricFn>
static TargetSearchResult findLambdaForTarget(MetricFn metric,
                                              double target,
                                              double lo_lam,
                                              double hi_lam) {
    const int N = 40;                      // grid resolution
    std::vector<double> log_lams(N), ms(N);
    for (int i = 0; i < N; ++i) {
        const double t = static_cast<double>(i) /
                         static_cast<double>(N - 1);
        log_lams[i] = std::log(lo_lam) +
                      t * (std::log(hi_lam) - std::log(lo_lam));
        ms[i] = metric(std::exp(log_lams[i]));
    }

    double m_min = ms[0], m_max = ms[0];
    for (double v : ms) { m_min = std::min(m_min, v); m_max = std::max(m_max, v); }

    if (target > m_max + 1e-12 || target < m_min - 1e-12) {
        // Out of achievable range — return endpoint closest to target.
        double best_lam = std::exp(log_lams[0]);
        double best_dist = std::abs(ms[0] - target);
        for (int i = 1; i < N; ++i) {
            const double d = std::abs(ms[i] - target);
            if (d < best_dist) { best_dist = d; best_lam = std::exp(log_lams[i]); }
        }
        return {best_lam, 0, m_min, m_max};
    }

    // Find every segment where (m_k - target) changes sign.
    std::vector<int> brackets;
    for (int i = 0; i + 1 < N; ++i) {
        const double a = ms[i]   - target;
        const double b = ms[i+1] - target;
        if (a == 0.0 || b == 0.0 || (a < 0) != (b < 0))
            brackets.push_back(i);
    }
    if (brackets.empty()) {
        // No sign change ⇒ target equals one of the endpoints.
        double best_lam = std::exp(log_lams[0]);
        double best_dist = std::abs(ms[0] - target);
        for (int i = 1; i < N; ++i) {
            const double d = std::abs(ms[i] - target);
            if (d < best_dist) { best_dist = d; best_lam = std::exp(log_lams[i]); }
        }
        return {best_lam, 0, m_min, m_max};
    }

    // Choose the bracket whose midpoint metric is closest to target.
    int chosen = brackets.front();
    double best_score = std::abs(0.5 * (ms[chosen] + ms[chosen+1]) - target);
    for (int idx : brackets) {
        const double score =
            std::abs(0.5 * (ms[idx] + ms[idx+1]) - target);
        if (score < best_score) { best_score = score; chosen = idx; }
    }

    // Bisect within the chosen segment using the actual sign of the
    // metric difference (no global monotonicity assumption).
    double a_log = log_lams[chosen];
    double b_log = log_lams[chosen + 1];
    double a_val = ms[chosen]     - target;
    for (int iter = 0; iter < 60; ++iter) {
        const double mid_log = 0.5 * (a_log + b_log);
        const double mid_val = metric(std::exp(mid_log)) - target;
        if (std::abs(mid_val) < 1e-7) {
            return {std::exp(mid_log),
                    static_cast<int>(brackets.size()), m_min, m_max};
        }
        if ((a_val < 0) == (mid_val < 0)) {
            a_log = mid_log; a_val = mid_val;
        } else {
            b_log = mid_log;
        }
    }
    return {std::exp(0.5 * (a_log + b_log)),
            static_cast<int>(brackets.size()), m_min, m_max};
}

OptimizationResult MVOptimizer::optimizeForTargetVolatility(const MarketData& data,
                                                            double target_volatility) {
    validateData(data);
    if (target_volatility <= 0.0)
        throw std::invalid_argument("target_volatility must be > 0");

    const double lo_lam = std::max(params_.min_risk_aversion, 1e-4);
    const double hi_lam = std::max(params_.max_risk_aversion, lo_lam * 1e4);

    auto vol_at = [&](double lam) {
        return optimizeFor(data, lam).metrics.volatility;
    };

    const auto srch = findLambdaForTarget(vol_at, target_volatility, lo_lam, hi_lam);
    auto r = optimizeFor(data, srch.lambda);
    if (srch.brackets == 0) {
        if (target_volatility > srch.m_max) {
            r.status_message =
                "Target volatility (" + std::to_string(target_volatility) +
                ") exceeds max achievable (" + std::to_string(srch.m_max) +
                "); returning closest portfolio";
        } else {
            r.status_message =
                "Target volatility (" + std::to_string(target_volatility) +
                ") below min achievable (" + std::to_string(srch.m_min) +
                "); returning closest portfolio";
        }
    } else if (srch.brackets > 1) {
        r.status_message =
            "Target-volatility portfolio (warning: " +
            std::to_string(srch.brackets) +
            " bracketing segments — λ → vol is non-monotonic; selected "
            "the bracket nearest the target)";
    } else {
        r.status_message = "Target-volatility portfolio";
    }
    return r;
}

OptimizationResult MVOptimizer::optimizeForTargetReturn(const MarketData& data,
                                                        double target_return) {
    validateData(data);
    const double lo_lam = std::max(params_.min_risk_aversion, 1e-4);
    const double hi_lam = std::max(params_.max_risk_aversion, lo_lam * 1e4);

    auto ret_at = [&](double lam) {
        return optimizeFor(data, lam).metrics.expected_return;
    };

    const auto srch = findLambdaForTarget(ret_at, target_return, lo_lam, hi_lam);
    auto r = optimizeFor(data, srch.lambda);
    if (srch.brackets == 0) {
        if (target_return > srch.m_max) {
            r.status_message =
                "Target return (" + std::to_string(target_return) +
                ") exceeds max achievable (" + std::to_string(srch.m_max) +
                "); returning closest portfolio";
        } else {
            r.status_message =
                "Target return (" + std::to_string(target_return) +
                ") below min achievable (" + std::to_string(srch.m_min) +
                "); returning closest portfolio";
        }
    } else if (srch.brackets > 1) {
        r.status_message =
            "Target-return portfolio (warning: " +
            std::to_string(srch.brackets) +
            " bracketing segments — λ → return is non-monotonic; selected "
            "the bracket nearest the target)";
    } else {
        r.status_message = "Target-return portfolio";
    }
    return r;
}

} // namespace portopt
