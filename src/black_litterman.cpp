#include <portopt/black_litterman.hpp>
#include <portopt/logging.hpp>

#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace portopt {

BlackLittermanOptimizer::BlackLittermanOptimizer(BlackLittermanParameters params)
    : params_(std::move(params)) {}

// ── Validation ────────────────────────────────────────────────────────────────

void BlackLittermanOptimizer::validateData(const MarketData& data) const {
    const int n = static_cast<int>(data.assets.size());
    if (n == 0)
        throw std::invalid_argument("BL: asset universe is empty");
    if (data.expected_returns.size() != n)
        throw std::invalid_argument("BL: expected_returns size mismatch");
    if (data.covariance.rows() != n || data.covariance.cols() != n)
        throw std::invalid_argument("BL: covariance matrix size mismatch");
    if (!data.market_weights.has_value())
        throw std::invalid_argument(
            "BL: market_weights required to compute equilibrium prior");
    if (data.market_weights->size() != n)
        throw std::invalid_argument("BL: market_weights size mismatch");

    const double mw_sum = data.market_weights->sum();
    if (std::abs(mw_sum - 1.0) > 1e-6)
        throw std::invalid_argument(
            "BL: market_weights must sum to 1 within 1e-6 (got " +
            std::to_string(mw_sum) +
            "). Renormalise w_mkt before passing it to the optimiser.");

    for (const auto& v : params_.views) {
        if (v.pick_vector.size() != n)
            throw std::invalid_argument(
                "BL view \"" + v.description + "\": pick_vector size mismatch");

        if (params_.confidence_mode == ViewConfidenceMode::Variance) {
            if (v.confidence <= 0.0)
                throw std::invalid_argument(
                    "BL view \"" + v.description +
                    "\": confidence (variance) must be > 0");
        } else {
            // Idzorek: confidence ∈ [0, 1]
            if (v.confidence < 0.0 || v.confidence > 1.0)
                throw std::invalid_argument(
                    "BL view \"" + v.description +
                    "\": Idzorek confidence must be in [0, 1] "
                    "(got " + std::to_string(v.confidence) + ")");
        }
    }
}

// ── Idzorek (2005) confidence translation ────────────────────────────────────
//
// For each view k:
//   1. Compute the 100%-confident posterior tilt:
//        d_k_100 = τΣ p_k (p_k'τΣ p_k)⁻¹ (q_k − p_k'π)
//   2. Compute the no-info tilt (ω_k = ∞):
//        d_k_0   = 0
//   3. Interpolate: target_tilt = c_k * d_k_100
//   4. Solve for ω_k so that the actual posterior tilt equals target_tilt:
//        ω_k = (p_k'τΣ p_k) * (1/c_k − 1)
//
// This makes confidence ∈ [0, 1] linear in "how far the posterior moves
// towards the certain answer."

static double idzorekOmegaForView(const Vector& p,
                                  const Matrix& tau_sigma,
                                  double confidence) {
    if (confidence >= 0.999) return 1e-12;       // ≈ 100% confident
    if (confidence <= 1e-6)  return 1e12;        // ≈ 0% confident
    const double p_ts_p = p.dot(tau_sigma * p);
    if (p_ts_p < 1e-15) return 1e12;
    return p_ts_p * (1.0 / confidence - 1.0);
}

// ── Model computation ─────────────────────────────────────────────────────────

BLModelOutput BlackLittermanOptimizer::computeModel(const MarketData& data) const {
    const int n = static_cast<int>(data.assets.size());
    const int k = static_cast<int>(params_.views.size());
    const double tau = params_.tau;
    const double delta = params_.risk_aversion;

    log::debug("BL model: n={} assets, k={} views, tau={:.4f}, delta={:.4f}, "
               "confidence_mode={}",
               n, k, tau, delta,
               params_.confidence_mode == ViewConfidenceMode::Idzorek
                   ? "Idzorek" : "Variance");

    // ── Step 1: Equilibrium prior ─────────────────────────────────────────────
    const Vector& w_mkt = *data.market_weights;
    const Vector pi = delta * data.covariance * w_mkt;

    log::debug("BL prior returns (pi): min={:.4f} max={:.4f}",
               pi.minCoeff(), pi.maxCoeff());

    if (k == 0) {
        log::info("BL: no views specified, returning equilibrium prior");
        BLModelOutput out;
        out.prior_returns        = pi;
        out.posterior_returns    = pi;
        out.posterior_cov        = tau * data.covariance;
        out.blended_cov          = data.covariance + tau * data.covariance;
        out.pick_matrix          = Matrix::Zero(0, n);
        out.view_returns         = Vector::Zero(0);
        out.view_uncertainty     = Matrix::Zero(0, 0);
        out.view_confidence_pct  = Vector::Zero(0);
        out.pick_matrix_min_singular = 0.0;
        out.pick_matrix_rank         = 0;
        // Posterior cov is τΣ here; report its condition number.
        Eigen::SelfAdjointEigenSolver<Matrix> es_prior(out.posterior_cov);
        const double smin_p = es_prior.eigenvalues().minCoeff();
        const double smax_p = es_prior.eigenvalues().maxCoeff();
        out.posterior_condition_number =
            (smin_p > 0.0) ? smax_p / smin_p
                           : std::numeric_limits<double>::infinity();
        return out;
    }

    // ── Step 2: Build P, q, Ω ────────────────────────────────────────────────
    Matrix P(k, n);
    Vector q(k);
    Vector omega_diag(k);
    Vector confidence_pct(k);

    const Matrix tauSigma = tau * data.covariance;

    for (int i = 0; i < k; ++i) {
        P.row(i) = params_.views[i].pick_vector.transpose();
        q[i]     = params_.views[i].expected_return;

        if (params_.confidence_mode == ViewConfidenceMode::Idzorek) {
            omega_diag[i] = idzorekOmegaForView(
                params_.views[i].pick_vector, tauSigma, params_.views[i].confidence);
            confidence_pct[i] = params_.views[i].confidence;
        } else {
            omega_diag[i] = params_.views[i].confidence;
            // Report-style "confidence pct": variance-based mapping
            const double p_ts_p = params_.views[i].pick_vector.dot(
                tauSigma * params_.views[i].pick_vector);
            confidence_pct[i] = (p_ts_p > 1e-15)
                ? p_ts_p / (p_ts_p + omega_diag[i])
                : 0.0;
        }
    }

    const Matrix Omega      = omega_diag.asDiagonal();
    const Matrix Omega_inv  = (1.0 / omega_diag.array()).matrix().asDiagonal();

    log::debug("BL pick matrix P ({}x{}), view returns q range [{:.4f},{:.4f}]",
               k, n, q.minCoeff(), q.maxCoeff());

    // ── Pick-matrix conditioning ─────────────────────────────────────────────
    // Detect linearly dependent / near-collinear views before they surface
    // as an LDLT failure on M further down. A row that lies (approximately)
    // in the row-space of the others has a singular value ≲ 1e-10 · σ_max.
    Eigen::JacobiSVD<Matrix> svd(P);
    const Vector svals = svd.singularValues();
    const double smax_P = (svals.size() > 0) ? svals.maxCoeff() : 0.0;
    const double smin_P = (svals.size() > 0) ? svals.minCoeff() : 0.0;
    const double rank_tol = 1e-10 * std::max(1.0, smax_P);
    int rank_P = 0;
    for (int i = 0; i < svals.size(); ++i)
        if (svals[i] > rank_tol) ++rank_P;
    if (rank_P < k) {
        log::warn("BL: pick matrix P is rank-deficient (rank={} of {} views, "
                  "σ_min/σ_max={:.3e}). Posterior may be ill-conditioned; "
                  "consider removing or merging the redundant view(s).",
                  rank_P, k,
                  (smax_P > 0.0) ? smin_P / smax_P
                                  : std::numeric_limits<double>::quiet_NaN());
    }

    // ── Step 3: Posterior covariance  ─────────────────────────────────────────
    const Matrix tauSigma_inv =
        tauSigma.ldlt().solve(Matrix::Identity(n, n));
    const Matrix M = tauSigma_inv + P.transpose() * Omega_inv * P;

    Eigen::LDLT<Matrix> ldlt(M);
    if (ldlt.info() != Eigen::Success) {
        std::string msg = "BL: posterior precision matrix M = (τΣ)⁻¹ + P'Ω⁻¹P "
                          "is not positive definite";
        if (rank_P < k)
            msg += " (likely cause: " + std::to_string(k - rank_P) +
                   " linearly dependent view(s); drop redundant rows of P)";
        throw std::runtime_error(msg);
    }

    const Matrix Sigma_BL = ldlt.solve(Matrix::Identity(n, n));

    // Posterior condition number (σ_max / σ_min of Σ_BL).
    Eigen::SelfAdjointEigenSolver<Matrix> es(Sigma_BL);
    const double smin_post = es.eigenvalues().minCoeff();
    const double smax_post = es.eigenvalues().maxCoeff();
    const double cond_post = (smin_post > 0.0)
        ? smax_post / smin_post
        : std::numeric_limits<double>::infinity();
    if (cond_post > 1e10)
        log::warn("BL: posterior covariance Σ_BL is ill-conditioned "
                  "(cond={:.2e}); results may be sensitive to view inputs.",
                  cond_post);

    // ── Step 4: Posterior expected returns ────────────────────────────────────
    const Vector rhs   = tauSigma_inv * pi + P.transpose() * (Omega_inv * q);
    const Vector mu_BL = Sigma_BL * rhs;

    log::debug("BL posterior returns: min={:.4f} max={:.4f}",
               mu_BL.minCoeff(), mu_BL.maxCoeff());

    // ── Step 5: Total covariance for MVO ──────────────────────────────────────
    const Matrix blended_cov = data.covariance + Sigma_BL;

    BLModelOutput out;
    out.prior_returns        = pi;
    out.posterior_returns    = mu_BL;
    out.posterior_cov        = Sigma_BL;
    out.blended_cov          = blended_cov;
    out.pick_matrix          = P;
    out.view_returns         = q;
    out.view_uncertainty     = Omega;
    out.view_confidence_pct  = confidence_pct;
    out.pick_matrix_min_singular   = smin_P;
    out.pick_matrix_rank           = rank_P;
    out.posterior_condition_number = cond_post;

    return out;
}

MarketData BlackLittermanOptimizer::buildBLMarketData(
        const MarketData& data, const BLModelOutput& bl) const {
    MarketData bl_data;
    bl_data.assets             = data.assets;
    bl_data.expected_returns   = bl.posterior_returns;
    bl_data.covariance         = bl.blended_cov;
    bl_data.market_weights     = data.market_weights;
    bl_data.benchmark_weights  = data.benchmark_weights;
    bl_data.risk_free_rate     = data.risk_free_rate;
    return bl_data;
}

MVOParameters BlackLittermanOptimizer::resolveMVOParameters() const {
    MVOParameters m = params_.mvo_params;
    // If the user left mvo_params.risk_aversion at the default (1.0) and
    // propagate_risk_aversion is true, inherit δ.
    if (params_.propagate_risk_aversion &&
        std::abs(m.risk_aversion - 1.0) < 1e-12) {
        m.risk_aversion = params_.risk_aversion;
    }
    return m;
}

// ── Public interface ──────────────────────────────────────────────────────────

BLModelOutput BlackLittermanOptimizer::modelOutput(const MarketData& data) const {
    validateData(data);
    return computeModel(data);
}

OptimizationResult BlackLittermanOptimizer::optimize(const MarketData& data) {
    validateData(data);
    const int n = static_cast<int>(data.assets.size());
    log::info("Black-Litterman optimize: n={} assets, {} views",
              n, params_.views.size());

    auto bl = computeModel(data);
    auto bl_data = buildBLMarketData(data, bl);

    MVOptimizer mvo(resolveMVOParameters());
    auto result = mvo.optimize(bl_data);
    result.method = "Black-Litterman";

    log::info("BL result: return={:.4f} vol={:.4f} sharpe={:.4f}",
              result.metrics.expected_return,
              result.metrics.volatility,
              result.metrics.sharpe_ratio);

    return result;
}

EfficientFrontier BlackLittermanOptimizer::efficientFrontier(const MarketData& data) {
    validateData(data);
    log::info("Black-Litterman efficient frontier: n={} assets, {} views",
              data.assets.size(), params_.views.size());

    auto bl = computeModel(data);
    auto bl_data = buildBLMarketData(data, bl);

    MVOptimizer mvo(resolveMVOParameters());
    auto frontier = mvo.efficientFrontier(bl_data);
    frontier.method = "Black-Litterman";
    return frontier;
}

} // namespace portopt
