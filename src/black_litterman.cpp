#include <portopt/black_litterman.hpp>
#include <portopt/logging.hpp>

#include <stdexcept>
#include <cmath>

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

    double mw_sum = data.market_weights->sum();
    if (std::abs(mw_sum - 1.0) > 1e-4)
        throw std::invalid_argument(
            "BL: market_weights must sum to 1 (got " + std::to_string(mw_sum) + ")");

    for (const auto& v : params_.views) {
        if (v.pick_vector.size() != n)
            throw std::invalid_argument(
                "BL view \"" + v.description + "\": pick_vector size mismatch");
        if (v.confidence <= 0.0)
            throw std::invalid_argument(
                "BL view \"" + v.description + "\": confidence must be > 0");
    }
}

// ── Model computation ─────────────────────────────────────────────────────────

BLModelOutput BlackLittermanOptimizer::computeModel(const MarketData& data) const {
    const int n = static_cast<int>(data.assets.size());
    const int k = static_cast<int>(params_.views.size());
    const double tau = params_.tau;
    const double delta = params_.risk_aversion;

    log::debug("BL model: n={} assets, k={} views, tau={:.4f}, delta={:.4f}",
               n, k, tau, delta);

    // ── Step 1: Equilibrium prior ─────────────────────────────────────────────
    // π = δ Σ w_mkt
    const Vector& w_mkt = *data.market_weights;
    Vector pi = delta * data.covariance * w_mkt;

    log::debug("BL prior returns (pi): min={:.4f} max={:.4f}",
               pi.minCoeff(), pi.maxCoeff());

    if (k == 0) {
        // No views: posterior = prior
        log::info("BL: no views specified, returning equilibrium prior");
        BLModelOutput out;
        out.prior_returns    = pi;
        out.posterior_returns = pi;
        out.posterior_cov    = tau * data.covariance;
        out.blended_cov      = data.covariance + tau * data.covariance;
        out.pick_matrix      = Matrix::Zero(0, n);
        out.view_returns     = Vector::Zero(0);
        out.view_uncertainty = Matrix::Zero(0, 0);
        return out;
    }

    // ── Step 2: Build P, q, Ω ────────────────────────────────────────────────
    Matrix P(k, n);
    Vector q(k);
    Vector omega_diag(k);

    for (int i = 0; i < k; ++i) {
        P.row(i) = params_.views[i].pick_vector.transpose();
        q[i]     = params_.views[i].expected_return;
        omega_diag[i] = params_.views[i].confidence;
    }

    Matrix Omega = omega_diag.asDiagonal();
    Matrix Omega_inv = (1.0 / omega_diag.array()).matrix().asDiagonal();

    log::debug("BL pick matrix P ({}x{}), view returns q range [{:.4f},{:.4f}]",
               k, n, q.minCoeff(), q.maxCoeff());

    // ── Step 3: Posterior covariance  ─────────────────────────────────────────
    // Σ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹
    Matrix tauSigma = tau * data.covariance;
    Matrix tauSigma_inv = tauSigma.ldlt().solve(Matrix::Identity(n, n));
    Matrix M = tauSigma_inv + P.transpose() * Omega_inv * P;

    // Use LDLT decomposition for stability
    Eigen::LDLT<Matrix> ldlt(M);
    if (ldlt.info() != Eigen::Success)
        throw std::runtime_error("BL: posterior covariance is not positive definite");

    Matrix Sigma_BL = ldlt.solve(Matrix::Identity(n, n));

    // ── Step 4: Posterior expected returns ────────────────────────────────────
    // μ_BL = Σ_BL [(τΣ)⁻¹ π + P'Ω⁻¹ q]
    Vector rhs = tauSigma_inv * pi + P.transpose() * (Omega_inv * q);
    Vector mu_BL = Sigma_BL * rhs;

    log::debug("BL posterior returns: min={:.4f} max={:.4f}",
               mu_BL.minCoeff(), mu_BL.maxCoeff());

    // ── Step 5: Total covariance for MVO ──────────────────────────────────────
    // Combined = Σ + Σ_BL (accounts for estimation uncertainty)
    Matrix blended_cov = data.covariance + Sigma_BL;

    BLModelOutput out;
    out.prior_returns     = pi;
    out.posterior_returns = mu_BL;
    out.posterior_cov     = Sigma_BL;
    out.blended_cov       = blended_cov;
    out.pick_matrix       = P;
    out.view_returns      = q;
    out.view_uncertainty  = Omega;

    return out;
}

MarketData BlackLittermanOptimizer::buildBLMarketData(
        const MarketData& data, const BLModelOutput& bl) const {
    MarketData bl_data;
    bl_data.assets           = data.assets;
    bl_data.expected_returns = bl.posterior_returns;
    bl_data.covariance       = bl.blended_cov;
    bl_data.market_weights   = data.market_weights;
    return bl_data;
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

    MVOptimizer mvo(params_.mvo_params);
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

    MVOptimizer mvo(params_.mvo_params);
    auto frontier = mvo.efficientFrontier(bl_data);
    frontier.method = "Black-Litterman";
    return frontier;
}

} // namespace portopt
