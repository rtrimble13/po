#pragma once
/**
 * @file mvo.hpp
 * @brief Mean-Variance Optimisation (Markowitz, 1952).
 *
 * Solves the classic portfolio selection problem:
 *
 *   min  w' Σ w − λ μ' w
 *   s.t. 1'w = 1
 *        lb_i <= w_i <= ub_i
 *
 * where Σ is the asset covariance matrix, μ is the vector of expected excess
 * returns, and λ is the risk-aversion parameter.  Sweeping λ traces the
 * efficient frontier from minimum-variance to maximum-return portfolios.
 */

#include "optimizer.hpp"
#include "qp_solver.hpp"

namespace portopt {

/**
 * @brief Mean-Variance Optimiser.
 *
 * Example usage:
 * @code
 *   MarketData data = ...;
 *   MVOParameters params;
 *   params.risk_aversion = 2.0;
 *   params.constraints   = PortfolioConstraints::longOnly(n);
 *
 *   MVOptimizer opt(params);
 *   auto result   = opt.optimize(data);
 *   auto frontier = opt.efficientFrontier(data);
 * @endcode
 */
class MVOptimizer : public IOptimizer {
public:
    explicit MVOptimizer(MVOParameters params = {});

    /**
     * @brief Compute the MVO-optimal portfolio.
     *
     * Uses risk_aversion from the constructor parameters.
     * Constraints default to long-only if not set.
     */
    OptimizationResult optimize(const MarketData& data) override;

    /**
     * @brief Compute the efficient frontier.
     *
     * Sweeps λ logarithmically from params.min_risk_aversion to
     * params.max_risk_aversion using params.frontier_points steps.
     */
    EfficientFrontier efficientFrontier(const MarketData& data) override;

    /// Update optimiser parameters.
    void setParameters(const MVOParameters& params) { params_ = params; }

    /// Return current parameters.
    const MVOParameters& parameters() const { return params_; }

    /**
     * @brief Compute portfolio analytics for given weights.
     *
     * @param weights  Weight vector (n×1)
     * @param mu       Expected returns (n×1)
     * @param sigma    Covariance matrix (n×n)
     * @return         Computed PortfolioMetrics
     */
    static PortfolioMetrics computeMetrics(const Vector& weights,
                                           const Vector& mu,
                                           const Matrix& sigma);

private:
    MVOParameters    params_;
    qp::SolverConfig solver_cfg_;

    OptimizationResult optimizeFor(const MarketData& data, double risk_aversion);
    void validateData(const MarketData& data) const;
};

} // namespace portopt
