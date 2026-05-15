#pragma once
/**
 * @file mvo.hpp
 * @brief Mean-Variance Optimisation (Markowitz, 1952).
 *
 * Solves the classic portfolio selection problem:
 *
 *   min  w' Σ w − λ μ' w  (+ optional turnover / group penalties)
 *   s.t. 1'w = b              (budget; default 1)
 *        lb_i <= w_i <= ub_i
 *
 * where Σ is the asset covariance matrix, μ is the vector of expected
 * returns, λ is the risk-aversion parameter, and b is the budget.
 * Sweeping λ traces the efficient frontier.
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
 *
 *   // PM-friendly helpers
 *   auto min_var  = opt.minVariancePortfolio(data);
 *   auto max_shp  = opt.maxSharpePortfolio(data);
 *   auto tgt_vol  = opt.optimizeForTargetVolatility(data, 0.15); // 15% vol
 *   auto tgt_ret  = opt.optimizeForTargetReturn(data, 0.10);     // 10% return
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

    // ── PM-friendly portfolio constructors ───────────────────────────────────

    /// Minimum-variance portfolio (λ → 0).
    OptimizationResult minVariancePortfolio(const MarketData& data);

    /// Maximum-Sharpe (tangency) portfolio. Found by binary search over λ.
    OptimizationResult maxSharpePortfolio(const MarketData& data);

    /// Optimal portfolio with realised volatility ≈ @p target_volatility.
    /// Binary-search over λ on the efficient frontier.
    OptimizationResult optimizeForTargetVolatility(const MarketData& data,
                                                    double target_volatility);

    /// Optimal portfolio with expected return ≈ @p target_return.
    OptimizationResult optimizeForTargetReturn(const MarketData& data,
                                                double target_return);

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
     * @param risk_free_rate  Risk-free rate for Sharpe (default 0)
     * @return         Computed PortfolioMetrics (return, vol, Sharpe, variance, RC, …)
     */
    static PortfolioMetrics computeMetrics(const Vector& weights,
                                           const Vector& mu,
                                           const Matrix& sigma,
                                           double risk_free_rate = 0.0);

    /**
     * @brief Augment a metrics block with benchmark-relative quantities.
     *
     * Updates tracking_error, information_ratio, active_share, beta_to_benchmark.
     */
    static void augmentBenchmarkMetrics(PortfolioMetrics& metrics,
                                        const Vector& weights,
                                        const Vector& mu,
                                        const Matrix& sigma,
                                        const Vector& benchmark_weights,
                                        double risk_free_rate = 0.0);

    /**
     * @brief Validate covariance PSD-ness and other inputs.
     *
     * Throws std::invalid_argument with descriptive message on failure.
     */
    static void validateMarketData(const MarketData& data);

private:
    MVOParameters    params_;
    qp::SolverConfig solver_cfg_;

    OptimizationResult optimizeFor(const MarketData& data, double risk_aversion);
    void validateData(const MarketData& data) const;
};

} // namespace portopt
