#pragma once
/**
 * @file black_litterman.hpp
 * @brief Black-Litterman model (Black & Litterman, 1992).
 *
 * The Black-Litterman model blends a market-equilibrium prior with
 * investor views to produce posterior expected returns, which are then
 * fed into MVO to compute optimal portfolios.
 *
 * ### Model summary
 *
 * **Prior (market equilibrium):**
 * @code
 *   π = δ Σ w_mkt
 * @endcode
 * where δ is the market risk aversion and w_mkt are the market-cap weights.
 *
 * **Views:**
 * @code
 *   P μ ~ N(q, Ω)
 * @endcode
 * where P is the pick matrix (k×n), q is the expected-return vector (k×1),
 * and Ω is the view uncertainty matrix.
 *
 * **Posterior expected returns:**
 * @code
 *   Σ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹
 *   μ_BL = Σ_BL [(τΣ)⁻¹ π + P'Ω⁻¹ q]
 * @endcode
 *
 * The posterior covariance used in MVO is Σ + Σ_BL (parameter uncertainty
 * adds to market risk).
 *
 * ### Ω construction
 *
 * - **ViewConfidenceMode::Variance**: View.confidence is used directly as
 *   the diagonal of Ω. Smaller = more confident. Industry-standard but
 *   unintuitive to set.
 * - **ViewConfidenceMode::Idzorek**: View.confidence ∈ [0,1] is a percentage
 *   confidence level (1 = certain, 0 = no information). Ω is derived using
 *   the Idzorek (2005) procedure that matches the implied tilt against τΣ.
 */

#include "optimizer.hpp"
#include "mvo.hpp"

namespace portopt {

/// Intermediate Black-Litterman model outputs (useful for diagnostics).
struct BLModelOutput {
    Vector prior_returns;       ///< π — market equilibrium returns
    Vector posterior_returns;   ///< μ_BL — posterior expected returns
    Matrix posterior_cov;       ///< Σ_BL — posterior covariance
    Matrix blended_cov;         ///< Σ + Σ_BL — total covariance for MVO
    Matrix pick_matrix;         ///< P
    Vector view_returns;        ///< q
    Matrix view_uncertainty;    ///< Ω (full matrix — diagonal under both modes)
    Vector view_confidence_pct; ///< Reported confidence ∈ [0,1] for each view

    /// Smallest singular value of P. Near-zero (≲ 1e-10) indicates
    /// linearly dependent / collinear views — the posterior is ill-defined.
    double pick_matrix_min_singular{0.0};
    /// Condition number (σ_max / σ_min) of the posterior covariance Σ_BL.
    /// Large values (≳ 1e10) indicate the posterior is ill-conditioned.
    double posterior_condition_number{0.0};
    /// Numerical rank of the pick matrix P (singular values > 1e-10 · σ_max).
    int    pick_matrix_rank{0};
};

/**
 * @brief Black-Litterman Portfolio Optimiser.
 *
 * Example usage:
 * @code
 *   BlackLittermanParameters params;
 *   params.tau = 0.05;
 *   params.risk_aversion = 2.5;
 *   params.confidence_mode = ViewConfidenceMode::Idzorek;
 *   params.views = {
 *       { "Tech outperforms", p1, 0.03, 0.50 },   // 50% confident
 *       { "Bonds underperform", p2, -0.02, 0.75 } // 75% confident
 *   };
 *
 *   BlackLittermanOptimizer bl(params);
 *   auto result   = bl.optimize(data);
 *   auto frontier = bl.efficientFrontier(data);
 *   auto model    = bl.modelOutput(data);  // inspect BL internals
 * @endcode
 */
class BlackLittermanOptimizer : public IOptimizer {
public:
    explicit BlackLittermanOptimizer(BlackLittermanParameters params = {});

    /**
     * @brief Compute the BL-optimal portfolio.
     *
     * Requires market_weights in MarketData (used to build equilibrium prior).
     */
    OptimizationResult optimize(const MarketData& data) override;

    /**
     * @brief Compute the efficient frontier with BL posterior returns.
     */
    EfficientFrontier efficientFrontier(const MarketData& data) override;

    /**
     * @brief Return the raw Black-Litterman model outputs.
     *
     * Useful for diagnostic reporting and the Jupyter notebook.
     */
    BLModelOutput modelOutput(const MarketData& data) const;

    void setParameters(const BlackLittermanParameters& params) { params_ = params; }
    const BlackLittermanParameters& parameters() const { return params_; }

private:
    BlackLittermanParameters params_;

    BLModelOutput computeModel(const MarketData& data) const;
    MarketData    buildBLMarketData(const MarketData& data,
                                    const BLModelOutput& bl) const;
    MVOParameters resolveMVOParameters() const;
    void validateData(const MarketData& data) const;
};

} // namespace portopt
