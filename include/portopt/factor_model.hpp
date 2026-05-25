#pragma once
/**
 * @file factor_model.hpp
 * @brief Factor risk model (B7) and factor-neutrality helpers (B8).
 *
 * A multi-factor risk model decomposes asset returns as
 *     r = B·f + ε
 * where B (n×k) are factor loadings, f (k×1) are factor returns with
 * covariance Ω_F (k×k), and ε ∈ R^n is idiosyncratic / specific risk
 * with diagonal covariance D = diag(d_1,…,d_n). The reconstructed asset
 * covariance is
 *     Σ = B·Ω_F·B' + D.
 *
 * This module provides:
 *   - the FactorRiskModel struct and its Σ-reconstruction helper,
 *   - factor-risk decomposition for a given portfolio (B7),
 *   - factor-neutrality constraint generators (B8) — for any sub-vector
 *     of factor loadings, emit GroupConstraints that pin the portfolio's
 *     net loading to a target range, e.g. β-neutral, sector-neutral,
 *     style-neutral.
 */

#include "types.hpp"
#include <string>
#include <vector>

namespace portopt {

/// Linear factor risk model.
struct FactorRiskModel {
    /// Asset-by-factor loading matrix (n×k).
    Matrix loadings;
    /// Factor covariance Ω_F (k×k).
    Matrix factor_covariance;
    /// Per-asset idiosyncratic variances d_i ≥ 0 (length n).
    Vector specific_variance;
    /// Optional human-readable factor names (length k).
    std::vector<std::string> factor_names;

    /// Validate dimensions; throw std::invalid_argument on mismatch.
    void validate() const;

    /// Reconstruct the full asset covariance Σ = B·Ω_F·B' + D.
    Matrix assetCovariance() const;

    /// Number of assets (n).
    int numAssets() const { return static_cast<int>(loadings.rows()); }
    /// Number of factors (k).
    int numFactors() const { return static_cast<int>(loadings.cols()); }
};

/// Factor-attribution of portfolio risk.
struct FactorRiskDecomposition {
    /// Net factor exposure of the portfolio:  e = B'w  (k×1).
    Vector factor_exposures;
    /// Per-factor variance contribution  v_k = (Ω_F·e)_k · e_k  (k×1).
    Vector factor_variance;
    /// Total systematic variance = Σ_k factor_variance_k.
    double systematic_variance{0.0};
    /// Specific variance = w' D w.
    double specific_variance{0.0};
    /// Total variance = systematic + specific.
    double total_variance{0.0};
};

/**
 * @brief Decompose portfolio risk into systematic-by-factor and specific.
 */
FactorRiskDecomposition decomposeRisk(const FactorRiskModel& model,
                                       const Vector& weights);

/**
 * @brief Build a factor-neutral group constraint (B8).
 *
 * Emits a GroupConstraint that bounds the portfolio's net loading on a
 * single factor: `lower ≤ b_k'w ≤ upper` where b_k is the k-th column of
 * @p model.loadings.
 *
 * @param model    Loaded factor risk model.
 * @param factor_index  Column of `loadings` to constrain.
 * @param lower    Lower bound on net exposure (default 0).
 * @param upper    Upper bound on net exposure (default 0).
 * @param description  Optional human-readable label.
 */
GroupConstraint factorNeutralConstraint(const FactorRiskModel& model,
                                         int factor_index,
                                         double lower = 0.0,
                                         double upper = 0.0,
                                         std::string description = "");

/**
 * @brief Build a beta-neutral constraint against the benchmark.
 *
 * Computes β_i = Cov(r_i, r_b)/Var(r_b) using the model's Σ and the supplied
 * benchmark weights, then emits a group constraint  lower ≤ β'w ≤ upper.
 */
GroupConstraint betaNeutralConstraint(const FactorRiskModel& model,
                                       const Vector& benchmark_weights,
                                       double lower = -0.05,
                                       double upper =  0.05);

} // namespace portopt
