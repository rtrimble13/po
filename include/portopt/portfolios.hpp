#pragma once
/**
 * @file portfolios.hpp
 * @brief Convenience portfolio constructors (B15).
 *
 * Closed-form / heuristic weight constructions that don't require the
 * full MVO machinery. Used as institutional baselines (equal-weight,
 * inverse-variance) and as feasibility checks.
 */

#include "types.hpp"

namespace portopt {
namespace portfolios {

/// Equal-weight long-only portfolio: w_i = 1/n.
Vector equalWeight(int n);

/// Inverse-variance: w_i ∝ 1/σ_ii, normalised to sum to 1.
/// Long-only. Useful as a benchmark / risk-budget baseline.
Vector inverseVariance(const Matrix& covariance);

/// Inverse-volatility (a.k.a. "risk-weighted"): w_i ∝ 1/σ_i.
/// Approximates risk parity when correlations are low.
Vector inverseVolatility(const Matrix& covariance);

/// Market-cap weighted portfolio: w_i ∝ market_cap_i.
/// Throws if no asset has a positive market_cap.
Vector marketCapWeighted(const AssetUniverse& assets);

/**
 * @brief Equal Risk Contribution (risk parity) portfolio.
 *
 * Finds long-only weights such that w_i · (Σw)_i is equal across all
 * assets, normalised so 1'w = 1. Uses the cyclic coordinate
 * iteration of Spinu (2013): each step solves a 1-D quadratic in
 * w_i with the others fixed, converging globally for any PSD Σ.
 *
 * @param covariance   Symmetric positive (semi-)definite Σ
 * @param tolerance    Convergence threshold on ‖w_{k+1} − w_k‖
 * @param max_iters    Hard cap on iterations
 * @return             Long-only weights summing to 1 with equal risk
 *                     contributions
 */
Vector equalRiskContribution(const Matrix& covariance,
                              double tolerance = 1e-8,
                              int    max_iters = 5000);

/**
 * @brief Hierarchical Risk Parity (B4) — López de Prado (2016).
 *
 * Three-step procedure:
 *   1. Hierarchical clustering of assets by correlation-distance
 *      d_ij = sqrt(0.5·(1 − ρ_ij)). Linkage: single-linkage agglomerative.
 *   2. Quasi-diagonalisation: reorder assets so similar ones are adjacent.
 *   3. Recursive bisection: walk the cluster tree top-down; at each split,
 *      allocate weights inversely proportional to the cluster variance
 *      (computed from inverse-variance within the cluster).
 *
 * Pure linear algebra — no optimisation, no constraints beyond long-only
 * and budget=1. Robust to ill-conditioned covariances where MVO struggles.
 *
 * @param covariance   Symmetric Σ; diagonal must be strictly positive.
 * @return             Long-only weights summing to 1.
 */
Vector hierarchicalRiskParity(const Matrix& covariance);

/**
 * @brief Maximum-diversification portfolio (Choueifaty & Coignard, 2008).
 *
 * Maximises the diversification ratio (Σ w_i σ_i)/σ_p subject to
 * long-only and 1'w = 1. Equivalent to MVO with μ_i = σ_i and λ → ∞.
 * Useful when forward-looking returns are unreliable but risk is.
 *
 * @param covariance   Symmetric PSD Σ.
 * @return             Long-only weights summing to 1.
 */
Vector maximumDiversification(const Matrix& covariance);

/**
 * @brief Resampled MVO weights (Michaud, 1998) — bootstrap-averaged MVO.
 *
 * Draws @p n_resamples synthetic return histories from N(μ, Σ/n_samples),
 * runs MVO on each, and averages the resulting weights. Mitigates
 * estimation error in μ and Σ.
 *
 * @param mu            Sample mean returns.
 * @param sigma         Sample covariance.
 * @param risk_aversion λ for each MVO sub-problem.
 * @param n_samples     Number of "observations" per resampled history.
 * @param n_resamples   Number of bootstrap draws (50–500 typical).
 * @param seed          RNG seed for reproducibility (default 42).
 * @return              Averaged long-only weights summing to 1.
 */
Vector resampledMVO(const Vector& mu,
                     const Matrix& sigma,
                     double risk_aversion = 2.0,
                     int    n_samples     = 252,
                     int    n_resamples   = 100,
                     unsigned seed        = 42);

} // namespace portfolios
} // namespace portopt
