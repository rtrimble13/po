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

} // namespace portfolios
} // namespace portopt
