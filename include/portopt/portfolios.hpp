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

} // namespace portfolios
} // namespace portopt
