#pragma once
/**
 * @file estimation.hpp
 * @brief Covariance and expected-return estimators from a returns time series.
 *
 * The portopt MVO/BL optimisers consume μ and Σ directly. In practice users
 * have a returns matrix R (T×n) — these helpers transform it into a
 * MarketData object, optionally applying covariance shrinkage.
 *
 * ### Available estimators
 *
 * | Function                | Description                                        |
 * |-------------------------|----------------------------------------------------|
 * | `sampleCovariance`      | Plain MLE (1/T) or unbiased (1/(T-1))              |
 * | `ledoitWolfShrinkage`   | Ledoit-Wolf shrinkage toward constant-correlation  |
 * | `oasShrinkage`          | Oracle Approximating Shrinkage (Chen 2010)         |
 * | `linearShrinkage`       | Manual shrinkage with caller-supplied δ ∈ [0,1]    |
 * | `sampleMean`            | Annualised sample mean of returns                  |
 *
 * Output frequencies are annualised by multiplying μ by `periods_per_year`
 * and Σ by `periods_per_year`. Adjust accordingly for daily/weekly/monthly.
 */

#include "types.hpp"

namespace portopt {
namespace estimation {

/**
 * @brief Plain sample covariance of a returns matrix.
 *
 * @param returns           Returns matrix (T rows = periods, n columns = assets)
 * @param unbiased          If true, divide by (T-1); else by T (MLE)
 * @param periods_per_year  Annualisation factor (e.g. 252 for daily)
 * @return                  Annualised covariance matrix (n×n)
 */
Matrix sampleCovariance(const Matrix& returns,
                        bool unbiased = true,
                        double periods_per_year = 1.0);

/**
 * @brief Annualised sample mean of returns.
 */
Vector sampleMean(const Matrix& returns, double periods_per_year = 1.0);

/**
 * @brief Manual linear shrinkage of Σ toward a target T:
 *
 *   Σ_shrunk = (1 − δ) Σ_sample + δ T
 *
 * Default target is the diagonal of Σ_sample (i.e. shrink off-diagonals).
 *
 * @param sample_cov  Sample covariance (n×n)
 * @param delta       Shrinkage intensity ∈ [0, 1]
 * @param target      Optional explicit target matrix; defaults to diag(Σ_sample)
 */
Matrix linearShrinkage(const Matrix& sample_cov,
                       double delta,
                       const std::optional<Matrix>& target = std::nullopt);

/**
 * @brief Ledoit-Wolf (2004) shrinkage toward the constant-correlation target.
 *
 * Computes the optimal shrinkage intensity δ* analytically from the data and
 * returns the shrunk covariance. Set @p out_delta to retrieve δ*.
 *
 * @param returns           Returns matrix (T×n)
 * @param periods_per_year  Annualisation factor
 * @param out_delta         If non-null, receives the optimal δ ∈ [0,1]
 */
Matrix ledoitWolfShrinkage(const Matrix& returns,
                           double periods_per_year = 1.0,
                           double* out_delta = nullptr);

/**
 * @brief Oracle Approximating Shrinkage (Chen et al. 2010) toward the
 *        scaled-identity target (μ̂ * I).
 *
 * Robust for small T relative to n. Returns shrunk covariance.
 */
Matrix oasShrinkage(const Matrix& returns,
                    double periods_per_year = 1.0,
                    double* out_delta = nullptr);

/**
 * @brief Exponentially-weighted covariance (RiskMetrics-style).
 *
 * Weights the most recent observation by (1 − λ), the one before by
 * λ (1 − λ), … Σ_t = (1 − λ) · Σ_{k=0..T-1} λ^k · x_{T-1-k} x_{T-1-k}'.
 * The mean is removed using the same exponential weighting. Equivalent
 * to a discount factor `lambda` on the previous covariance with shock
 * (1 − λ) · r_t r_t' each step (the standard RiskMetrics recursion).
 *
 * @param returns           Returns matrix (T×n)
 * @param lambda            Decay factor ∈ (0, 1) — RiskMetrics uses 0.94
 *                          for daily, 0.97 for monthly
 * @param periods_per_year  Annualisation factor
 */
Matrix ewmaCovariance(const Matrix& returns,
                      double lambda = 0.94,
                      double periods_per_year = 1.0);

/**
 * @brief Convenience: build a MarketData from a returns matrix.
 *
 * @param tickers           Asset tickers (length n)
 * @param returns           Returns matrix (T×n)
 * @param periods_per_year  Annualisation factor
 * @param shrinkage         "none" | "linear" | "ledoit-wolf" | "oas"
 * @param shrinkage_delta   Used only when shrinkage == "linear" (else ignored)
 */
MarketData fromReturns(const std::vector<std::string>& tickers,
                       const Matrix& returns,
                       double periods_per_year = 1.0,
                       const std::string& shrinkage = "none",
                       double shrinkage_delta = 0.2);

} // namespace estimation
} // namespace portopt
