#pragma once
/**
 * @file optimizer.hpp
 * @brief Abstract base interface for portfolio optimisers.
 *
 * Both MVOptimizer and BlackLittermanOptimizer implement this interface,
 * allowing them to be used interchangeably in higher-level code.
 */

#include "types.hpp"

namespace portopt {

/**
 * @brief Abstract portfolio optimiser interface.
 *
 * Implementations must provide:
 * - A single optimal portfolio for a given risk aversion
 * - The full efficient frontier
 */
class IOptimizer {
public:
    virtual ~IOptimizer() = default;

    /**
     * @brief Compute the optimal portfolio for a specific risk aversion.
     *
     * @param data   Market data (μ, Σ, optional market weights)
     * @return       Optimisation result with weights and analytics
     */
    virtual OptimizationResult optimize(const MarketData& data) = 0;

    /**
     * @brief Compute the full efficient frontier.
     *
     * Sweeps risk aversion over [min_λ, max_λ] and returns one portfolio
     * per step.
     *
     * @param data   Market data
     * @return       Efficient frontier with all points
     */
    virtual EfficientFrontier efficientFrontier(const MarketData& data) = 0;
};

} // namespace portopt
