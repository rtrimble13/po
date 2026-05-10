#pragma once
/**
 * @file types.hpp
 * @brief Core data types for the portfolio optimization library.
 *
 * Defines all plain-data structures exchanged between the library's
 * components: asset descriptors, constraint specifications, optimization
 * results, and the view inputs required by the Black-Litterman model.
 */

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <optional>
#include <stdexcept>

namespace portopt {

// ── Convenience aliases ──────────────────────────────────────────────────────
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

// ── Asset universe ───────────────────────────────────────────────────────────

/// A single tradeable asset.
struct Asset {
    std::string ticker;       ///< Unique identifier (e.g. "AAPL")
    std::string name;         ///< Human-readable name
    double      expected_return{0.0}; ///< Annualised expected excess return
    double      market_cap{0.0};      ///< Market capitalisation (for BL prior)
};

/// Collection of assets forming the investment universe.
using AssetUniverse = std::vector<Asset>;

// ── Constraints ──────────────────────────────────────────────────────────────

/// Weight bounds and linear constraints for portfolio construction.
struct PortfolioConstraints {
    Vector lower_bounds;   ///< Per-asset weight lower bounds (default 0)
    Vector upper_bounds;   ///< Per-asset weight upper bounds (default 1)
    bool   allow_short_selling{false}; ///< If true, lower_bounds default to -1

    /// Construct default long-only constraints for @p n assets.
    static PortfolioConstraints longOnly(int n) {
        PortfolioConstraints c;
        c.lower_bounds = Vector::Zero(n);
        c.upper_bounds = Vector::Ones(n);
        return c;
    }

    /// Construct constraints allowing shorting up to @p max_short per asset.
    static PortfolioConstraints withShorts(int n, double max_short = 1.0) {
        PortfolioConstraints c;
        c.lower_bounds = Vector::Constant(n, -max_short);
        c.upper_bounds = Vector::Ones(n);
        c.allow_short_selling = true;
        return c;
    }

    void validate(int n) const {
        if (lower_bounds.size() != n || upper_bounds.size() != n)
            throw std::invalid_argument("Constraint dimension mismatch");
        if ((lower_bounds.array() > upper_bounds.array()).any())
            throw std::invalid_argument("lower_bounds must be <= upper_bounds");
    }
};

// ── MVO parameters ───────────────────────────────────────────────────────────

/// Parameters for a single Mean-Variance optimisation run.
struct MVOParameters {
    double risk_aversion{1.0};   ///< λ in min w'Σw − λ μ'w
    int    frontier_points{50};  ///< Number of points on the efficient frontier
    double min_risk_aversion{0.01};
    double max_risk_aversion{100.0};
    PortfolioConstraints constraints;
};

// ── Black-Litterman parameters ───────────────────────────────────────────────

/// A single investor view for the Black-Litterman model.
struct View {
    std::string description;   ///< Human-readable description
    Vector      pick_vector;   ///< Row of the P matrix (asset weights in view)
    double      expected_return;  ///< q: expected excess return
    double      confidence{0.1};  ///< Variance of this view (Omega diagonal)
};

/// Parameters for the Black-Litterman model.
struct BlackLittermanParameters {
    double tau{0.05};           ///< Uncertainty scaling of the prior (τ)
    double risk_aversion{2.5};  ///< Market risk aversion (δ), used to back out π
    std::vector<View> views;    ///< Investor views
    MVOParameters mvo_params;   ///< MVO params applied to BL posterior returns
};

// ── Optimization results ─────────────────────────────────────────────────────

/// Portfolio analytics for a single optimal point.
struct PortfolioMetrics {
    double expected_return{0.0}; ///< μ'w
    double volatility{0.0};      ///< sqrt(w'Σw)
    double sharpe_ratio{0.0};    ///< expected_return / volatility
    double variance{0.0};        ///< w'Σw
};

/// Result of a single portfolio optimisation.
struct OptimizationResult {
    Vector             weights;      ///< Optimal asset weights
    PortfolioMetrics   metrics;      ///< Portfolio analytics
    std::vector<Asset> assets;       ///< Asset descriptors (for labelling)
    bool               converged{false};
    int                iterations{0};
    std::string        method;       ///< "MVO" or "Black-Litterman"
    std::string        status_message;
};

/// A single point on the efficient frontier.
struct EfficientFrontierPoint {
    double risk_aversion;
    Vector weights;
    PortfolioMetrics metrics;
};

/// Full efficient frontier.
struct EfficientFrontier {
    std::vector<EfficientFrontierPoint> points;
    std::vector<Asset>                  assets;
    std::string                         method;
};

// ── Input data ───────────────────────────────────────────────────────────────

/// Raw market data for building μ and Σ.
struct MarketData {
    AssetUniverse       assets;
    Vector              expected_returns;  ///< μ vector (n×1)
    Matrix              covariance;        ///< Σ matrix (n×n)
    std::optional<Vector> market_weights;  ///< w_mkt for BL (optional)
};

} // namespace portopt
