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
#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <stdexcept>

namespace portopt {

// ── Typed exception hierarchy (C4) ───────────────────────────────────────────
//
// All library failures derive from PortoptError. Each subclass carries a
// stable machine-readable reason code in addition to the human-readable
// message. The codes are intended to be consumed by MCP wrappers / LLM
// agents so they can recover from infeasibility, validate inputs, or
// re-prompt for a different configuration without parsing error strings.

class PortoptError : public std::runtime_error {
public:
    PortoptError(std::string code, std::string message)
        : std::runtime_error(std::move(message)), code_(std::move(code)) {}
    const std::string& code() const noexcept { return code_; }
private:
    std::string code_;
};

class InvalidMarketData : public PortoptError {
public:
    using PortoptError::PortoptError;
};

class InvalidParameters : public PortoptError {
public:
    using PortoptError::PortoptError;
};

class InfeasibleProblem : public PortoptError {
public:
    using PortoptError::PortoptError;
};

class SolverDidNotConverge : public PortoptError {
public:
    using PortoptError::PortoptError;
};

class SolverCancelled : public PortoptError {
public:
    using PortoptError::PortoptError;
};

class SolverTimeout : public PortoptError {
public:
    using PortoptError::PortoptError;
};

// ── Cancellation token (C5) ──────────────────────────────────────────────────
//
// A lightweight, copyable handle that wraps a shared atomic flag. The
// FISTA inner loop checks `isCancellationRequested()` between iterations.
// `SolverConfig` and `MVOParameters` accept an optional CancellationToken
// plus a timeout_ms; when either trips, the solver throws SolverCancelled
// / SolverTimeout with the corresponding reason code.
class CancellationToken {
public:
    CancellationToken() : flag_(std::make_shared<std::atomic<bool>>(false)) {}
    void cancel() { if (flag_) flag_->store(true, std::memory_order_relaxed); }
    bool isCancellationRequested() const {
        return flag_ && flag_->load(std::memory_order_relaxed);
    }
    void reset() { if (flag_) flag_->store(false, std::memory_order_relaxed); }
private:
    std::shared_ptr<std::atomic<bool>> flag_;
};

// ── Convenience aliases ──────────────────────────────────────────────────────
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

// ── Asset universe ───────────────────────────────────────────────────────────

/// A single tradeable asset.
struct Asset {
    std::string ticker;       ///< Unique identifier (e.g. "AAPL")
    std::string name;         ///< Human-readable name
    double      expected_return{0.0}; ///< Annualised expected return (μ — see notes)
    double      market_cap{0.0};      ///< Market capitalisation (for BL prior)
    std::string sector;       ///< Optional group/sector label (e.g. "Technology")
    /// Quote currency, e.g. "USD", "EUR", "JPY". Empty = treated as the base
    /// currency. Used by portopt::fx helpers (B13) to compute hedged
    /// returns and currency exposures.
    std::string currency;
};

/// Collection of assets forming the investment universe.
using AssetUniverse = std::vector<Asset>;

// ── Constraints ──────────────────────────────────────────────────────────────

/// A linear group constraint: lower ≤ a'w ≤ upper.
struct GroupConstraint {
    std::string  description;     ///< Human-readable label (e.g. "Technology weight")
    Vector       coefficients;    ///< Coefficient vector a (size n)
    double       lower{-1e30};    ///< Lower bound (use −∞ to disable)
    double       upper{ 1e30};    ///< Upper bound (use +∞ to disable)
};

/// Weight bounds and linear constraints for portfolio construction.
struct PortfolioConstraints {
    Vector lower_bounds;            ///< Per-asset weight lower bounds (default 0)
    Vector upper_bounds;            ///< Per-asset weight upper bounds (default 1)
    bool   allow_short_selling{false}; ///< If true, lower_bounds default to -1
    double budget{1.0};             ///< Sum of weights (1.0 = fully invested,
                                    ///< 0.0 = dollar-neutral L/S, 0.3 = 130/30 has budget=1 with bound 1.3/-0.3)

    // ── Turnover / rebalancing (L2 penalty) ───────────────────────────────────
    Vector current_weights;         ///< Current portfolio weights w₀ (empty = no penalty)
    double turnover_penalty{0.0};   ///< κ in min ... + κ‖w − w₀‖²

    // ── Linear group constraints (applied via penalty / soft enforcement) ─────
    std::vector<GroupConstraint> groups;

    // ── Tracking-error constraint (B1) ───────────────────────────────────────
    /// Maximum tracking-error variance against the benchmark:
    ///   (w − b)' Σ (w − b) ≤ tracking_error_limit²
    /// 0 = disabled. Requires MarketData::benchmark_weights to be set.
    /// Internally converted to a quadratic group inequality whose enforcement
    /// follows MVOParameters::hard_group_constraints / group_penalty.
    double tracking_error_limit{0.0};

    // ── Gross-exposure / leverage cap (B2) ────────────────────────────────────
    /// Maximum Σ|w_i| ≤ gross_exposure_limit. 0 = disabled.
    /// Approximated via a linear group constraint Σ s_i w_i ≤ L where
    /// s_i = sign(w_i^current) (a fixed-point iteration is run if no
    /// current_weights are supplied).
    double gross_exposure_limit{0.0};

    // ── Convenience helpers ───────────────────────────────────────────────────

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

    /// Construct dollar-neutral long/short bounds (budget = 0).
    static PortfolioConstraints dollarNeutral(int n, double max_gross = 1.0) {
        PortfolioConstraints c;
        c.lower_bounds = Vector::Constant(n, -max_gross);
        c.upper_bounds = Vector::Constant(n,  max_gross);
        c.allow_short_selling = true;
        c.budget = 0.0;
        return c;
    }

    /// Pin asset @p idx to weight @p w (lb = ub = w). Constraints must already be sized.
    void fixWeight(int idx, double w) {
        if (idx < 0 || idx >= static_cast<int>(lower_bounds.size()))
            throw std::invalid_argument("fixWeight: index out of range");
        lower_bounds[idx] = w;
        upper_bounds[idx] = w;
    }

    /// Forbid (zero-weight) asset @p idx — equivalent to fixWeight(idx, 0).
    void forbid(int idx) { fixWeight(idx, 0.0); }

    void validate(int n) const {
        if (lower_bounds.size() != n || upper_bounds.size() != n)
            throw InvalidParameters("constraint_dimension_mismatch",
                                     "Constraint dimension mismatch");
        if ((lower_bounds.array() > upper_bounds.array()).any())
            throw InvalidParameters(
                "lower_exceeds_upper_bound",
                "lower_bounds must be <= upper_bounds");
        // Budget feasibility: sum(lb) ≤ budget ≤ sum(ub)
        double lb_sum = lower_bounds.sum();
        double ub_sum = upper_bounds.sum();
        if (budget < lb_sum - 1e-9 || budget > ub_sum + 1e-9)
            throw InfeasibleProblem(
                "budget_outside_bounds",
                "Infeasible: budget=" + std::to_string(budget) +
                " not in [sum(lb)=" + std::to_string(lb_sum) +
                ", sum(ub)=" + std::to_string(ub_sum) + "]");
        if (current_weights.size() != 0 && current_weights.size() != n)
            throw InvalidParameters("current_weights_size_mismatch",
                                     "current_weights size mismatch");
        if (turnover_penalty < 0.0)
            throw InvalidParameters("negative_turnover_penalty",
                                     "turnover_penalty must be >= 0");
        for (const auto& g : groups) {
            if (g.coefficients.size() != n)
                throw InvalidParameters(
                    "group_coefficient_size_mismatch",
                    "Group constraint \"" + g.description +
                    "\": coefficient size mismatch");
            if (g.lower > g.upper)
                throw InvalidParameters(
                    "group_lower_exceeds_upper",
                    "Group constraint \"" + g.description +
                    "\": lower > upper");
        }
    }
};

// ── MVO parameters ───────────────────────────────────────────────────────────

/// Parameters for a single Mean-Variance optimisation run.
struct MVOParameters {
    double risk_aversion{1.0};         ///< λ in min w'Σw − λ μ'w
    int    frontier_points{50};        ///< Number of points on the efficient frontier
    double min_risk_aversion{0.01};
    double max_risk_aversion{100.0};

    /// Risk-free rate (annualised). Used in Sharpe calculation: (μ_p − r_f)/σ.
    /// If expected_returns are already excess returns, leave this at 0.
    double risk_free_rate{0.0};
    /// If true, `risk_free_rate` overrides MarketData::risk_free_rate even
    /// when the override value is exactly 0.0.
    bool   risk_free_rate_is_set{false};

    /// Penalty weight on group-constraint violations (soft enforcement).
    /// Set to 0 to disable group constraints, larger values enforce more strictly.
    double group_penalty{1e3};

    // ── B6: transaction-cost model ───────────────────────────────────────────
    /// Per-asset linear transaction cost coefficient. Cost paid:
    ///   Σ_i linear_transaction_cost_i · |w_i − w_prev_i|
    /// Requires PortfolioConstraints::current_weights to be populated.
    Vector linear_transaction_cost;
    /// Per-asset quadratic / market-impact coefficient (Almgren-Chriss style).
    /// Cost paid:  Σ_i quadratic_transaction_cost_i · (w_i − w_prev_i)²
    /// This composes additively with the L2 turnover penalty.
    Vector quadratic_transaction_cost;

    /// When true, enforce group constraints **exactly** via the augmented-
    /// Lagrangian method (A5). `group_penalty` is then used as the *initial*
    /// penalty weight κ₀; the multipliers are updated automatically until
    /// every group bound is satisfied to `group_tolerance`.
    bool   hard_group_constraints{false};

    /// Maximum allowed violation per group constraint when
    /// `hard_group_constraints` is true.
    double group_tolerance{1e-6};

    /// When true, max-Sharpe portfolio uses the analytical tangent QP fast
    /// path (A3) whenever the constraints permit it (long-only, budget 1,
    /// no binding upper bounds at the unconstrained tangent). When the fast
    /// path is not applicable, falls back to the log-λ sweep + golden-
    /// section refinement.
    bool   use_tangent_reformulation{true};

    // ── C5: cancellation + timeout ───────────────────────────────────────────
    /// Caller-supplied cancellation handle. If set, the FISTA inner loop
    /// polls it between iterations and throws SolverCancelled on request.
    CancellationToken cancellation;
    /// Soft deadline in milliseconds; 0 disables. When elapsed solver
    /// time exceeds this, the inner loop throws SolverTimeout.
    double timeout_ms{0.0};

    PortfolioConstraints constraints;
};

// ── Black-Litterman parameters ───────────────────────────────────────────────

/// Method used to translate view confidence into the Ω matrix.
enum class ViewConfidenceMode {
    Variance,   ///< View.confidence is interpreted as variance (Ω_ii directly)
    Idzorek     ///< View.confidence ∈ [0,1] is a percentage; Ω computed via Idzorek (2005)
};

/// A single investor view for the Black-Litterman model.
struct View {
    std::string description;      ///< Human-readable description
    Vector      pick_vector;      ///< Row of the P matrix (asset weights in view)
    double      expected_return;  ///< q: expected excess return
    double      confidence{0.1};  ///< Interpretation depends on confidence_mode (see params)
};

/// Parameters for the Black-Litterman model.
struct BlackLittermanParameters {
    double tau{0.05};                  ///< Uncertainty scaling of the prior (τ)
    double risk_aversion{2.5};         ///< Market risk aversion (δ), used to back out π
    std::vector<View> views;           ///< Investor views
    MVOParameters mvo_params;          ///< MVO params applied to BL posterior returns
    ViewConfidenceMode confidence_mode{ViewConfidenceMode::Variance};
    bool propagate_risk_aversion{true};///< If true, mvo_params.risk_aversion defaults to risk_aversion
                                       ///< when mvo_params.risk_aversion == 1.0 (its default).
};

// ── Optimization results ─────────────────────────────────────────────────────

/// Portfolio analytics for a single optimal point.
struct PortfolioMetrics {
    double expected_return{0.0};        ///< μ'w
    double volatility{0.0};             ///< sqrt(w'Σw)
    double sharpe_ratio{0.0};           ///< (μ'w − r_f) / σ
    double variance{0.0};               ///< w'Σw

    // Risk decomposition
    Vector risk_contribution;           ///< RC_i = w_i (Σw)_i / σ  — sums to σ
    double diversification_ratio{0.0};  ///< (Σ w_i σ_i) / σ_p  (higher = more diversified)
    double effective_n_assets{0.0};     ///< 1 / Σ w_i² (Herfindahl-based)

    // Benchmark-relative (NaN if no benchmark)
    double tracking_error{std::numeric_limits<double>::quiet_NaN()};
    double information_ratio{std::numeric_limits<double>::quiet_NaN()};
    double active_share{std::numeric_limits<double>::quiet_NaN()};
    double beta_to_benchmark{std::numeric_limits<double>::quiet_NaN()};

    // Trading
    double turnover{std::numeric_limits<double>::quiet_NaN()}; ///< ‖w − w_prev‖₁ / 2 if w_prev given
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
    double             solve_time_ms{0.0};

    // Diagnostics for "explain" mode
    Vector             gradient_at_optimum;   ///< Σw − λμ at the solution
    std::vector<int>   active_lower_bounds;   ///< Indices where w_i = lb_i
    std::vector<int>   active_upper_bounds;   ///< Indices where w_i = ub_i

    // Solver convergence diagnostics
    double             primal_residual{0.0};  ///< ‖w_{k+1} − w_k‖ at exit
    double             kkt_residual{0.0};     ///< L∞ KKT optimality residual (see qp_solver.hpp)
    double             dual_estimate{0.0};    ///< Estimated multiplier ν̂ on the budget constraint

    // Audit trail (B16) — populated by the optimiser; stable across runs
    // with identical inputs and library version. Use to verify
    // reproducibility / detect data or parameter drift between runs.
    std::string        library_version;    ///< portopt version string at solve time
    std::string        input_hash;         ///< Hex digest of (μ, Σ, w_mkt, b, rf)
    std::string        params_hash;        ///< Hex digest of (λ, lb, ub, budget, …)
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
    Vector              expected_returns;     ///< μ vector (n×1)
    Matrix              covariance;           ///< Σ matrix (n×n)
    std::optional<Vector> market_weights;     ///< w_mkt for BL (optional)
    std::optional<Vector> benchmark_weights;  ///< b for tracking-error / IR / active share
    double              risk_free_rate{0.0};  ///< Annualised r_f, used in Sharpe
};

} // namespace portopt
