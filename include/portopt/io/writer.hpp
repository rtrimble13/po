#pragma once
/**
 * @file io/writer.hpp
 * @brief Output writers for optimisation results and efficient frontiers.
 *
 * Supports console (human-readable table), JSON, and CSV output.
 *
 * ### Console rendering
 *
 * Default uses Unicode box-drawing characters (e.g. ═, ─). Set
 * `WriterConfig::ascii_only = true` to fall back to ASCII for terminals
 * that don't render UTF-8 (notably the legacy Windows console).
 *
 * ### JSON output schema (single result)
 * @code{.json}
 * {
 *   "method":  "MVO",
 *   "converged": true,
 *   "iterations": 42,
 *   "solve_time_ms": 1.23,
 *   "metrics": { "expected_return": ..., "volatility": ...,
 *                "sharpe_ratio": ..., "tracking_error": ...,
 *                "information_ratio": ..., "active_share": ...,
 *                "beta_to_benchmark": ..., "turnover": ...,
 *                "diversification_ratio": ..., "effective_n_assets": ... },
 *   "weights": [ { "ticker": "AAPL", "weight": 0.35,
 *                  "risk_contribution": 0.08 }, ... ],
 *   "active_constraints": { "lower": [...indices], "upper": [...indices] }
 * }
 * @endcode
 */

#include "../types.hpp"
#include <filesystem>
#include <ostream>
#include <string>

namespace portopt {
namespace io {

/// Output format selector.
enum class OutputFormat { Console, JSON, CSV };

/// Writer configuration.
struct WriterConfig {
    OutputFormat format{OutputFormat::Console};
    int          json_indent{2};         ///< JSON pretty-print indent spaces
    int          console_weight_prec{4}; ///< Decimal places for weights
    int          console_return_prec{4}; ///< Decimal places for returns
    bool         show_zero_weights{false}; ///< Include assets with w≈0
    double       weight_threshold{1e-6};   ///< Threshold for "near zero"
    bool         ascii_only{false};        ///< Use ASCII separators (no Unicode box chars)
    double       total_capital{0.0};       ///< If > 0, print notional $ per asset
    bool         show_risk_contribution{true};
    bool         explain{false};           ///< Print active bounds + gradient diagnostics
};

// ── Single result ────────────────────────────────────────────────────────────

void writeResult(const OptimizationResult& result,
                 std::ostream&             out,
                 const WriterConfig&       cfg = {});

void writeResult(const OptimizationResult& result,
                 const std::filesystem::path& path,
                 const WriterConfig&          cfg = {});

/// Serialise OptimizationResult to a JSON string.
std::string resultToJSON(const OptimizationResult& result, int indent = 2);

/// Serialise OptimizationResult to a CSV string (header + per-asset rows + metrics).
std::string resultToCSV(const OptimizationResult& result);

// ── Efficient frontier ────────────────────────────────────────────────────────

void writeFrontier(const EfficientFrontier& frontier,
                   std::ostream&            out,
                   const WriterConfig&      cfg = {});

void writeFrontier(const EfficientFrontier&     frontier,
                   const std::filesystem::path& path,
                   const WriterConfig&          cfg = {});

std::string frontierToJSON(const EfficientFrontier& frontier, int indent = 2);
std::string frontierToCSV(const EfficientFrontier&  frontier);

// ── Black-Litterman diagnostics ───────────────────────────────────────────────

void writeBLModel(const BLModelOutput& bl,
                  const AssetUniverse& assets,
                  std::ostream&        out,
                  const WriterConfig&  cfg = {});

std::string blModelToJSON(const BLModelOutput& bl,
                          const AssetUniverse& assets,
                          int indent = 2);

} // namespace io
} // namespace portopt
