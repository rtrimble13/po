#pragma once
/**
 * @file io/writer.hpp
 * @brief Output writers for optimisation results and efficient frontiers.
 *
 * Supports console (human-readable table), JSON, and CSV output.
 *
 * ### JSON output schema
 * @code{.json}
 * {
 *   "method":  "MVO",
 *   "converged": true,
 *   "iterations": 42,
 *   "metrics": { "expected_return": 0.12, "volatility": 0.18,
 *                "sharpe_ratio": 0.67,    "variance": 0.0324 },
 *   "weights": [ { "ticker": "AAPL", "weight": 0.35 }, ... ]
 * }
 * @endcode
 *
 * ### CSV output (one row per asset)
 * @code
 * ticker,name,weight,expected_return,volatility,sharpe_ratio
 * AAPL,Apple Inc.,0.35,...
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
};

// ── Single result ────────────────────────────────────────────────────────────

/**
 * @brief Write optimisation result to an output stream.
 *
 * @param result  Optimisation result
 * @param out     Destination stream
 * @param cfg     Writer configuration
 */
void writeResult(const OptimizationResult& result,
                 std::ostream&             out,
                 const WriterConfig&       cfg = {});

/**
 * @brief Write optimisation result to a file.
 *
 * Format is inferred from the file extension (.json → JSON, .csv → CSV)
 * unless overridden via cfg.format.
 */
void writeResult(const OptimizationResult& result,
                 const std::filesystem::path& path,
                 const WriterConfig&          cfg = {});

/**
 * @brief Serialise OptimizationResult to a JSON string.
 */
std::string resultToJSON(const OptimizationResult& result,
                         int indent = 2);

/**
 * @brief Serialise OptimizationResult to a CSV string.
 */
std::string resultToCSV(const OptimizationResult& result);

// ── Efficient frontier ────────────────────────────────────────────────────────

/**
 * @brief Write efficient frontier to a stream.
 *
 * Console: table of (λ, return, volatility, Sharpe)
 * JSON:    array of frontier points with full weight vectors
 * CSV:     one row per frontier point
 */
void writeFrontier(const EfficientFrontier& frontier,
                   std::ostream&            out,
                   const WriterConfig&      cfg = {});

void writeFrontier(const EfficientFrontier&     frontier,
                   const std::filesystem::path& path,
                   const WriterConfig&          cfg = {});

std::string frontierToJSON(const EfficientFrontier& frontier, int indent = 2);
std::string frontierToCSV(const EfficientFrontier&  frontier);

// ── Black-Litterman diagnostics ───────────────────────────────────────────────

/**
 * @brief Write BL model internals (prior/posterior returns, views).
 */
void writeBLModel(const BLModelOutput& bl,
                  const AssetUniverse& assets,
                  std::ostream&        out,
                  const WriterConfig&  cfg = {});

std::string blModelToJSON(const BLModelOutput& bl,
                          const AssetUniverse& assets,
                          int indent = 2);

} // namespace io
} // namespace portopt
