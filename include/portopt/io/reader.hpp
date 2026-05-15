#pragma once
/**
 * @file io/reader.hpp
 * @brief Input readers for market data, returns series, and parameters.
 *
 * Supports:
 * - Market data (assets + μ + Σ + optional market weights): JSON and CSV
 * - Daily/monthly returns CSV (raw time series) → MarketData via estimation
 * - Optimisation parameters: JSON and TOML
 *
 * ### JSON market data schema
 * @code{.json}
 * {
 *   "assets": [
 *     { "ticker": "AAPL", "name": "Apple Inc.",
 *       "expected_return": 0.12, "market_cap": 3e12, "sector": "Tech" }
 *   ],
 *   "covariance": [[0.04, 0.01], [0.01, 0.03]],
 *   "market_weights":    [0.6, 0.4],   // optional, for BL
 *   "benchmark_weights": [0.5, 0.5],   // optional, for TE / IR
 *   "risk_free_rate":    0.04          // optional
 * }
 * @endcode
 *
 * ### Parameter files
 * Both JSON and TOML may contain `mvo` and `black_litterman` sections.
 * Constraints (`lower_bounds`, `upper_bounds`, `groups`, `current_weights`,
 * `turnover_penalty`) may live under either `mvo` or `black_litterman`.
 */

#include "../types.hpp"
#include <filesystem>
#include <string>

namespace portopt {
namespace io {

/// Format tag for reading operations.
enum class Format { JSON, CSV, TOML, Auto };

/**
 * @brief Infer input format from file extension.
 * @throws std::invalid_argument for unrecognised extensions.
 */
Format inferFormat(const std::filesystem::path& path);

/**
 * @brief Read market data from a file.
 */
MarketData readMarketData(const std::filesystem::path& path,
                          Format fmt = Format::Auto);

/// Read market data from a JSON string (in-memory).
MarketData readMarketDataFromJSON(const std::string& json_str);

/// Read market data from CSV (returns + covariance as separate files).
MarketData readMarketDataFromCSV(
    const std::filesystem::path& assets_csv,
    const std::filesystem::path& covariance_csv,
    const std::filesystem::path* weights_csv = nullptr);

/**
 * @brief Read a returns time-series CSV and estimate μ, Σ.
 *
 * Expected CSV format: first column is date/period (any string), subsequent
 * columns are per-asset returns. Header row must contain ticker labels.
 *
 * Example:
 *   date,AAPL,MSFT,GOOG
 *   2024-01-02,0.0123,-0.0045,0.0089
 *   ...
 *
 * @param returns_csv       Path to returns CSV
 * @param periods_per_year  Annualisation factor (e.g. 252 for daily, 12 for monthly)
 * @param shrinkage         "none" | "linear" | "ledoit-wolf" | "oas"
 * @param shrinkage_delta   Manual δ for shrinkage == "linear"
 */
MarketData readReturnsCSV(const std::filesystem::path& returns_csv,
                          double periods_per_year = 252.0,
                          const std::string& shrinkage = "none",
                          double shrinkage_delta = 0.2);

/**
 * @brief Read MVO parameters from a JSON or TOML file.
 */
MVOParameters readMVOParameters(const std::filesystem::path& path,
                                Format fmt = Format::Auto);

/**
 * @brief Read Black-Litterman parameters from a JSON or TOML file.
 *
 * Constraints under `[black_litterman]` and `[mvo]` are both honoured; the
 * BL-section constraints take precedence.
 */
BlackLittermanParameters readBLParameters(const std::filesystem::path& path,
                                          Format fmt = Format::Auto);

} // namespace io
} // namespace portopt
