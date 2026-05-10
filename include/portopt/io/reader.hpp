#pragma once
/**
 * @file io/reader.hpp
 * @brief Input readers for market data and optimisation parameters.
 *
 * Supports:
 * - Market data (assets + μ + Σ + optional market weights): JSON and CSV
 * - Optimisation parameters: JSON and TOML
 *
 * ### JSON market data schema
 * @code{.json}
 * {
 *   "assets": [
 *     { "ticker": "AAPL", "name": "Apple Inc.",
 *       "expected_return": 0.12, "market_cap": 3e12 }
 *   ],
 *   "covariance": [[0.04, 0.01], [0.01, 0.03]],
 *   "market_weights": [0.6, 0.4]   // optional
 * }
 * @endcode
 *
 * ### CSV market data format
 * First row: comma-separated asset tickers (header)
 * Subsequent rows: covariance matrix (one row per asset)
 * A separate assets.csv can supply ticker,name,expected_return,market_cap
 *
 * ### JSON parameter schema — see docs/user_guide.md for full reference.
 * ### TOML parameter schema — see docs/user_guide.md for full reference.
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
 *
 * @param path    Path to input file (.json or .csv)
 * @param fmt     Format override; Format::Auto infers from extension
 * @return        Populated MarketData struct
 * @throws        std::runtime_error on parse or validation failure
 */
MarketData readMarketData(const std::filesystem::path& path,
                          Format fmt = Format::Auto);

/**
 * @brief Read market data from a JSON string (in-memory).
 */
MarketData readMarketDataFromJSON(const std::string& json_str);

/**
 * @brief Read market data from CSV (returns + covariance as separate files).
 *
 * @param assets_csv     CSV with columns: ticker,name,expected_return,market_cap
 * @param covariance_csv CSV with covariance matrix (header row = tickers)
 * @param weights_csv    Optional CSV with market weights
 */
MarketData readMarketDataFromCSV(
    const std::filesystem::path& assets_csv,
    const std::filesystem::path& covariance_csv,
    const std::filesystem::path* weights_csv = nullptr);

/**
 * @brief Read MVO parameters from a JSON or TOML file.
 *
 * @param path  Path to parameter file (.json or .toml)
 * @return      Populated MVOParameters
 */
MVOParameters readMVOParameters(const std::filesystem::path& path,
                                Format fmt = Format::Auto);

/**
 * @brief Read Black-Litterman parameters from a JSON or TOML file.
 */
BlackLittermanParameters readBLParameters(const std::filesystem::path& path,
                                          Format fmt = Format::Auto);

} // namespace io
} // namespace portopt
