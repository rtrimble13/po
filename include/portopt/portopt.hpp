#pragma once
/**
 * @file portopt.hpp
 * @brief Umbrella header — include this file for full library access.
 *
 * @code
 *   #include <portopt/portopt.hpp>
 *
 *   portopt::log::init(portopt::log::Level::Info);
 *
 *   portopt::MarketData data = portopt::io::readMarketData("assets.json");
 *
 *   portopt::MVOParameters params;
 *   params.risk_aversion = 2.0;
 *   params.constraints   = portopt::PortfolioConstraints::longOnly(data.assets.size());
 *
 *   portopt::MVOptimizer opt(params);
 *   auto result   = opt.optimize(data);
 *   auto frontier = opt.efficientFrontier(data);
 *
 *   portopt::io::writeResult(result, std::cout);
 *   portopt::io::writeFrontier(frontier, "frontier.csv");
 * @endcode
 */

#include "types.hpp"
#include "optimizer.hpp"
#include "qp_solver.hpp"
#include "mvo.hpp"
#include "black_litterman.hpp"
#include "estimation.hpp"
#include "portfolios.hpp"
#include "factor_model.hpp"
#include "fx.hpp"
#include "io/reader.hpp"
#include "io/writer.hpp"
#include "logging.hpp"

/// Library version information.
namespace portopt {
    constexpr int    VERSION_MAJOR = 1;
    constexpr int    VERSION_MINOR = 0;
    constexpr int    VERSION_PATCH = 0;
    constexpr char   VERSION_STRING[] = "1.0.0";
}
