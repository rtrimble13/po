#pragma once
/**
 * @file fx.hpp
 * @brief Multi-currency support (B13).
 *
 * A small set of helpers for portfolios that span multiple quote
 * currencies. The library treats every Σ and μ as already-translated to a
 * single base currency, but downstream consumers often need to:
 *   1. Convert per-asset expected returns from local to base currency,
 *      with or without a forward-hedge.
 *   2. Aggregate the portfolio's net currency exposure (Σ_{i in c} w_i).
 *
 * These helpers are deliberately stateless and operate on the data
 * already attached to MarketData via Asset::currency.
 */

#include "types.hpp"
#include <map>
#include <string>
#include <vector>

namespace portopt {
namespace fx {

/**
 * @brief A view of currency-attribution for a portfolio.
 */
struct CurrencyExposure {
    /// Currency code (e.g. "USD") in the order returned.
    std::vector<std::string> currency;
    /// Net long exposure to that currency (Σ_{i in currency} w_i).
    Vector                   weight;
};

/**
 * @brief Sum portfolio weights by Asset::currency.
 *
 * Assets with an empty currency string are grouped under the supplied
 * @p base_currency label (default "BASE"). The returned vectors are in
 * lexicographic order so the result is reproducible.
 */
CurrencyExposure currencyExposure(const AssetUniverse& assets,
                                   const Vector& weights,
                                   const std::string& base_currency = "BASE");

/**
 * @brief Convert per-asset expected returns from local to base currency.
 *
 * Given a map `fx_returns[c] = E[r_c/base − 1]` (the expected one-period
 * appreciation of currency `c` against the base), the converted expected
 * return for an asset quoted in `c` is
 *     μ_base = (1 + μ_local)·(1 + r_fx) − 1
 * which for small returns is ≈ μ_local + r_fx. We use the exact form.
 *
 * Assets with an empty currency are passed through unchanged.
 *
 * @param assets        Asset list (currency strings consulted).
 * @param mu_local      Expected returns in each asset's local currency.
 * @param fx_returns    Map from currency code to expected FX return.
 * @param hedge_ratio   In [0,1]; 0 = unhedged, 1 = fully hedged (FX
 *                      contribution removed). 0.5 = half-hedged.
 * @return              Translated expected returns.
 */
Vector convertExpectedReturns(const AssetUniverse& assets,
                               const Vector& mu_local,
                               const std::map<std::string, double>& fx_returns,
                               double hedge_ratio = 0.0);

/**
 * @brief Build a per-currency net-exposure group constraint.
 *
 * Emits one GroupConstraint that bounds the portfolio's net weight in
 * @p currency to [@p lower, @p upper]. Useful for FX-policy compliance
 * (e.g. "USD ≤ 60%", "EUR ≥ 20%").
 */
GroupConstraint currencyExposureConstraint(const AssetUniverse& assets,
                                            const std::string& currency,
                                            double lower,
                                            double upper);

} // namespace fx
} // namespace portopt
