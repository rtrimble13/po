#include <portopt/fx.hpp>

#include <algorithm>
#include <stdexcept>

namespace portopt {
namespace fx {

namespace {
std::string currencyOrBase(const std::string& c, const std::string& base) {
    return c.empty() ? base : c;
}
}

CurrencyExposure currencyExposure(const AssetUniverse& assets,
                                   const Vector& weights,
                                   const std::string& base_currency) {
    const int n = static_cast<int>(assets.size());
    if (weights.size() != n)
        throw std::invalid_argument(
            "currencyExposure: weights size does not match asset count");

    // Map currency → cumulative weight; std::map gives deterministic order.
    std::map<std::string, double> by_currency;
    for (int i = 0; i < n; ++i) {
        by_currency[currencyOrBase(assets[i].currency, base_currency)] +=
            weights[i];
    }

    CurrencyExposure out;
    out.currency.reserve(by_currency.size());
    out.weight = Vector(static_cast<int>(by_currency.size()));
    int k = 0;
    for (const auto& kv : by_currency) {
        out.currency.push_back(kv.first);
        out.weight[k++] = kv.second;
    }
    return out;
}

Vector convertExpectedReturns(const AssetUniverse& assets,
                               const Vector& mu_local,
                               const std::map<std::string, double>& fx_returns,
                               double hedge_ratio) {
    const int n = static_cast<int>(assets.size());
    if (mu_local.size() != n)
        throw std::invalid_argument(
            "convertExpectedReturns: mu_local size mismatch");
    if (hedge_ratio < 0.0 || hedge_ratio > 1.0)
        throw std::invalid_argument(
            "convertExpectedReturns: hedge_ratio must be in [0, 1]");

    Vector mu_base(n);
    for (int i = 0; i < n; ++i) {
        const std::string& c = assets[i].currency;
        if (c.empty()) { mu_base[i] = mu_local[i]; continue; }
        const auto it = fx_returns.find(c);
        const double r_fx = (it == fx_returns.end()) ? 0.0 : it->second;
        const double r_fx_eff = (1.0 - hedge_ratio) * r_fx;
        mu_base[i] =
            (1.0 + mu_local[i]) * (1.0 + r_fx_eff) - 1.0;
    }
    return mu_base;
}

GroupConstraint currencyExposureConstraint(const AssetUniverse& assets,
                                            const std::string& currency,
                                            double lower,
                                            double upper) {
    const int n = static_cast<int>(assets.size());
    GroupConstraint g;
    g.description  = "currency=" + currency;
    g.coefficients = Vector::Zero(n);
    for (int i = 0; i < n; ++i)
        if (assets[i].currency == currency)
            g.coefficients[i] = 1.0;
    g.lower = lower;
    g.upper = upper;
    return g;
}

} // namespace fx
} // namespace portopt
