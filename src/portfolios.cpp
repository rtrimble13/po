#include <portopt/portfolios.hpp>

#include <stdexcept>

namespace portopt {
namespace portfolios {

Vector equalWeight(int n) {
    if (n <= 0)
        throw std::invalid_argument("equalWeight: n must be > 0");
    return Vector::Constant(n, 1.0 / static_cast<double>(n));
}

Vector inverseVariance(const Matrix& covariance) {
    const int n = static_cast<int>(covariance.rows());
    if (n <= 0 || covariance.cols() != n)
        throw std::invalid_argument(
            "inverseVariance: covariance must be a non-empty square matrix");
    Vector w(n);
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        const double v = covariance(i, i);
        if (!(v > 0.0))
            throw std::invalid_argument(
                "inverseVariance: covariance(" + std::to_string(i) + "," +
                std::to_string(i) + ") must be > 0");
        w[i] = 1.0 / v;
        total += w[i];
    }
    w /= total;
    return w;
}

Vector inverseVolatility(const Matrix& covariance) {
    const int n = static_cast<int>(covariance.rows());
    if (n <= 0 || covariance.cols() != n)
        throw std::invalid_argument(
            "inverseVolatility: covariance must be a non-empty square matrix");
    Vector w(n);
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        const double v = covariance(i, i);
        if (!(v > 0.0))
            throw std::invalid_argument(
                "inverseVolatility: covariance(" + std::to_string(i) + "," +
                std::to_string(i) + ") must be > 0");
        w[i] = 1.0 / std::sqrt(v);
        total += w[i];
    }
    w /= total;
    return w;
}

Vector marketCapWeighted(const AssetUniverse& assets) {
    const int n = static_cast<int>(assets.size());
    if (n == 0)
        throw std::invalid_argument("marketCapWeighted: empty asset universe");
    Vector w(n);
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        if (assets[i].market_cap < 0.0)
            throw std::invalid_argument(
                "marketCapWeighted: negative market_cap on asset " +
                assets[i].ticker);
        w[i] = assets[i].market_cap;
        total += w[i];
    }
    if (!(total > 0.0))
        throw std::invalid_argument(
            "marketCapWeighted: total market cap is zero — no positive caps");
    w /= total;
    return w;
}

} // namespace portfolios
} // namespace portopt
