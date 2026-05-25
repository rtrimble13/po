#include <portopt/portfolios.hpp>
#include <portopt/logging.hpp>

#include <cmath>
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

// ── B3: Equal Risk Contribution (Spinu 2013) ─────────────────────────────────
//
// For the long-only ERC system  w_i (Σw)_i = c  for all i with 1'w = 1,
// fix all w_j (j ≠ i) and solve for w_i. With  a = Σ_{j≠i} Σ_{ij} w_j
// and  b = Σ_{ii},  the equilibrium condition reduces to the quadratic
//   b w_i² + a w_i − c = 0,
// giving w_i = (-a + sqrt(a² + 4 b c)) / (2 b) (positive root).
// The target c is the cross-asset average of w_i (Σw)_i so the
// iteration is scale-invariant; we renormalise to 1'w = 1 every sweep.
//
// Spinu (2013) shows global convergence for any PSD Σ.
Vector equalRiskContribution(const Matrix& covariance,
                              double tolerance,
                              int    max_iters) {
    const int n = static_cast<int>(covariance.rows());
    if (n <= 0 || covariance.cols() != n)
        throw std::invalid_argument(
            "equalRiskContribution: covariance must be a non-empty square matrix");
    for (int i = 0; i < n; ++i)
        if (!(covariance(i, i) > 0.0))
            throw std::invalid_argument(
                "equalRiskContribution: covariance(" + std::to_string(i) +
                "," + std::to_string(i) + ") must be > 0");

    Vector w = Vector::Constant(n, 1.0 / static_cast<double>(n));
    for (int iter = 0; iter < max_iters; ++iter) {
        const Vector w_prev = w;
        for (int i = 0; i < n; ++i) {
            const double b = covariance(i, i);
            // a = (Σw)_i − Σ_ii · w_i (cross-term contribution)
            const double Sigma_w_i = covariance.row(i).dot(w);
            const double a = Sigma_w_i - b * w[i];
            // Target c — use the current per-asset average RC as the
            // shared budget so the iteration is scale-invariant.
            double c = 0.0;
            for (int j = 0; j < n; ++j) c += w[j] * covariance.row(j).dot(w);
            c /= static_cast<double>(n);
            const double w_new =
                (-a + std::sqrt(a * a + 4.0 * b * c)) / (2.0 * b);
            w[i] = w_new;
        }
        // Renormalise to budget = 1.
        const double total = w.sum();
        if (total > 0.0) w /= total;
        if ((w - w_prev).cwiseAbs().maxCoeff() < tolerance)
            return w;
    }
    log::warn("equalRiskContribution did not converge in {} iterations",
              max_iters);
    return w;
}

} // namespace portfolios
} // namespace portopt
