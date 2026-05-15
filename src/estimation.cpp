#include <portopt/estimation.hpp>
#include <portopt/logging.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace portopt {
namespace estimation {

// ── Basic moments ────────────────────────────────────────────────────────────

Vector sampleMean(const Matrix& returns, double periods_per_year) {
    const int T = static_cast<int>(returns.rows());
    if (T == 0)
        throw std::invalid_argument("sampleMean: empty returns matrix");
    return (returns.colwise().mean() * periods_per_year).transpose();
}

Matrix sampleCovariance(const Matrix& returns, bool unbiased,
                        double periods_per_year) {
    const int T = static_cast<int>(returns.rows());
    if (T < 2)
        throw std::invalid_argument("sampleCovariance: need at least 2 periods");

    const Vector mean = returns.colwise().mean().transpose();
    const Matrix centered = returns.rowwise() - mean.transpose();
    const double denom = unbiased ? static_cast<double>(T - 1)
                                   : static_cast<double>(T);
    return (centered.transpose() * centered) * (periods_per_year / denom);
}

// ── Linear shrinkage ─────────────────────────────────────────────────────────

Matrix linearShrinkage(const Matrix& sample_cov, double delta,
                       const std::optional<Matrix>& target) {
    if (delta < 0.0 || delta > 1.0)
        throw std::invalid_argument("linearShrinkage: delta must be in [0, 1]");
    const int n = static_cast<int>(sample_cov.rows());

    Matrix T = target.value_or(Matrix());
    if (T.size() == 0) {
        T = Matrix::Zero(n, n);
        T.diagonal() = sample_cov.diagonal();
    }
    if (T.rows() != n || T.cols() != n)
        throw std::invalid_argument("linearShrinkage: target dimension mismatch");

    return (1.0 - delta) * sample_cov + delta * T;
}

// ── Ledoit-Wolf shrinkage toward constant-correlation target ─────────────────

Matrix ledoitWolfShrinkage(const Matrix& returns, double periods_per_year,
                           double* out_delta) {
    const int T = static_cast<int>(returns.rows());
    const int n = static_cast<int>(returns.cols());
    if (T < 2)
        throw std::invalid_argument("ledoitWolfShrinkage: need at least 2 periods");

    const Vector mean = returns.colwise().mean().transpose();
    const Matrix X = returns.rowwise() - mean.transpose();
    const Matrix S = (X.transpose() * X) / static_cast<double>(T);

    // Constant-correlation target F
    const Vector vars = S.diagonal();
    Vector stds = vars.array().max(0.0).sqrt();
    double avg_corr = 0.0;
    int    pairs = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (stds[i] > 1e-15 && stds[j] > 1e-15) {
                avg_corr += S(i, j) / (stds[i] * stds[j]);
                ++pairs;
            }
        }
    }
    if (pairs > 0) avg_corr /= static_cast<double>(pairs);

    Matrix F(n, n);
    for (int i = 0; i < n; ++i) {
        F(i, i) = vars[i];
        for (int j = 0; j < n; ++j) {
            if (i != j) F(i, j) = avg_corr * stds[i] * stds[j];
        }
    }

    // π = (1/T) Σ_t Σ_{ij} (x_{ti} x_{tj} − S_{ij})²
    double pi_hat = 0.0;
    Matrix X2 = X.array().square();
    Matrix pi_mat = (X2.transpose() * X2) / static_cast<double>(T) -
                    2.0 * S.cwiseProduct(
                        (X.transpose() * X) / static_cast<double>(T)) +
                    S.cwiseProduct(S);
    pi_hat = pi_mat.sum();

    // ρ — off-diagonal contributions w.r.t. constant-correlation target
    // Use the simplified Ledoit-Wolf "pi - rho" form:
    //   rho ≈ Σ_i π_{ii} + Σ_{i≠j} (avg_corr / 2) * (sqrt(S_{jj}/S_{ii}) * π̂_{iijj} + ...)
    // For practical use, approximate rho by the diagonal of pi_mat (this is
    // the most commonly cited form for the constant-correlation target).
    double rho_hat = pi_mat.diagonal().sum();
    // Add off-diagonal cross-term approximation
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            if (stds[i] < 1e-15 || stds[j] < 1e-15) continue;
            const double term =
                0.5 * avg_corr *
                ((stds[j] / stds[i]) * pi_mat(i, i) +
                 (stds[i] / stds[j]) * pi_mat(j, j));
            rho_hat += term;
        }
    }

    // γ = ‖S − F‖_F²
    const double gamma = (S - F).squaredNorm();

    double delta = 0.0;
    if (gamma > 1e-15) {
        delta = (pi_hat - rho_hat) / (gamma * static_cast<double>(T));
    }
    delta = std::clamp(delta, 0.0, 1.0);
    if (out_delta) *out_delta = delta;

    log::debug("Ledoit-Wolf shrinkage: delta={:.4f}, pi={:.3e}, rho={:.3e}, gamma={:.3e}",
               delta, pi_hat, rho_hat, gamma);

    return periods_per_year * ((1.0 - delta) * S + delta * F);
}

// ── Oracle Approximating Shrinkage ───────────────────────────────────────────

Matrix oasShrinkage(const Matrix& returns, double periods_per_year,
                    double* out_delta) {
    const int T = static_cast<int>(returns.rows());
    const int n = static_cast<int>(returns.cols());
    if (T < 2)
        throw std::invalid_argument("oasShrinkage: need at least 2 periods");

    const Vector mean = returns.colwise().mean().transpose();
    const Matrix X = returns.rowwise() - mean.transpose();
    const Matrix S = (X.transpose() * X) / static_cast<double>(T);
    const double mu = S.trace() / static_cast<double>(n);
    const Matrix F = mu * Matrix::Identity(n, n);

    const double trS2 = (S * S).trace();
    const double trS  = S.trace();

    const double num =
        (1.0 - 2.0 / static_cast<double>(n)) * trS2 +
        (trS * trS);
    const double den =
        (static_cast<double>(T) + 1.0 - 2.0 / static_cast<double>(n)) *
        (trS2 - (trS * trS) / static_cast<double>(n));

    double delta = (den > 1e-15) ? num / den : 1.0;
    delta = std::clamp(delta, 0.0, 1.0);
    if (out_delta) *out_delta = delta;

    log::debug("OAS shrinkage: delta={:.4f}", delta);

    return periods_per_year * ((1.0 - delta) * S + delta * F);
}

// ── MarketData factory ───────────────────────────────────────────────────────

MarketData fromReturns(const std::vector<std::string>& tickers,
                       const Matrix& returns,
                       double periods_per_year,
                       const std::string& shrinkage,
                       double shrinkage_delta) {
    const int n = static_cast<int>(returns.cols());
    if (static_cast<int>(tickers.size()) != n)
        throw std::invalid_argument(
            "fromReturns: tickers size (" + std::to_string(tickers.size()) +
            ") != returns columns (" + std::to_string(n) + ")");

    MarketData data;
    data.assets.reserve(n);
    for (int i = 0; i < n; ++i) {
        Asset a;
        a.ticker = tickers[i];
        a.name   = tickers[i];
        data.assets.push_back(std::move(a));
    }
    data.expected_returns = sampleMean(returns, periods_per_year);

    std::string s = shrinkage;
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (s == "none" || s.empty()) {
        data.covariance = sampleCovariance(returns, true, periods_per_year);
    } else if (s == "linear") {
        const Matrix sc = sampleCovariance(returns, true, periods_per_year);
        data.covariance = linearShrinkage(sc, shrinkage_delta);
    } else if (s == "ledoit-wolf" || s == "lw") {
        data.covariance = ledoitWolfShrinkage(returns, periods_per_year);
    } else if (s == "oas") {
        data.covariance = oasShrinkage(returns, periods_per_year);
    } else {
        throw std::invalid_argument(
            "fromReturns: unknown shrinkage \"" + shrinkage +
            "\" (expected: none | linear | ledoit-wolf | oas)");
    }

    // Reflect μ back into per-asset expected_return
    for (int i = 0; i < n; ++i)
        data.assets[i].expected_return = data.expected_returns[i];

    return data;
}

} // namespace estimation
} // namespace portopt
