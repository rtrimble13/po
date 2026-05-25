#include <portopt/estimation.hpp>
#include <portopt/logging.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace portopt {
namespace estimation {

// Reject NaN / Inf in a returns matrix with an actionable message.
// We do not silently impute or use pairwise covariance here — callers
// should clean missing data upstream (drop / forward-fill / EM).
static void rejectNonFinite(const Matrix& returns, const char* fn) {
    if (!returns.allFinite()) {
        // Locate the first offending cell to give the caller a hint.
        const int T = static_cast<int>(returns.rows());
        const int n = static_cast<int>(returns.cols());
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < n; ++i)
                if (!std::isfinite(returns(t, i)))
                    throw std::invalid_argument(
                        std::string(fn) +
                        ": returns matrix contains non-finite value at row " +
                        std::to_string(t) + ", column " + std::to_string(i) +
                        " (value=" + std::to_string(returns(t, i)) +
                        "). Drop or impute missing observations before "
                        "passing returns to the estimator.");
    }
}

// ── Basic moments ────────────────────────────────────────────────────────────

Vector sampleMean(const Matrix& returns, double periods_per_year) {
    const int T = static_cast<int>(returns.rows());
    if (T == 0)
        throw std::invalid_argument("sampleMean: empty returns matrix");
    rejectNonFinite(returns, "sampleMean");
    return (returns.colwise().mean() * periods_per_year).transpose();
}

Matrix sampleCovariance(const Matrix& returns, bool unbiased,
                        double periods_per_year) {
    const int T = static_cast<int>(returns.rows());
    if (T < 2)
        throw std::invalid_argument("sampleCovariance: need at least 2 periods");
    rejectNonFinite(returns, "sampleCovariance");

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
    rejectNonFinite(returns, "ledoitWolfShrinkage");

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

    // π̂ — Ledoit-Wolf 2004 Lemma 4 estimator of asymptotic variance of S.
    //   π̂_{ij} = (1/T) Σ_t (x_{ti} x_{tj} − S_{ij})²
    //   π̂     = Σ_{ij} π̂_{ij}
    const Matrix X2 = X.array().square();
    const Matrix pi_mat =
        (X2.transpose() * X2) / static_cast<double>(T)
      - 2.0 * S.cwiseProduct((X.transpose() * X) / static_cast<double>(T))
      + S.cwiseProduct(S);
    const double pi_hat = pi_mat.sum();

    // ρ̂ — closed-form estimator from Ledoit-Wolf 2004 (matches the
    // reference MATLAB code at https://www.ledoit.net/honey.pdf, which
    // scikit-learn's `LedoitWolf` mirrors). Earlier versions of this
    // file used a self-described "simplified" approximation that
    // mis-weighted the off-diagonal cross-terms and biased δ̂.
    //
    // The off-diagonal asymptotic covariance involves
    //   θ̂_{ii,ij} = (1/T) Σ_t x_{ti}³ x_{tj} − S_{ii} · S_{ij}
    // and the constant-correlation contribution is
    //   r̄/2 · ( sqrt(σ_{jj}/σ_{ii}) · θ̂_{ii,ij}
    //         + sqrt(σ_{ii}/σ_{jj}) · θ̂_{jj,ij} )
    // which by i↔j symmetry collapses to r̄ · sqrt(σ_{jj}/σ_{ii}) · θ̂_{ii,ij}
    // summed over i≠j. Diagonal (i=j) contributions are π̂_{ii}.
    const Matrix X3 = X.array().cube();
    const Matrix term1 = (X3.transpose() * X) / static_cast<double>(T);
    Matrix term2(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            term2(i, j) = S(i, i) * S(i, j);

    double rho_hat = pi_mat.diagonal().sum();
    for (int i = 0; i < n; ++i) {
        if (stds[i] < 1e-15) continue;
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            if (stds[j] < 1e-15) continue;
            const double w_ij = stds[j] / stds[i];   // sqrt(σ_jj/σ_ii)
            rho_hat += avg_corr * w_ij * (term1(i, j) - term2(i, j));
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
    rejectNonFinite(returns, "oasShrinkage");

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
