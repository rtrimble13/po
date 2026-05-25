#include <portopt/portfolios.hpp>
#include <portopt/logging.hpp>
#include <portopt/mvo.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

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

// ── B4: Hierarchical Risk Parity (López de Prado 2016) ───────────────────────
//
// 1. Distance matrix from correlations:  d_ij = sqrt(0.5 (1 − ρ_ij))
// 2. Single-linkage agglomerative clustering — at each step merge the two
//    clusters whose minimum-distance pair is smallest.
// 3. Quasi-diagonalisation — read the linkage tree to produce a leaf order
//    where similar assets are adjacent.
// 4. Recursive bisection — split the ordered list in half; weight each
//    sub-cluster by inverse-variance (within cluster), then split each side.

static Matrix corrFromCov(const Matrix& cov) {
    const int n = static_cast<int>(cov.rows());
    Vector stds(n);
    for (int i = 0; i < n; ++i)
        stds[i] = std::sqrt(std::max(1e-30, cov(i, i)));
    Matrix C(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C(i, j) = cov(i, j) / (stds[i] * stds[j]);
    return C;
}

// Inverse-variance weight of a cluster (subset of indices).
static Vector clusterIVP(const Matrix& cov, const std::vector<int>& idx) {
    Vector w(static_cast<int>(idx.size()));
    double total = 0.0;
    for (std::size_t k = 0; k < idx.size(); ++k) {
        w[static_cast<int>(k)] = 1.0 / std::max(1e-30, cov(idx[k], idx[k]));
        total += w[static_cast<int>(k)];
    }
    return w / total;
}

// Variance of a cluster under inverse-variance internal weighting.
static double clusterVar(const Matrix& cov, const std::vector<int>& idx) {
    const Vector w_local = clusterIVP(cov, idx);
    const int k = static_cast<int>(idx.size());
    Matrix sub(k, k);
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j)
            sub(i, j) = cov(idx[i], idx[j]);
    return w_local.dot(sub * w_local);
}

Vector hierarchicalRiskParity(const Matrix& covariance) {
    const int n = static_cast<int>(covariance.rows());
    if (n <= 0 || covariance.cols() != n)
        throw std::invalid_argument(
            "hierarchicalRiskParity: covariance must be a non-empty square matrix");
    if (n == 1) return Vector::Ones(1);
    for (int i = 0; i < n; ++i)
        if (!(covariance(i, i) > 0.0))
            throw std::invalid_argument(
                "hierarchicalRiskParity: covariance(" + std::to_string(i) +
                "," + std::to_string(i) + ") must be > 0");

    // ── Step 1: distance matrix from correlation ────────────────────────────
    const Matrix C = corrFromCov(covariance);
    Matrix D(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            D(i, j) = std::sqrt(std::max(0.0, 0.5 * (1.0 - C(i, j))));

    // ── Steps 2 & 3: single-linkage clustering → leaf ordering ─────────────
    // We track active clusters as vectors of leaf indices. Each merge picks
    // the pair of active clusters with minimum inter-cluster distance.
    std::vector<std::vector<int>> clusters(n);
    for (int i = 0; i < n; ++i) clusters[i].push_back(i);

    auto pairwiseMin = [&](const std::vector<int>& a,
                            const std::vector<int>& b) {
        double dmin = std::numeric_limits<double>::infinity();
        for (int i : a)
            for (int j : b)
                dmin = std::min(dmin, D(i, j));
        return dmin;
    };

    while (clusters.size() > 1) {
        double best = std::numeric_limits<double>::infinity();
        std::size_t a_idx = 0, b_idx = 1;
        for (std::size_t a = 0; a < clusters.size(); ++a) {
            for (std::size_t b = a + 1; b < clusters.size(); ++b) {
                const double d = pairwiseMin(clusters[a], clusters[b]);
                if (d < best) { best = d; a_idx = a; b_idx = b; }
            }
        }
        // Merge b into a
        std::vector<int> merged = clusters[a_idx];
        merged.insert(merged.end(), clusters[b_idx].begin(),
                      clusters[b_idx].end());
        clusters.erase(clusters.begin() + b_idx);
        clusters[a_idx] = std::move(merged);
    }
    const std::vector<int> order = clusters[0];

    // ── Step 4: recursive bisection ─────────────────────────────────────────
    Vector w = Vector::Ones(n);
    struct Range { int lo, hi; };
    std::vector<Range> stack{ {0, n - 1} };
    while (!stack.empty()) {
        Range r = stack.back();
        stack.pop_back();
        if (r.hi <= r.lo) continue;
        const int mid = (r.lo + r.hi) / 2;
        std::vector<int> left(order.begin() + r.lo,
                               order.begin() + mid + 1);
        std::vector<int> right(order.begin() + mid + 1,
                                order.begin() + r.hi + 1);
        const double v_l = clusterVar(covariance, left);
        const double v_r = clusterVar(covariance, right);
        const double alpha = 1.0 - v_l / (v_l + v_r);  // weight to LEFT side
        for (int idx : left)  w[idx] *= alpha;
        for (int idx : right) w[idx] *= (1.0 - alpha);
        stack.push_back({r.lo, mid});
        stack.push_back({mid + 1, r.hi});
    }

    // w starts at all-ones and gets multiplied by α / (1-α) at each level;
    // sum is approximately 1 by construction (within numerical tolerance).
    const double s = w.sum();
    if (s > 0.0) w /= s;
    return w;
}

// ── B15 supplement: maximum diversification ──────────────────────────────────
//
// MDP maximises (Σ w_i σ_i)/σ_p subject to long-only and 1'w = 1. The
// optimality condition (Choueifaty & Coignard 2008): w ∝ Σ⁻¹·σ. We
// project onto the long-only simplex by setting negatives to zero and
// renormalising; for well-conditioned Σ this is exact.

Vector maximumDiversification(const Matrix& covariance) {
    const int n = static_cast<int>(covariance.rows());
    if (n <= 0 || covariance.cols() != n)
        throw std::invalid_argument(
            "maximumDiversification: covariance must be a non-empty square matrix");
    Vector sigma(n);
    for (int i = 0; i < n; ++i)
        sigma[i] = std::sqrt(std::max(1e-30, covariance(i, i)));
    Eigen::LDLT<Matrix> ldlt(0.5 * (covariance + covariance.transpose()));
    if (ldlt.info() != Eigen::Success) {
        log::warn("maximumDiversification: covariance LDLT failed; "
                  "returning inverse-volatility as fallback");
        return inverseVolatility(covariance);
    }
    Vector w = ldlt.solve(sigma);
    // Project to long-only and renormalise.
    for (int i = 0; i < n; ++i) w[i] = std::max(0.0, w[i]);
    const double s = w.sum();
    if (!(s > 0.0)) {
        log::warn("maximumDiversification: degenerate solution; "
                  "returning inverse-volatility as fallback");
        return inverseVolatility(covariance);
    }
    return w / s;
}

// ── B15 supplement: resampled MVO (Michaud 1998) ────────────────────────────

Vector resampledMVO(const Vector& mu,
                     const Matrix& sigma,
                     double risk_aversion,
                     int    n_samples,
                     int    n_resamples,
                     unsigned seed) {
    const int n = static_cast<int>(mu.size());
    if (sigma.rows() != n || sigma.cols() != n)
        throw std::invalid_argument(
            "resampledMVO: dimension mismatch between mu and sigma");
    if (n_samples < 2 || n_resamples < 1)
        throw std::invalid_argument(
            "resampledMVO: n_samples must be ≥ 2 and n_resamples ≥ 1");

    // Cholesky factor for sampling: synthetic returns r ~ N(mu, sigma)
    Eigen::LLT<Matrix> llt(0.5 * (sigma + sigma.transpose()));
    if (llt.info() != Eigen::Success) {
        log::warn("resampledMVO: sigma is not positive-definite; "
                  "falling back to inverse-volatility");
        return inverseVolatility(sigma);
    }
    const Matrix L = llt.matrixL();

    std::mt19937 rng(seed);
    std::normal_distribution<double> ndist(0.0, 1.0);

    Vector w_sum = Vector::Zero(n);
    int    success = 0;
    for (int b = 0; b < n_resamples; ++b) {
        // Generate n_samples × n synthetic returns; estimate μ̂, Σ̂.
        Matrix R(n_samples, n);
        for (int t = 0; t < n_samples; ++t) {
            Vector z(n);
            for (int j = 0; j < n; ++j) z[j] = ndist(rng);
            R.row(t) = (mu + L * z).transpose();
        }
        const Vector mu_b = R.colwise().mean().transpose();
        const Matrix X    = R.rowwise() - mu_b.transpose();
        const Matrix S_b  = (X.transpose() * X) /
                            static_cast<double>(n_samples - 1);

        // Run a single-call MVO on the resampled inputs.
        MarketData d_b;
        d_b.assets.resize(n);
        for (int i = 0; i < n; ++i) d_b.assets[i].ticker = std::to_string(i);
        d_b.expected_returns = mu_b;
        d_b.covariance       = S_b;

        MVOParameters p_b;
        p_b.risk_aversion = risk_aversion;
        p_b.constraints   = PortfolioConstraints::longOnly(n);
        MVOptimizer opt(p_b);
        try {
            auto r = opt.optimize(d_b);
            w_sum += r.weights;
            ++success;
        } catch (...) {
            // Skip pathological bootstrap draws.
        }
    }
    if (success == 0)
        throw std::runtime_error("resampledMVO: all bootstrap draws failed");
    Vector w = w_sum / static_cast<double>(success);
    const double s = w.sum();
    if (s > 0.0) w /= s;
    return w;
}

} // namespace portfolios
} // namespace portopt
