#include <portopt/factor_model.hpp>

#include <stdexcept>

namespace portopt {

void FactorRiskModel::validate() const {
    const int n = static_cast<int>(loadings.rows());
    const int k = static_cast<int>(loadings.cols());
    if (n == 0 || k == 0)
        throw std::invalid_argument(
            "FactorRiskModel: loadings must be non-empty");
    if (factor_covariance.rows() != k || factor_covariance.cols() != k)
        throw std::invalid_argument(
            "FactorRiskModel: factor_covariance must be " +
            std::to_string(k) + "x" + std::to_string(k));
    if (specific_variance.size() != n)
        throw std::invalid_argument(
            "FactorRiskModel: specific_variance must be length " +
            std::to_string(n));
    if (!factor_names.empty() &&
        static_cast<int>(factor_names.size()) != k)
        throw std::invalid_argument(
            "FactorRiskModel: factor_names must be empty or length " +
            std::to_string(k));
    for (int i = 0; i < n; ++i)
        if (specific_variance[i] < 0.0)
            throw std::invalid_argument(
                "FactorRiskModel: specific_variance has a negative entry");
    // Symmetry of factor_covariance — relative tolerance.
    const double sym_err =
        (factor_covariance - factor_covariance.transpose()).norm();
    if (sym_err > 1e-6 * std::max(1.0, factor_covariance.norm()))
        throw std::invalid_argument(
            "FactorRiskModel: factor_covariance is not symmetric");
}

Matrix FactorRiskModel::assetCovariance() const {
    validate();
    Matrix S = loadings * factor_covariance * loadings.transpose();
    for (int i = 0; i < S.rows(); ++i) S(i, i) += specific_variance[i];
    return 0.5 * (S + S.transpose());   // ensure exact symmetry
}

FactorRiskDecomposition decomposeRisk(const FactorRiskModel& model,
                                       const Vector& weights) {
    model.validate();
    const int n = model.numAssets();
    const int k = model.numFactors();
    if (weights.size() != n)
        throw std::invalid_argument(
            "decomposeRisk: weights size must equal model.numAssets()");

    FactorRiskDecomposition d;
    d.factor_exposures  = model.loadings.transpose() * weights;
    const Vector OmegaE = model.factor_covariance * d.factor_exposures;
    d.factor_variance   = d.factor_exposures.cwiseProduct(OmegaE);
    d.systematic_variance = d.factor_variance.sum();
    d.specific_variance   = 0.0;
    for (int i = 0; i < n; ++i)
        d.specific_variance += weights[i] * weights[i] *
                               model.specific_variance[i];
    d.total_variance = d.systematic_variance + d.specific_variance;
    (void)k;
    return d;
}

GroupConstraint factorNeutralConstraint(const FactorRiskModel& model,
                                         int factor_index,
                                         double lower,
                                         double upper,
                                         std::string description) {
    model.validate();
    if (factor_index < 0 || factor_index >= model.numFactors())
        throw std::invalid_argument(
            "factorNeutralConstraint: factor_index out of range");
    GroupConstraint g;
    if (description.empty()) {
        g.description = "factor " +
            (model.factor_names.empty()
                ? std::to_string(factor_index)
                : model.factor_names[static_cast<std::size_t>(factor_index)]);
    } else {
        g.description = std::move(description);
    }
    g.coefficients = model.loadings.col(factor_index);
    g.lower = lower;
    g.upper = upper;
    return g;
}

GroupConstraint betaNeutralConstraint(const FactorRiskModel& model,
                                       const Vector& benchmark_weights,
                                       double lower,
                                       double upper) {
    model.validate();
    const int n = model.numAssets();
    if (benchmark_weights.size() != n)
        throw std::invalid_argument(
            "betaNeutralConstraint: benchmark_weights size mismatch");
    const Matrix Sigma = model.assetCovariance();
    const double bm_var =
        benchmark_weights.dot(Sigma * benchmark_weights);
    if (!(bm_var > 0.0))
        throw std::invalid_argument(
            "betaNeutralConstraint: benchmark has zero variance");
    GroupConstraint g;
    g.description  = "beta-neutral";
    g.coefficients = (Sigma * benchmark_weights) / bm_var;
    g.lower = lower;
    g.upper = upper;
    return g;
}

} // namespace portopt
