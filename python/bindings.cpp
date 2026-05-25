/**
 * @file bindings.cpp
 * @brief pybind11 bindings exposing the full portopt C++ library to Python.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include <portopt/portopt.hpp>

namespace py = pybind11;
using namespace portopt;

// ── Helpers ───────────────────────────────────────────────────────────────────

static MarketData makeMarketData(
        const std::vector<Asset>&           assets,
        const Eigen::Ref<const Vector>&     expected_returns,
        const Eigen::Ref<const Matrix>&     covariance,
        const std::optional<Vector>&        market_weights      = std::nullopt,
        const std::optional<Vector>&        benchmark_weights   = std::nullopt,
        double                              risk_free_rate      = 0.0) {
    MarketData d;
    d.assets             = assets;
    d.expected_returns   = expected_returns;
    d.covariance         = covariance;
    d.market_weights     = market_weights;
    d.benchmark_weights  = benchmark_weights;
    d.risk_free_rate     = risk_free_rate;
    return d;
}

// ── Module definition ─────────────────────────────────────────────────────────

PYBIND11_MODULE(_portopt, m) {
    m.doc() = "Portfolio optimisation library — MVO and Black-Litterman";

    // ── Typed exception hierarchy (C4) ────────────────────────────────────────
    // Register each portopt error class with pybind11 so Python callers can
    // catch them as `portopt.PortoptError`, `portopt.InfeasibleProblem`,
    // etc. The `.code` attribute carries the stable machine-readable
    // reason code defined alongside each throw site.
    static py::exception<PortoptError>         ex_base(m, "PortoptError");
    static py::exception<InvalidMarketData>    ex_imd (m, "InvalidMarketData",       ex_base.ptr());
    static py::exception<InvalidParameters>    ex_ip  (m, "InvalidParameters",       ex_base.ptr());
    static py::exception<InfeasibleProblem>    ex_inf (m, "InfeasibleProblem",       ex_base.ptr());
    static py::exception<SolverDidNotConverge> ex_nc  (m, "SolverDidNotConverge",    ex_base.ptr());
    static py::exception<SolverCancelled>      ex_can (m, "SolverCancelled",         ex_base.ptr());
    static py::exception<SolverTimeout>        ex_to  (m, "SolverTimeout",           ex_base.ptr());
    // PyErr_SetString carries only a string; we prefix every portopt error
    // with `[code]` so Python wrappers can parse the machine-readable code
    // back out alongside the human message.
    py::register_exception_translator([](std::exception_ptr p) {
        auto with_code = [](const PortoptError& e) {
            return std::string("[") + e.code() + "] " + e.what();
        };
        if (!p) return;
        try { std::rethrow_exception(p); }
        catch (const SolverTimeout&        e) { PyErr_SetString(ex_to.ptr(),  with_code(e).c_str()); }
        catch (const SolverCancelled&      e) { PyErr_SetString(ex_can.ptr(), with_code(e).c_str()); }
        catch (const SolverDidNotConverge& e) { PyErr_SetString(ex_nc.ptr(),  with_code(e).c_str()); }
        catch (const InfeasibleProblem&    e) { PyErr_SetString(ex_inf.ptr(), with_code(e).c_str()); }
        catch (const InvalidParameters&    e) { PyErr_SetString(ex_ip.ptr(),  with_code(e).c_str()); }
        catch (const InvalidMarketData&    e) { PyErr_SetString(ex_imd.ptr(), with_code(e).c_str()); }
        catch (const PortoptError&         e) { PyErr_SetString(ex_base.ptr(),with_code(e).c_str()); }
    });

    // ── CancellationToken (C5) ────────────────────────────────────────────────
    py::class_<CancellationToken>(m, "CancellationToken")
        .def(py::init<>())
        .def("cancel", &CancellationToken::cancel)
        .def("reset",  &CancellationToken::reset)
        .def("is_cancellation_requested",
             &CancellationToken::isCancellationRequested);

    // ── Logging ───────────────────────────────────────────────────────────────
    py::enum_<log::Level>(m, "LogLevel")
        .value("Trace",    log::Level::Trace)
        .value("Debug",    log::Level::Debug)
        .value("Info",     log::Level::Info)
        .value("Warn",     log::Level::Warn)
        .value("Error",    log::Level::Error)
        .value("Critical", log::Level::Critical)
        .value("Off",      log::Level::Off);

    m.def("init_logging", &log::init,
          py::arg("level")   = log::Level::Info,
          py::arg("console") = true,
          py::arg("force")   = false);
    m.def("set_log_level", &log::setLevel, py::arg("level"));
    m.def("add_file_log", &log::addFileLog,
          py::arg("filename"),
          py::arg("level") = log::Level::Trace);

    // ── Asset ─────────────────────────────────────────────────────────────────
    py::class_<Asset>(m, "Asset")
        .def(py::init<>())
        .def_readwrite("ticker",          &Asset::ticker)
        .def_readwrite("name",            &Asset::name)
        .def_readwrite("expected_return", &Asset::expected_return)
        .def_readwrite("market_cap",      &Asset::market_cap)
        .def_readwrite("sector",          &Asset::sector)
        .def_readwrite("currency",        &Asset::currency)
        .def("__repr__", [](const Asset& a) {
            return "<Asset ticker=" + a.ticker + " return=" +
                   std::to_string(a.expected_return) + ">";
        });

    // ── GroupConstraint ───────────────────────────────────────────────────────
    py::class_<GroupConstraint>(m, "GroupConstraint")
        .def(py::init<>())
        .def_readwrite("description",  &GroupConstraint::description)
        .def_readwrite("coefficients", &GroupConstraint::coefficients)
        .def_readwrite("lower",        &GroupConstraint::lower)
        .def_readwrite("upper",        &GroupConstraint::upper);

    // ── PortfolioConstraints ──────────────────────────────────────────────────
    py::class_<PortfolioConstraints>(m, "PortfolioConstraints")
        .def(py::init<>())
        .def_readwrite("lower_bounds",        &PortfolioConstraints::lower_bounds)
        .def_readwrite("upper_bounds",        &PortfolioConstraints::upper_bounds)
        .def_readwrite("allow_short_selling", &PortfolioConstraints::allow_short_selling)
        .def_readwrite("budget",              &PortfolioConstraints::budget)
        .def_readwrite("current_weights",     &PortfolioConstraints::current_weights)
        .def_readwrite("turnover_penalty",    &PortfolioConstraints::turnover_penalty)
        .def_readwrite("groups",              &PortfolioConstraints::groups)
        .def_readwrite("tracking_error_limit",
                       &PortfolioConstraints::tracking_error_limit)
        .def_readwrite("gross_exposure_limit",
                       &PortfolioConstraints::gross_exposure_limit)
        .def_static("long_only", &PortfolioConstraints::longOnly, py::arg("n"))
        .def_static("with_shorts", &PortfolioConstraints::withShorts,
                    py::arg("n"), py::arg("max_short") = 1.0)
        .def_static("dollar_neutral", &PortfolioConstraints::dollarNeutral,
                    py::arg("n"), py::arg("max_gross") = 1.0)
        .def("fix_weight", &PortfolioConstraints::fixWeight,
             py::arg("index"), py::arg("weight"),
             "Pin an asset to a specific weight (lb = ub = weight).")
        .def("forbid", &PortfolioConstraints::forbid, py::arg("index"),
             "Zero-weight an asset.");

    // ── MVOParameters ─────────────────────────────────────────────────────────
    py::class_<MVOParameters>(m, "MVOParameters")
        .def(py::init<>())
        .def_readwrite("risk_aversion",            &MVOParameters::risk_aversion)
        .def_readwrite("frontier_points",          &MVOParameters::frontier_points)
        .def_readwrite("min_risk_aversion",        &MVOParameters::min_risk_aversion)
        .def_readwrite("max_risk_aversion",        &MVOParameters::max_risk_aversion)
        .def_readwrite("risk_free_rate",           &MVOParameters::risk_free_rate)
        .def_readwrite("group_penalty",            &MVOParameters::group_penalty)
        .def_readwrite("hard_group_constraints",   &MVOParameters::hard_group_constraints)
        .def_readwrite("group_tolerance",          &MVOParameters::group_tolerance)
        .def_readwrite("use_tangent_reformulation",&MVOParameters::use_tangent_reformulation)
        .def_readwrite("linear_transaction_cost",  &MVOParameters::linear_transaction_cost)
        .def_readwrite("quadratic_transaction_cost",&MVOParameters::quadratic_transaction_cost)
        .def_readwrite("cancellation",             &MVOParameters::cancellation)
        .def_readwrite("timeout_ms",               &MVOParameters::timeout_ms)
        .def_readwrite("constraints",              &MVOParameters::constraints);

    // ── ViewConfidenceMode ────────────────────────────────────────────────────
    py::enum_<ViewConfidenceMode>(m, "ViewConfidenceMode")
        .value("Variance", ViewConfidenceMode::Variance)
        .value("Idzorek",  ViewConfidenceMode::Idzorek);

    // ── View ──────────────────────────────────────────────────────────────────
    py::class_<View>(m, "View")
        .def(py::init<>())
        .def_readwrite("description",     &View::description)
        .def_readwrite("pick_vector",     &View::pick_vector)
        .def_readwrite("expected_return", &View::expected_return)
        .def_readwrite("confidence",      &View::confidence);

    // ── BlackLittermanParameters ──────────────────────────────────────────────
    py::class_<BlackLittermanParameters>(m, "BlackLittermanParameters")
        .def(py::init<>())
        .def_readwrite("tau",                     &BlackLittermanParameters::tau)
        .def_readwrite("risk_aversion",           &BlackLittermanParameters::risk_aversion)
        .def_readwrite("views",                   &BlackLittermanParameters::views)
        .def_readwrite("mvo_params",              &BlackLittermanParameters::mvo_params)
        .def_readwrite("confidence_mode",         &BlackLittermanParameters::confidence_mode)
        .def_readwrite("propagate_risk_aversion", &BlackLittermanParameters::propagate_risk_aversion);

    // ── PortfolioMetrics ──────────────────────────────────────────────────────
    py::class_<PortfolioMetrics>(m, "PortfolioMetrics")
        .def_readonly("expected_return",       &PortfolioMetrics::expected_return)
        .def_readonly("volatility",            &PortfolioMetrics::volatility)
        .def_readonly("sharpe_ratio",          &PortfolioMetrics::sharpe_ratio)
        .def_readonly("variance",              &PortfolioMetrics::variance)
        .def_readonly("risk_contribution",     &PortfolioMetrics::risk_contribution)
        .def_readonly("diversification_ratio", &PortfolioMetrics::diversification_ratio)
        .def_readonly("effective_n_assets",    &PortfolioMetrics::effective_n_assets)
        .def_readonly("tracking_error",        &PortfolioMetrics::tracking_error)
        .def_readonly("information_ratio",     &PortfolioMetrics::information_ratio)
        .def_readonly("active_share",          &PortfolioMetrics::active_share)
        .def_readonly("beta_to_benchmark",     &PortfolioMetrics::beta_to_benchmark)
        .def_readonly("turnover",              &PortfolioMetrics::turnover)
        .def("__repr__", [](const PortfolioMetrics& m_) {
            return "<PortfolioMetrics ret=" + std::to_string(m_.expected_return) +
                   " vol=" + std::to_string(m_.volatility) +
                   " sharpe=" + std::to_string(m_.sharpe_ratio) + ">";
        });

    // ── OptimizationResult ────────────────────────────────────────────────────
    py::class_<OptimizationResult>(m, "OptimizationResult")
        .def_readonly("weights",              &OptimizationResult::weights)
        .def_readonly("metrics",              &OptimizationResult::metrics)
        .def_readonly("assets",               &OptimizationResult::assets)
        .def_readonly("converged",            &OptimizationResult::converged)
        .def_readonly("iterations",           &OptimizationResult::iterations)
        .def_readonly("method",               &OptimizationResult::method)
        .def_readonly("status_message",       &OptimizationResult::status_message)
        .def_readonly("solve_time_ms",        &OptimizationResult::solve_time_ms)
        .def_readonly("gradient_at_optimum",  &OptimizationResult::gradient_at_optimum)
        .def_readonly("active_lower_bounds",  &OptimizationResult::active_lower_bounds)
        .def_readonly("active_upper_bounds",  &OptimizationResult::active_upper_bounds)
        .def_readonly("primal_residual",      &OptimizationResult::primal_residual)
        .def_readonly("kkt_residual",         &OptimizationResult::kkt_residual)
        .def_readonly("dual_estimate",        &OptimizationResult::dual_estimate)
        .def_readonly("library_version",      &OptimizationResult::library_version)
        .def_readonly("input_hash",           &OptimizationResult::input_hash)
        .def_readonly("params_hash",          &OptimizationResult::params_hash)
        .def("to_dict", [](const OptimizationResult& r) {
            py::dict d;
            d["method"]     = r.method;
            d["converged"]  = r.converged;
            d["iterations"] = r.iterations;
            d["solve_time_ms"] = r.solve_time_ms;
            d["weights"]    = r.weights;
            py::dict mtr;
            mtr["expected_return"]       = r.metrics.expected_return;
            mtr["volatility"]            = r.metrics.volatility;
            mtr["sharpe_ratio"]          = r.metrics.sharpe_ratio;
            mtr["variance"]              = r.metrics.variance;
            mtr["risk_contribution"]     = r.metrics.risk_contribution;
            mtr["diversification_ratio"] = r.metrics.diversification_ratio;
            mtr["effective_n_assets"]    = r.metrics.effective_n_assets;
            mtr["tracking_error"]        = r.metrics.tracking_error;
            mtr["information_ratio"]     = r.metrics.information_ratio;
            mtr["active_share"]          = r.metrics.active_share;
            mtr["beta_to_benchmark"]     = r.metrics.beta_to_benchmark;
            mtr["turnover"]              = r.metrics.turnover;
            d["metrics"] = mtr;
            py::dict conv;
            conv["primal_residual"] = r.primal_residual;
            conv["kkt_residual"]    = r.kkt_residual;
            conv["dual_estimate"]   = r.dual_estimate;
            d["convergence"] = conv;
            py::dict aud;
            aud["library_version"] = r.library_version;
            aud["input_hash"]      = r.input_hash;
            aud["params_hash"]     = r.params_hash;
            d["audit"] = aud;
            return d;
        });

    // ── EfficientFrontierPoint ────────────────────────────────────────────────
    py::class_<EfficientFrontierPoint>(m, "EfficientFrontierPoint")
        .def_readonly("risk_aversion", &EfficientFrontierPoint::risk_aversion)
        .def_readonly("weights",       &EfficientFrontierPoint::weights)
        .def_readonly("metrics",       &EfficientFrontierPoint::metrics);

    // ── EfficientFrontier ─────────────────────────────────────────────────────
    py::class_<EfficientFrontier>(m, "EfficientFrontier")
        .def_readonly("points", &EfficientFrontier::points)
        .def_readonly("assets", &EfficientFrontier::assets)
        .def_readonly("method", &EfficientFrontier::method)
        // C8 — pandas-free list-of-dicts. Each record contains the scalar
        // metrics plus a `weights` dict keyed by ticker. Use this from MCP
        // tools so the pandas import in to_dataframe() stays optional.
        .def("to_records", [](const EfficientFrontier& ef) {
            py::list records;
            for (const auto& p : ef.points) {
                py::dict rec;
                rec["risk_aversion"]         = p.risk_aversion;
                rec["expected_return"]       = p.metrics.expected_return;
                rec["volatility"]            = p.metrics.volatility;
                rec["sharpe_ratio"]          = p.metrics.sharpe_ratio;
                rec["diversification_ratio"] = p.metrics.diversification_ratio;
                rec["effective_n_assets"]    = p.metrics.effective_n_assets;
                py::dict wts;
                for (std::size_t i = 0; i < ef.assets.size(); ++i)
                    wts[py::str(ef.assets[i].ticker)] = p.weights[i];
                rec["weights"] = wts;
                records.append(rec);
            }
            return records;
        })
        .def("to_dataframe", [](const EfficientFrontier& ef) -> py::object {
            py::object pd = py::module_::import("pandas");
            std::vector<double> lambdas, rets, vols, sharpes, divr, eff_n;
            for (const auto& p : ef.points) {
                lambdas.push_back(p.risk_aversion);
                rets.push_back(p.metrics.expected_return);
                vols.push_back(p.metrics.volatility);
                sharpes.push_back(p.metrics.sharpe_ratio);
                divr.push_back(p.metrics.diversification_ratio);
                eff_n.push_back(p.metrics.effective_n_assets);
            }
            py::dict data;
            data["risk_aversion"]          = py::cast(lambdas);
            data["expected_return"]        = py::cast(rets);
            data["volatility"]             = py::cast(vols);
            data["sharpe_ratio"]           = py::cast(sharpes);
            data["diversification_ratio"]  = py::cast(divr);
            data["effective_n_assets"]     = py::cast(eff_n);
            py::object df = pd.attr("DataFrame")(data);

            // Append per-asset weight columns
            py::object pd_concat = pd.attr("concat");
            std::vector<std::vector<double>> wts(ef.assets.size());
            for (const auto& p : ef.points) {
                for (std::size_t i = 0; i < ef.assets.size(); ++i)
                    wts[i].push_back(p.weights[static_cast<int>(i)]);
            }
            py::dict wdict;
            for (std::size_t i = 0; i < ef.assets.size(); ++i)
                wdict[py::str(ef.assets[i].ticker)] = py::cast(wts[i]);
            py::object wdf = pd.attr("DataFrame")(wdict);
            return pd_concat(py::make_tuple(df, wdf), py::arg("axis") = 1);
        });

    // ── MarketData ────────────────────────────────────────────────────────────
    py::class_<MarketData>(m, "MarketData")
        .def(py::init<>())
        .def_readwrite("assets",             &MarketData::assets)
        .def_readwrite("expected_returns",   &MarketData::expected_returns)
        .def_readwrite("covariance",         &MarketData::covariance)
        .def_readwrite("market_weights",     &MarketData::market_weights)
        .def_readwrite("benchmark_weights",  &MarketData::benchmark_weights)
        .def_readwrite("risk_free_rate",     &MarketData::risk_free_rate)
        .def_static("from_arrays", &makeMarketData,
                    py::arg("assets"),
                    py::arg("expected_returns"),
                    py::arg("covariance"),
                    py::arg("market_weights")    = std::optional<Vector>{},
                    py::arg("benchmark_weights") = std::optional<Vector>{},
                    py::arg("risk_free_rate")    = 0.0);

    // ── BLModelOutput ─────────────────────────────────────────────────────────
    py::class_<BLModelOutput>(m, "BLModelOutput")
        .def_readonly("prior_returns",              &BLModelOutput::prior_returns)
        .def_readonly("posterior_returns",          &BLModelOutput::posterior_returns)
        .def_readonly("posterior_cov",              &BLModelOutput::posterior_cov)
        .def_readonly("blended_cov",                &BLModelOutput::blended_cov)
        .def_readonly("pick_matrix",                &BLModelOutput::pick_matrix)
        .def_readonly("view_returns",               &BLModelOutput::view_returns)
        .def_readonly("view_uncertainty",           &BLModelOutput::view_uncertainty)
        .def_readonly("view_confidence_pct",        &BLModelOutput::view_confidence_pct)
        .def_readonly("pick_matrix_rank",           &BLModelOutput::pick_matrix_rank)
        .def_readonly("pick_matrix_min_singular",   &BLModelOutput::pick_matrix_min_singular)
        .def_readonly("posterior_condition_number", &BLModelOutput::posterior_condition_number);

    // ── MVOptimizer ───────────────────────────────────────────────────────────
    py::class_<MVOptimizer>(m, "MVOptimizer",
        "Mean-Variance Optimiser (Markowitz, 1952).\n\n"
        "Solves  min w'Σw − λ μ'w  subject to budget, box, group, and\n"
        "optional turnover / transaction-cost / tracking-error /\n"
        "gross-exposure constraints.")
        .def(py::init<MVOParameters>(),
             py::arg("params") = MVOParameters{},
             "Construct an optimiser with the supplied parameters "
             "(default: long-only with λ=1).")
        .def("optimize", &MVOptimizer::optimize, py::arg("data"),
             "Compute the MVO-optimal portfolio. Returns an "
             "OptimizationResult with weights, metrics, and convergence "
             "diagnostics. Throws InvalidMarketData / InfeasibleProblem / "
             "SolverCancelled / SolverTimeout on error.")
        .def("efficient_frontier", &MVOptimizer::efficientFrontier,
             py::arg("data"),
             "Trace the efficient frontier by sweeping λ logarithmically "
             "from min_risk_aversion to max_risk_aversion across "
             "frontier_points steps. Returns an EfficientFrontier.")
        .def("min_variance_portfolio", &MVOptimizer::minVariancePortfolio,
             py::arg("data"),
             "Minimum-variance portfolio (λ → 0). Long-only by default.")
        .def("max_sharpe_portfolio", &MVOptimizer::maxSharpePortfolio,
             py::arg("data"),
             "Maximum-Sharpe (tangency) portfolio. Uses the analytical "
             "Σ⁻¹(μ-rf) fast path when constraints permit (A3); falls "
             "back to log-λ sweep + golden-section refinement otherwise.")
        .def("optimize_for_target_volatility",
             &MVOptimizer::optimizeForTargetVolatility,
             py::arg("data"), py::arg("target_volatility"),
             "Find the portfolio whose realised volatility is closest to "
             "target_volatility. Robust to non-monotonic λ→vol behaviour "
             "via frontier-bracket bisection (A4).")
        .def("optimize_for_target_return",
             &MVOptimizer::optimizeForTargetReturn,
             py::arg("data"), py::arg("target_return"),
             "Find the portfolio whose expected return is closest to "
             "target_return. Same frontier-bracket bisection as "
             "optimize_for_target_volatility.")
        .def("set_parameters", &MVOptimizer::setParameters, py::arg("params"))
        .def_property("parameters",
            [](const MVOptimizer& o) { return o.parameters(); },
            [](MVOptimizer& o, const MVOParameters& p) { o.setParameters(p); })
        .def_static("compute_metrics", &MVOptimizer::computeMetrics,
                    py::arg("weights"), py::arg("mu"), py::arg("sigma"),
                    py::arg("risk_free_rate") = 0.0)
        .def_static("augment_benchmark_metrics", &MVOptimizer::augmentBenchmarkMetrics,
                    py::arg("metrics"), py::arg("weights"), py::arg("mu"),
                    py::arg("sigma"), py::arg("benchmark_weights"),
                    py::arg("risk_free_rate") = 0.0)
        .def_static("validate_market_data", &MVOptimizer::validateMarketData,
                    py::arg("data"));

    // ── BlackLittermanOptimizer ───────────────────────────────────────────────
    py::class_<BlackLittermanOptimizer>(m, "BlackLittermanOptimizer",
        "Black-Litterman portfolio optimiser. Combines the market "
        "equilibrium prior π = δΣw_mkt with investor views P·μ ~ N(q,Ω) to "
        "produce posterior expected returns, then feeds them into MVO.")
        .def(py::init<BlackLittermanParameters>(),
             py::arg("params") = BlackLittermanParameters{},
             "Construct a BL optimiser with the supplied parameters.")
        .def("optimize", &BlackLittermanOptimizer::optimize, py::arg("data"),
             "Compute the BL-optimal portfolio. Requires "
             "MarketData.market_weights for the prior.")
        .def("efficient_frontier", &BlackLittermanOptimizer::efficientFrontier,
             py::arg("data"),
             "Trace the BL efficient frontier (posterior returns and "
             "blended covariance).")
        .def("model_output", &BlackLittermanOptimizer::modelOutput,
             py::arg("data"),
             "Return the BL diagnostic block — prior π, posterior μ_BL, "
             "Σ_BL, pick matrix P, Ω, view-confidence percentages, plus "
             "rank/condition-number checks (A7).");

    // ── Estimation ────────────────────────────────────────────────────────────
    py::module_ est = m.def_submodule("estimation",
        "Covariance and expected-return estimators from a returns time series");
    est.def("sample_mean", &estimation::sampleMean,
            py::arg("returns"), py::arg("periods_per_year") = 1.0);
    est.def("sample_covariance", &estimation::sampleCovariance,
            py::arg("returns"), py::arg("unbiased") = true,
            py::arg("periods_per_year") = 1.0);
    est.def("linear_shrinkage", &estimation::linearShrinkage,
            py::arg("sample_cov"), py::arg("delta"),
            py::arg("target") = std::optional<Matrix>{});
    est.def("ledoit_wolf_shrinkage",
        [](const Matrix& R, double ppy) {
            double delta = 0.0;
            Matrix S = estimation::ledoitWolfShrinkage(R, ppy, &delta);
            return py::make_tuple(S, delta);
        }, py::arg("returns"), py::arg("periods_per_year") = 1.0,
        "Returns (shrunk_cov, delta).");
    est.def("oas_shrinkage",
        [](const Matrix& R, double ppy) {
            double delta = 0.0;
            Matrix S = estimation::oasShrinkage(R, ppy, &delta);
            return py::make_tuple(S, delta);
        }, py::arg("returns"), py::arg("periods_per_year") = 1.0);
    est.def("ewma_covariance", &estimation::ewmaCovariance,
            py::arg("returns"), py::arg("lambda_") = 0.94,
            py::arg("periods_per_year") = 1.0,
            "Exponentially-weighted (RiskMetrics-style) covariance.");
    est.def("from_returns", &estimation::fromReturns,
            py::arg("tickers"), py::arg("returns"),
            py::arg("periods_per_year") = 1.0,
            py::arg("shrinkage") = std::string("none"),
            py::arg("shrinkage_delta") = 0.2);

    // ── IO functions ──────────────────────────────────────────────────────────
    m.def("read_market_data",
          [](const std::string& path) {
              return io::readMarketData(std::filesystem::path(path));
          }, py::arg("path"));
    m.def("read_market_data_json", &io::readMarketDataFromJSON, py::arg("json_str"));
    m.def("read_returns_csv",
          [](const std::string& path, double ppy,
             const std::string& shrink, double delta) {
              return io::readReturnsCSV(std::filesystem::path(path), ppy, shrink, delta);
          },
          py::arg("path"), py::arg("periods_per_year") = 252.0,
          py::arg("shrinkage") = std::string("none"),
          py::arg("shrinkage_delta") = 0.2);
    m.def("read_mvo_parameters",
          [](const std::string& path) {
              return io::readMVOParameters(std::filesystem::path(path));
          }, py::arg("path"));
    m.def("read_bl_parameters",
          [](const std::string& path) {
              return io::readBLParameters(std::filesystem::path(path));
          }, py::arg("path"));
    // C2 — in-memory JSON readers (no temp files)
    m.def("read_mvo_parameters_json", &io::readMVOParametersFromJSON,
          py::arg("json_str"),
          "Parse MVO parameters from a JSON string (no temp files).");
    m.def("read_bl_parameters_json", &io::readBLParametersFromJSON,
          py::arg("json_str"),
          "Parse Black-Litterman parameters from a JSON string (no temp files).");
    m.def("result_to_json",
          [](const OptimizationResult& r, int indent) {
              return io::resultToJSON(r, indent);
          }, py::arg("result"), py::arg("indent") = 2);
    m.def("result_to_csv",
          [](const OptimizationResult& r) { return io::resultToCSV(r); },
          py::arg("result"));
    m.def("frontier_to_json",
          [](const EfficientFrontier& ef, int indent) {
              return io::frontierToJSON(ef, indent);
          }, py::arg("frontier"), py::arg("indent") = 2);
    m.def("frontier_to_csv",
          [](const EfficientFrontier& ef) { return io::frontierToCSV(ef); },
          py::arg("frontier"));
    m.def("bl_model_to_json",
          [](const BLModelOutput& bl, const std::vector<Asset>& assets,
             int indent, bool summary) {
              return io::blModelToJSON(bl, assets, indent, summary);
          }, py::arg("bl_model"), py::arg("assets"),
             py::arg("indent") = 2, py::arg("summary") = false,
          "Serialise BL model to JSON. summary=True omits the n×n matrices.");

    // ── Convenience portfolios (B15) ──────────────────────────────────────────
    py::module_ pf = m.def_submodule("portfolios",
        "Closed-form convenience portfolio constructors.");
    pf.def("equal_weight", &portfolios::equalWeight, py::arg("n"));
    pf.def("inverse_variance", &portfolios::inverseVariance, py::arg("covariance"));
    pf.def("inverse_volatility", &portfolios::inverseVolatility,
           py::arg("covariance"));
    pf.def("market_cap_weighted", &portfolios::marketCapWeighted,
           py::arg("assets"));
    pf.def("equal_risk_contribution", &portfolios::equalRiskContribution,
           py::arg("covariance"),
           py::arg("tolerance") = 1e-8,
           py::arg("max_iters") = 5000,
           "Equal Risk Contribution (risk parity) portfolio.");
    pf.def("hierarchical_risk_parity", &portfolios::hierarchicalRiskParity,
           py::arg("covariance"),
           "Hierarchical Risk Parity (López de Prado, 2016). Long-only, "
           "budget=1, no return forecast required.");
    pf.def("maximum_diversification", &portfolios::maximumDiversification,
           py::arg("covariance"),
           "Maximum-diversification portfolio (Choueifaty & Coignard 2008).");
    pf.def("resampled_mvo", &portfolios::resampledMVO,
           py::arg("mu"), py::arg("sigma"),
           py::arg("risk_aversion") = 2.0,
           py::arg("n_samples")     = 252,
           py::arg("n_resamples")   = 100,
           py::arg("seed")          = 42u,
           "Resampled (Michaud) MVO via bootstrap averaging.");
    pf.def("minimum_cvar", &portfolios::minimumCVaR,
           py::arg("returns"), py::arg("alpha") = 0.95,
           py::arg("lower_bounds") = Vector(),
           py::arg("upper_bounds") = Vector(),
           py::arg("budget") = 1.0,
           py::arg("ridge")  = -1.0,
           py::arg("max_iters") = 50,
           "Minimum-CVaR portfolio (Rockafellar-Uryasev) over a sample "
           "return matrix.");
    pf.def("realised_cvar", &portfolios::realisedCVaR,
           py::arg("returns"), py::arg("weights"),
           py::arg("alpha") = 0.95,
           "Realised CVaR_α of a portfolio against a sample return history.");

    // ── Factor risk model (B7 / B8) ───────────────────────────────────────────
    py::class_<FactorRiskModel>(m, "FactorRiskModel")
        .def(py::init<>())
        .def_readwrite("loadings",          &FactorRiskModel::loadings)
        .def_readwrite("factor_covariance", &FactorRiskModel::factor_covariance)
        .def_readwrite("specific_variance", &FactorRiskModel::specific_variance)
        .def_readwrite("factor_names",      &FactorRiskModel::factor_names)
        .def("validate",          &FactorRiskModel::validate)
        .def("asset_covariance",  &FactorRiskModel::assetCovariance,
             "Reconstruct Σ = B·Ω_F·B' + diag(D).")
        .def_property_readonly("num_assets",  &FactorRiskModel::numAssets)
        .def_property_readonly("num_factors", &FactorRiskModel::numFactors);

    py::class_<FactorRiskDecomposition>(m, "FactorRiskDecomposition")
        .def_readonly("factor_exposures",   &FactorRiskDecomposition::factor_exposures)
        .def_readonly("factor_variance",    &FactorRiskDecomposition::factor_variance)
        .def_readonly("systematic_variance",&FactorRiskDecomposition::systematic_variance)
        .def_readonly("specific_variance",  &FactorRiskDecomposition::specific_variance)
        .def_readonly("total_variance",     &FactorRiskDecomposition::total_variance);

    m.def("decompose_risk", &decomposeRisk,
          py::arg("factor_model"), py::arg("weights"),
          "Decompose portfolio risk into per-factor systematic + specific.");
    m.def("factor_neutral_constraint", &factorNeutralConstraint,
          py::arg("factor_model"), py::arg("factor_index"),
          py::arg("lower")       = 0.0,
          py::arg("upper")       = 0.0,
          py::arg("description") = std::string(""),
          "Build a GroupConstraint pinning the portfolio's net loading on "
          "the named factor to [lower, upper].");
    m.def("beta_neutral_constraint", &betaNeutralConstraint,
          py::arg("factor_model"), py::arg("benchmark_weights"),
          py::arg("lower") = -0.05, py::arg("upper") = 0.05,
          "Build a GroupConstraint for β-neutrality against the benchmark.");

    // ── Multi-currency helpers (B13) ──────────────────────────────────────────
    py::module_ fx_mod = m.def_submodule("fx",
        "Multi-currency helpers: hedged returns and currency exposures.");
    py::class_<fx::CurrencyExposure>(m, "CurrencyExposure")
        .def_readonly("currency", &fx::CurrencyExposure::currency)
        .def_readonly("weight",   &fx::CurrencyExposure::weight);
    fx_mod.def("currency_exposure", &fx::currencyExposure,
               py::arg("assets"), py::arg("weights"),
               py::arg("base_currency") = std::string("BASE"));
    fx_mod.def("convert_expected_returns", &fx::convertExpectedReturns,
               py::arg("assets"), py::arg("mu_local"),
               py::arg("fx_returns"),
               py::arg("hedge_ratio") = 0.0);
    fx_mod.def("currency_exposure_constraint",
               &fx::currencyExposureConstraint,
               py::arg("assets"), py::arg("currency"),
               py::arg("lower"), py::arg("upper"));

    // ── Version ───────────────────────────────────────────────────────────────
    m.attr("__version__") = VERSION_STRING;
}
