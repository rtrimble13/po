/**
 * @file bindings.cpp
 * @brief pybind11 bindings exposing the full portopt C++ library to Python.
 *
 * Build with CMake (PORTOPT_BUILD_PYTHON=ON), then import in Python:
 * @code{.python}
 *   import portopt
 *   data = portopt.read_market_data("assets.json")
 *   opt  = portopt.MVOptimizer(portopt.MVOParameters())
 *   res  = opt.optimize(data)
 *   print(res.metrics.sharpe_ratio)
 * @endcode
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
        const std::optional<Vector>&        market_weights = std::nullopt) {
    MarketData d;
    d.assets           = assets;
    d.expected_returns = expected_returns;
    d.covariance       = covariance;
    d.market_weights   = market_weights;
    return d;
}

// ── Module definition ─────────────────────────────────────────────────────────

PYBIND11_MODULE(_portopt, m) {
    m.doc() = "Portfolio optimisation library — MVO and Black-Litterman";

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
          py::arg("force")   = false,
          "Initialise the portopt logger.");

    m.def("set_log_level", &log::setLevel, py::arg("level"),
          "Set the portopt log level.");

    m.def("add_file_log", &log::addFileLog,
          py::arg("filename"),
          py::arg("level") = log::Level::Trace,
          "Add a rotating file sink to the portopt logger.");

    // ── Asset ─────────────────────────────────────────────────────────────────
    py::class_<Asset>(m, "Asset")
        .def(py::init<>())
        .def_readwrite("ticker",          &Asset::ticker)
        .def_readwrite("name",            &Asset::name)
        .def_readwrite("expected_return", &Asset::expected_return)
        .def_readwrite("market_cap",      &Asset::market_cap)
        .def("__repr__", [](const Asset& a) {
            return "<Asset ticker=" + a.ticker + " return=" +
                   std::to_string(a.expected_return) + ">";
        });

    // ── PortfolioConstraints ──────────────────────────────────────────────────
    py::class_<PortfolioConstraints>(m, "PortfolioConstraints")
        .def(py::init<>())
        .def_readwrite("lower_bounds",        &PortfolioConstraints::lower_bounds)
        .def_readwrite("upper_bounds",        &PortfolioConstraints::upper_bounds)
        .def_readwrite("allow_short_selling", &PortfolioConstraints::allow_short_selling)
        .def_static("long_only", &PortfolioConstraints::longOnly, py::arg("n"),
                    "Create long-only constraints for n assets.")
        .def_static("with_shorts", &PortfolioConstraints::withShorts,
                    py::arg("n"), py::arg("max_short") = 1.0,
                    "Create constraints allowing short positions up to max_short.");

    // ── MVOParameters ─────────────────────────────────────────────────────────
    py::class_<MVOParameters>(m, "MVOParameters")
        .def(py::init<>())
        .def_readwrite("risk_aversion",     &MVOParameters::risk_aversion)
        .def_readwrite("frontier_points",   &MVOParameters::frontier_points)
        .def_readwrite("min_risk_aversion", &MVOParameters::min_risk_aversion)
        .def_readwrite("max_risk_aversion", &MVOParameters::max_risk_aversion)
        .def_readwrite("constraints",       &MVOParameters::constraints);

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
        .def_readwrite("tau",            &BlackLittermanParameters::tau)
        .def_readwrite("risk_aversion",  &BlackLittermanParameters::risk_aversion)
        .def_readwrite("views",          &BlackLittermanParameters::views)
        .def_readwrite("mvo_params",     &BlackLittermanParameters::mvo_params);

    // ── PortfolioMetrics ──────────────────────────────────────────────────────
    py::class_<PortfolioMetrics>(m, "PortfolioMetrics")
        .def_readonly("expected_return", &PortfolioMetrics::expected_return)
        .def_readonly("volatility",      &PortfolioMetrics::volatility)
        .def_readonly("sharpe_ratio",    &PortfolioMetrics::sharpe_ratio)
        .def_readonly("variance",        &PortfolioMetrics::variance)
        .def("__repr__", [](const PortfolioMetrics& m_) {
            return "<PortfolioMetrics ret=" + std::to_string(m_.expected_return) +
                   " vol=" + std::to_string(m_.volatility) +
                   " sharpe=" + std::to_string(m_.sharpe_ratio) + ">";
        });

    // ── OptimizationResult ────────────────────────────────────────────────────
    py::class_<OptimizationResult>(m, "OptimizationResult")
        .def_readonly("weights",         &OptimizationResult::weights)
        .def_readonly("metrics",         &OptimizationResult::metrics)
        .def_readonly("assets",          &OptimizationResult::assets)
        .def_readonly("converged",       &OptimizationResult::converged)
        .def_readonly("iterations",      &OptimizationResult::iterations)
        .def_readonly("method",          &OptimizationResult::method)
        .def_readonly("status_message",  &OptimizationResult::status_message)
        .def("to_dict", [](const OptimizationResult& r) {
            py::dict d;
            d["method"]     = r.method;
            d["converged"]  = r.converged;
            d["iterations"] = r.iterations;
            d["weights"]    = r.weights;
            py::dict m;
            m["expected_return"] = r.metrics.expected_return;
            m["volatility"]      = r.metrics.volatility;
            m["sharpe_ratio"]    = r.metrics.sharpe_ratio;
            m["variance"]        = r.metrics.variance;
            d["metrics"] = m;
            return d;
        }, "Convert result to a plain Python dict.");

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
        .def("to_dataframe", [](const EfficientFrontier& ef) -> py::object {
            // Lazy import of pandas — avoids hard dependency
            py::object pd = py::module_::import("pandas");
            std::vector<double> lambdas, rets, vols, sharpes;
            for (const auto& p : ef.points) {
                lambdas.push_back(p.risk_aversion);
                rets.push_back(p.metrics.expected_return);
                vols.push_back(p.metrics.volatility);
                sharpes.push_back(p.metrics.sharpe_ratio);
            }
            py::dict data;
            data["risk_aversion"]    = py::cast(lambdas);
            data["expected_return"]  = py::cast(rets);
            data["volatility"]       = py::cast(vols);
            data["sharpe_ratio"]     = py::cast(sharpes);
            return pd.attr("DataFrame")(data);
        }, "Convert to a pandas DataFrame (requires pandas).");

    // ── MarketData ────────────────────────────────────────────────────────────
    py::class_<MarketData>(m, "MarketData")
        .def(py::init<>())
        .def_readwrite("assets",           &MarketData::assets)
        .def_readwrite("expected_returns", &MarketData::expected_returns)
        .def_readwrite("covariance",       &MarketData::covariance)
        .def_readwrite("market_weights",   &MarketData::market_weights)
        .def_static("from_arrays", &makeMarketData,
                    py::arg("assets"),
                    py::arg("expected_returns"),
                    py::arg("covariance"),
                    py::arg("market_weights") = std::optional<Vector>{},
                    "Construct MarketData from numpy arrays.");

    // ── BLModelOutput ─────────────────────────────────────────────────────────
    py::class_<BLModelOutput>(m, "BLModelOutput")
        .def_readonly("prior_returns",     &BLModelOutput::prior_returns)
        .def_readonly("posterior_returns", &BLModelOutput::posterior_returns)
        .def_readonly("posterior_cov",     &BLModelOutput::posterior_cov)
        .def_readonly("blended_cov",       &BLModelOutput::blended_cov)
        .def_readonly("pick_matrix",       &BLModelOutput::pick_matrix)
        .def_readonly("view_returns",      &BLModelOutput::view_returns)
        .def_readonly("view_uncertainty",  &BLModelOutput::view_uncertainty);

    // ── MVOptimizer ───────────────────────────────────────────────────────────
    py::class_<MVOptimizer>(m, "MVOptimizer")
        .def(py::init<MVOParameters>(),
             py::arg("params") = MVOParameters{},
             "Create an MVO optimizer with given parameters.")
        .def("optimize", &MVOptimizer::optimize, py::arg("data"),
             "Compute the optimal portfolio.")
        .def("efficient_frontier", &MVOptimizer::efficientFrontier, py::arg("data"),
             "Compute the efficient frontier.")
        .def("set_parameters", &MVOptimizer::setParameters, py::arg("params"))
        .def_property("parameters",
            [](const MVOptimizer& o) { return o.parameters(); },
            [](MVOptimizer& o, const MVOParameters& p) { o.setParameters(p); })
        .def_static("compute_metrics", &MVOptimizer::computeMetrics,
                    py::arg("weights"),
                    py::arg("mu"),
                    py::arg("sigma"),
                    "Compute portfolio metrics for given weights.");

    // ── BlackLittermanOptimizer ───────────────────────────────────────────────
    py::class_<BlackLittermanOptimizer>(m, "BlackLittermanOptimizer")
        .def(py::init<BlackLittermanParameters>(),
             py::arg("params") = BlackLittermanParameters{},
             "Create a Black-Litterman optimizer with given parameters.")
        .def("optimize", &BlackLittermanOptimizer::optimize, py::arg("data"),
             "Compute the BL-optimal portfolio.")
        .def("efficient_frontier", &BlackLittermanOptimizer::efficientFrontier,
             py::arg("data"), "Compute the BL efficient frontier.")
        .def("model_output", &BlackLittermanOptimizer::modelOutput, py::arg("data"),
             "Return BL model internals (prior/posterior returns, views).");

    // ── IO functions ──────────────────────────────────────────────────────────
    m.def("read_market_data",
          [](const std::string& path) {
              return io::readMarketData(std::filesystem::path(path));
          },
          py::arg("path"), "Read market data from a JSON or CSV file.");

    m.def("read_market_data_json",
          &io::readMarketDataFromJSON,
          py::arg("json_str"), "Parse market data from a JSON string.");

    m.def("read_mvo_parameters",
          [](const std::string& path) {
              return io::readMVOParameters(std::filesystem::path(path));
          },
          py::arg("path"), "Read MVO parameters from a JSON or TOML file.");

    m.def("read_bl_parameters",
          [](const std::string& path) {
              return io::readBLParameters(std::filesystem::path(path));
          },
          py::arg("path"), "Read BL parameters from a JSON or TOML file.");

    m.def("result_to_json",
          [](const OptimizationResult& r, int indent) {
              return io::resultToJSON(r, indent);
          },
          py::arg("result"), py::arg("indent") = 2,
          "Serialise OptimizationResult to a JSON string.");

    m.def("frontier_to_json",
          [](const EfficientFrontier& ef, int indent) {
              return io::frontierToJSON(ef, indent);
          },
          py::arg("frontier"), py::arg("indent") = 2,
          "Serialise EfficientFrontier to a JSON string.");

    m.def("bl_model_to_json",
          [](const BLModelOutput& bl, const std::vector<Asset>& assets, int indent) {
              return io::blModelToJSON(bl, assets, indent);
          },
          py::arg("bl_model"), py::arg("assets"), py::arg("indent") = 2,
          "Serialise BLModelOutput to a JSON string.");

    // ── Version ───────────────────────────────────────────────────────────────
    m.attr("__version__") = VERSION_STRING;
}
