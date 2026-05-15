/**
 * @file main.cpp
 * @brief portopt — Portfolio Optimisation CLI
 *
 * Usage examples:
 *
 * MVO — single optimal portfolio:
 *   portopt mvo -d assets.json -p params.toml -o result.json
 *
 * MVO — efficient frontier:
 *   portopt frontier -d assets.json -p params.json -o frontier.csv
 *
 * Black-Litterman:
 *   portopt bl -d assets.json -p bl_params.toml -o result.json
 *
 * BL efficient frontier:
 *   portopt bl-frontier -d assets.json -p bl_params.toml -o frontier.csv
 *
 * PM-friendly portfolios:
 *   portopt min-variance -d assets.json
 *   portopt max-sharpe   -d assets.json
 *   portopt target-vol   -d assets.json --target 0.15
 *   portopt target-return -d assets.json --target 0.10
 *
 * Returns-series input:
 *   portopt mvo -d daily_returns.csv --returns --shrinkage ledoit-wolf
 *
 * Generate Jupyter diagnostic report:
 *   portopt report -d assets.json -p params.toml -o reports/
 *
 * Run `portopt --help` for full option reference.
 */

#include <portopt/portopt.hpp>

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
using namespace portopt;

// ── Helper: resolve output format ─────────────────────────────────────────────

static io::WriterConfig makeWriterConfig(const std::string& fmt_str,
                                          const std::string& out_path,
                                          int indent,
                                          bool ascii_only,
                                          double total_capital,
                                          bool explain) {
    io::WriterConfig cfg;
    cfg.json_indent    = indent;
    cfg.ascii_only     = ascii_only;
    cfg.total_capital  = total_capital;
    cfg.explain        = explain;

    std::string f = fmt_str;
    for (auto& c : f) c = static_cast<char>(std::tolower(c));

    if (f == "json")        cfg.format = io::OutputFormat::JSON;
    else if (f == "csv")    cfg.format = io::OutputFormat::CSV;
    else if (f == "console" || f.empty()) cfg.format = io::OutputFormat::Console;
    else if (!f.empty())    std::cerr << "Warning: unknown format '" << fmt_str
                                       << "', defaulting to console/inferred\n";

    if (cfg.format == io::OutputFormat::Console && !out_path.empty()) {
        fs::path p(out_path);
        auto ext = p.extension().string();
        for (auto& c : ext) c = static_cast<char>(std::tolower(c));
        if (ext == ".json") cfg.format = io::OutputFormat::JSON;
        else if (ext == ".csv") cfg.format = io::OutputFormat::CSV;
    }

    return cfg;
}

// ── Helper: load market data, dispatching on --returns flag ──────────────────

static MarketData loadMarketData(const std::string& path,
                                  bool is_returns,
                                  double periods_per_year,
                                  const std::string& shrinkage,
                                  double shrinkage_delta) {
    if (is_returns)
        return io::readReturnsCSV(path, periods_per_year, shrinkage, shrinkage_delta);
    return io::readMarketData(path);
}

// ── Apply CLI overrides to MVOParameters ─────────────────────────────────────

static void applyOverrides(MVOParameters& params, int n,
                           double risk_aversion_override,
                           double rf_override,
                           double turnover_pen,
                           double budget_override) {
    if (params.constraints.lower_bounds.size() != n)
        params.constraints = PortfolioConstraints::longOnly(n);
    if (risk_aversion_override > 0.0)
        params.risk_aversion = risk_aversion_override;
    if (rf_override >= 0.0)
        params.risk_free_rate = rf_override;
    if (turnover_pen >= 0.0)
        params.constraints.turnover_penalty = turnover_pen;
    if (budget_override >= -1e30)  // sentinel; user explicitly set?
        params.constraints.budget = budget_override;
}

// ── Sub-command: mvo ──────────────────────────────────────────────────────────

static int runMVO(const std::string& data_path,
                  const std::string& params_path,
                  const std::string& output_path,
                  const std::string& fmt_str,
                  int json_indent,
                  bool show_zero,
                  bool ascii_only,
                  double total_capital,
                  bool explain,
                  bool returns_mode,
                  double periods_per_year,
                  const std::string& shrinkage,
                  double shrinkage_delta,
                  double risk_aversion_cli,
                  double rf_cli,
                  double turnover_pen_cli,
                  double budget_cli) {
    auto data = loadMarketData(data_path, returns_mode, periods_per_year,
                                shrinkage, shrinkage_delta);

    MVOParameters params;
    if (!params_path.empty())
        params = io::readMVOParameters(params_path);
    const int n = static_cast<int>(data.assets.size());
    applyOverrides(params, n, risk_aversion_cli, rf_cli, turnover_pen_cli, budget_cli);

    MVOptimizer opt(params);
    auto result = opt.optimize(data);

    auto cfg = makeWriterConfig(fmt_str, output_path, json_indent,
                                 ascii_only, total_capital, explain);
    cfg.show_zero_weights = show_zero;

    if (output_path.empty()) io::writeResult(result, std::cout, cfg);
    else {
        io::writeResult(result, fs::path(output_path), cfg);
        std::cout << "Result written to: " << output_path << "\n";
    }
    return result.converged ? 0 : 1;
}

// ── Sub-command: frontier ─────────────────────────────────────────────────────

static int runFrontier(const std::string& data_path,
                       const std::string& params_path,
                       const std::string& output_path,
                       const std::string& fmt_str,
                       int json_indent,
                       bool ascii_only,
                       bool returns_mode,
                       double periods_per_year,
                       const std::string& shrinkage,
                       double shrinkage_delta) {
    auto data = loadMarketData(data_path, returns_mode, periods_per_year,
                                shrinkage, shrinkage_delta);

    MVOParameters params;
    if (!params_path.empty())
        params = io::readMVOParameters(params_path);
    const int n = static_cast<int>(data.assets.size());
    applyOverrides(params, n, -1.0, -1.0, -1.0, -1e31);

    MVOptimizer opt(params);
    auto frontier = opt.efficientFrontier(data);

    auto cfg = makeWriterConfig(fmt_str, output_path, json_indent,
                                 ascii_only, 0.0, false);

    if (output_path.empty()) io::writeFrontier(frontier, std::cout, cfg);
    else {
        io::writeFrontier(frontier, fs::path(output_path), cfg);
        std::cout << "Frontier written to: " << output_path << "\n";
    }
    return 0;
}

// ── Sub-command: bl ───────────────────────────────────────────────────────────

static int runBL(const std::string& data_path,
                 const std::string& params_path,
                 const std::string& output_path,
                 const std::string& fmt_str,
                 int json_indent,
                 bool show_zero,
                 bool show_model,
                 bool ascii_only,
                 double total_capital,
                 bool explain,
                 bool returns_mode,
                 double periods_per_year,
                 const std::string& shrinkage,
                 double shrinkage_delta) {
    auto data = loadMarketData(data_path, returns_mode, periods_per_year,
                                shrinkage, shrinkage_delta);

    BlackLittermanParameters params;
    if (!params_path.empty())
        params = io::readBLParameters(params_path);

    const int n = static_cast<int>(data.assets.size());
    if (params.mvo_params.constraints.lower_bounds.size() != n)
        params.mvo_params.constraints = PortfolioConstraints::longOnly(n);

    BlackLittermanOptimizer bl(params);

    auto cfg = makeWriterConfig(fmt_str, output_path, json_indent,
                                 ascii_only, total_capital, explain);
    cfg.show_zero_weights = show_zero;

    if (show_model) {
        auto model = bl.modelOutput(data);
        io::writeBLModel(model, data.assets, std::cout, cfg);
    }

    auto result = bl.optimize(data);

    if (output_path.empty()) io::writeResult(result, std::cout, cfg);
    else {
        io::writeResult(result, fs::path(output_path), cfg);
        std::cout << "Result written to: " << output_path << "\n";
    }
    return result.converged ? 0 : 1;
}

// ── Sub-command: bl-frontier ──────────────────────────────────────────────────

static int runBLFrontier(const std::string& data_path,
                          const std::string& params_path,
                          const std::string& output_path,
                          const std::string& fmt_str,
                          int json_indent,
                          bool ascii_only,
                          bool returns_mode,
                          double periods_per_year,
                          const std::string& shrinkage,
                          double shrinkage_delta) {
    auto data = loadMarketData(data_path, returns_mode, periods_per_year,
                                shrinkage, shrinkage_delta);

    BlackLittermanParameters params;
    if (!params_path.empty())
        params = io::readBLParameters(params_path);

    const int n = static_cast<int>(data.assets.size());
    if (params.mvo_params.constraints.lower_bounds.size() != n)
        params.mvo_params.constraints = PortfolioConstraints::longOnly(n);

    BlackLittermanOptimizer bl(params);
    auto frontier = bl.efficientFrontier(data);

    auto cfg = makeWriterConfig(fmt_str, output_path, json_indent,
                                 ascii_only, 0.0, false);

    if (output_path.empty()) io::writeFrontier(frontier, std::cout, cfg);
    else {
        io::writeFrontier(frontier, fs::path(output_path), cfg);
        std::cout << "Frontier written to: " << output_path << "\n";
    }
    return 0;
}

// ── Sub-command: min-variance / max-sharpe / target-vol / target-return ──────

enum class PMTarget { MinVariance, MaxSharpe, TargetVol, TargetReturn };

static int runPMTarget(PMTarget which,
                        const std::string& data_path,
                        const std::string& params_path,
                        const std::string& output_path,
                        const std::string& fmt_str,
                        int json_indent,
                        bool show_zero,
                        bool ascii_only,
                        double total_capital,
                        bool explain,
                        bool returns_mode,
                        double periods_per_year,
                        const std::string& shrinkage,
                        double shrinkage_delta,
                        double rf_cli,
                        double target) {
    auto data = loadMarketData(data_path, returns_mode, periods_per_year,
                                shrinkage, shrinkage_delta);

    MVOParameters params;
    if (!params_path.empty()) params = io::readMVOParameters(params_path);
    const int n = static_cast<int>(data.assets.size());
    applyOverrides(params, n, -1.0, rf_cli, -1.0, -1e31);

    MVOptimizer opt(params);
    OptimizationResult result;
    switch (which) {
        case PMTarget::MinVariance:  result = opt.minVariancePortfolio(data); break;
        case PMTarget::MaxSharpe:    result = opt.maxSharpePortfolio(data);   break;
        case PMTarget::TargetVol:    result = opt.optimizeForTargetVolatility(data, target); break;
        case PMTarget::TargetReturn: result = opt.optimizeForTargetReturn(data, target); break;
    }

    auto cfg = makeWriterConfig(fmt_str, output_path, json_indent,
                                 ascii_only, total_capital, explain);
    cfg.show_zero_weights = show_zero;

    if (output_path.empty()) io::writeResult(result, std::cout, cfg);
    else {
        io::writeResult(result, fs::path(output_path), cfg);
        std::cout << "Result written to: " << output_path << "\n";
    }
    return result.converged ? 0 : 1;
}

// ── Sub-command: report ───────────────────────────────────────────────────────

static int runReport(const std::string& data_path,
                     const std::string& params_path,
                     const std::string& notebook_path,
                     const std::string& output_dir,
                     const std::string& method,
                     bool returns_mode,
                     double periods_per_year,
                     const std::string& shrinkage,
                     double shrinkage_delta) {
    fs::path out_dir = output_dir.empty() ? fs::current_path()
                                           : fs::path(output_dir);
    fs::create_directories(out_dir);

    auto data = loadMarketData(data_path, returns_mode, periods_per_year,
                                shrinkage, shrinkage_delta);

    // MVO
    MVOParameters mvo_params;
    if (!params_path.empty()) {
        try { mvo_params = io::readMVOParameters(params_path); }
        catch (...) {}
    }
    const int n = static_cast<int>(data.assets.size());
    if (mvo_params.constraints.lower_bounds.size() != n)
        mvo_params.constraints = PortfolioConstraints::longOnly(n);

    MVOptimizer mvo(mvo_params);
    auto mvo_frontier = mvo.efficientFrontier(data);
    auto mvo_result   = mvo.optimize(data);

    io::WriterConfig json_cfg;
    json_cfg.format = io::OutputFormat::JSON;
    // File names align with notebooks/generate_report.py — no leading underscore.
    io::writeFrontier(mvo_frontier, out_dir / "mvo_frontier.json", json_cfg);
    io::writeResult(mvo_result,     out_dir / "mvo_result.json",   json_cfg);

    nlohmann::json manifest;
    manifest["data_path"]   = fs::absolute(data_path).string();
    manifest["params_path"] = params_path.empty() ? std::string{}
                                                  : fs::absolute(params_path).string();
    manifest["method"]      = method;
    manifest["output_dir"]  = fs::absolute(out_dir).string();
    manifest["assets"]      = nlohmann::json::array();
    for (const auto& a : data.assets)
        manifest["assets"].push_back({{"ticker", a.ticker}, {"name", a.name}});
    manifest["mvo_result"]   = (out_dir / "mvo_result.json").string();
    manifest["mvo_frontier"] = (out_dir / "mvo_frontier.json").string();

    if ((method == "bl" || method == "both") && !params_path.empty()) {
        try {
            BlackLittermanParameters bl_params = io::readBLParameters(params_path);
            if (bl_params.mvo_params.constraints.lower_bounds.size() != n)
                bl_params.mvo_params.constraints = PortfolioConstraints::longOnly(n);

            BlackLittermanOptimizer bl(bl_params);
            auto bl_frontier = bl.efficientFrontier(data);
            auto bl_result   = bl.optimize(data);
            auto bl_model    = bl.modelOutput(data);

            io::writeFrontier(bl_frontier, out_dir / "bl_frontier.json", json_cfg);
            io::writeResult(bl_result,     out_dir / "bl_result.json",   json_cfg);
            {
                std::ofstream f(out_dir / "bl_model.json");
                f << io::blModelToJSON(bl_model, data.assets);
            }
            manifest["bl_result"]   = (out_dir / "bl_result.json").string();
            manifest["bl_frontier"] = (out_dir / "bl_frontier.json").string();
            manifest["bl_model"]    = (out_dir / "bl_model.json").string();
        } catch (const std::exception& ex) {
            std::cerr << "BL failed (skipping): " << ex.what() << "\n";
        }
    }

    {
        std::ofstream f(out_dir / "manifest.json");
        f << manifest.dump(2);
    }

    const std::string nb = notebook_path.empty()
                           ? "notebooks/diagnostic_template.ipynb"
                           : notebook_path;
    const std::string outfile = (out_dir / "diagnostic_report.ipynb").string();

    std::string cmd =
        "jupyter nbconvert --to notebook --execute "
        "--ExecutePreprocessor.timeout=600 "
        "--output \"" + outfile + "\" "
        "--ExecutePreprocessor.kernel_name=python3 "
        "\"" + nb + "\"";

    std::cout << "Running: " << cmd << "\n";
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::cerr << "nbconvert failed (exit " << rc << ").\n"
                  << "Ensure Jupyter is installed: pip install jupyter nbconvert\n";
        return rc;
    }

    std::string html_cmd =
        "jupyter nbconvert --to html \"" + outfile + "\"";
    (void)std::system(html_cmd.c_str());

    std::cout << "Report generated: " << outfile << "\n";
    return 0;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    CLI::App app{"portopt — Portfolio Optimisation Tool v" +
                 std::string(portopt::VERSION_STRING)};
    app.require_subcommand(1);

    // Global options
    std::string log_level_str = "info";
    std::string log_file;
    bool        ascii_only = false;
    app.add_option("--log-level", log_level_str,
                   "Log level: trace|debug|info|warn|error|off")
       ->default_val("info");
    app.add_option("--log-file", log_file, "Optional log file path");
    app.add_flag("--ascii", ascii_only,
                 "Use ASCII-only console output (no Unicode box characters)");

    // ── Shared option declarations ────────────────────────────────────────────
    auto makeDataOpt = [](CLI::App* sub, std::string& path) {
        sub->add_option("-d,--data", path,
                        "Market data file (.json or .csv) — or returns CSV with --returns")
           ->required();
    };
    auto makeParamsOpt = [](CLI::App* sub, std::string& path) {
        sub->add_option("-p,--params", path,
                        "Parameters file (.json or .toml)");
    };

    struct OutOpts {
        std::string out;
        std::string fmt;
        int         indent{2};
        bool        show_zero{false};
        bool        explain{false};
        double      total_capital{0.0};
    };
    auto makeOutputOpts = [](CLI::App* sub, OutOpts& o) {
        sub->add_option("-o,--output", o.out,
                        "Output file (default: stdout)");
        sub->add_option("-f,--format", o.fmt,
                        "Output format: console|json|csv (default: auto)");
        sub->add_option("--json-indent", o.indent,
                        "JSON indent spaces")->default_val(2);
        sub->add_flag("--show-zero", o.show_zero,
                       "Include near-zero weight assets in output");
        sub->add_flag("--explain", o.explain,
                       "Show active constraints and gradient diagnostics");
        sub->add_option("--total-capital", o.total_capital,
                        "If > 0, print notional dollar amounts per asset")
           ->default_val(0.0);
    };

    struct CommonOpts {
        std::string data, params;
        OutOpts out{};
        bool   returns_mode{false};
        double periods_per_year{252.0};
        std::string shrinkage{"none"};
        double shrinkage_delta{0.2};
        double risk_aversion{-1.0};
        double risk_free_rate{-1.0};
        double turnover_penalty{-1.0};
        double budget{-1e31};
    };

    auto makeReturnsOpts = [](CLI::App* sub, CommonOpts& c) {
        sub->add_flag("--returns", c.returns_mode,
                       "Interpret -d as a daily/monthly returns CSV and estimate μ, Σ");
        sub->add_option("--periods-per-year", c.periods_per_year,
                         "Annualisation factor for --returns (252=daily, 12=monthly)")
           ->default_val(252.0);
        sub->add_option("--shrinkage", c.shrinkage,
                         "Covariance shrinkage for --returns: none|linear|ledoit-wolf|oas")
           ->default_val("none");
        sub->add_option("--shrinkage-delta", c.shrinkage_delta,
                         "Manual δ for --shrinkage=linear")->default_val(0.2);
    };
    auto makeRiskOpts = [](CLI::App* sub, CommonOpts& c) {
        sub->add_option("--risk-aversion", c.risk_aversion,
                         "Override risk-aversion λ");
        sub->add_option("--risk-free-rate", c.risk_free_rate,
                         "Risk-free rate for Sharpe calculation");
        sub->add_option("--turnover-penalty", c.turnover_penalty,
                         "L2 turnover penalty κ (requires current_weights in params)");
        sub->add_option("--budget", c.budget,
                         "Override sum-of-weights budget (1.0=fully invested, 0.0=dollar-neutral)");
    };

    // ── mvo ───────────────────────────────────────────────────────────────────
    auto* mvo_cmd = app.add_subcommand("mvo", "Mean-Variance Optimisation");
    CommonOpts mvo_o; mvo_o.out.indent = 2;
    makeDataOpt(mvo_cmd, mvo_o.data); makeParamsOpt(mvo_cmd, mvo_o.params);
    makeOutputOpts(mvo_cmd, mvo_o.out); makeReturnsOpts(mvo_cmd, mvo_o);
    makeRiskOpts(mvo_cmd, mvo_o);

    // ── frontier ──────────────────────────────────────────────────────────────
    auto* fr_cmd = app.add_subcommand("frontier", "MVO efficient frontier");
    CommonOpts fr_o; fr_o.out.indent = 2;
    makeDataOpt(fr_cmd, fr_o.data); makeParamsOpt(fr_cmd, fr_o.params);
    makeOutputOpts(fr_cmd, fr_o.out); makeReturnsOpts(fr_cmd, fr_o);

    // ── bl ────────────────────────────────────────────────────────────────────
    auto* bl_cmd = app.add_subcommand("bl", "Black-Litterman Optimisation");
    CommonOpts bl_o; bl_o.out.indent = 2;
    bool bl_show_model = false;
    makeDataOpt(bl_cmd, bl_o.data); makeParamsOpt(bl_cmd, bl_o.params);
    makeOutputOpts(bl_cmd, bl_o.out); makeReturnsOpts(bl_cmd, bl_o);
    bl_cmd->add_flag("--show-model", bl_show_model,
                      "Print BL model diagnostics (prior vs posterior returns)");

    // ── bl-frontier ───────────────────────────────────────────────────────────
    auto* blf_cmd = app.add_subcommand("bl-frontier",
                                        "Black-Litterman efficient frontier");
    CommonOpts blf_o; blf_o.out.indent = 2;
    makeDataOpt(blf_cmd, blf_o.data); makeParamsOpt(blf_cmd, blf_o.params);
    makeOutputOpts(blf_cmd, blf_o.out); makeReturnsOpts(blf_cmd, blf_o);

    // ── min-variance ──────────────────────────────────────────────────────────
    auto* mv_cmd = app.add_subcommand("min-variance",
                                       "Minimum-variance portfolio");
    CommonOpts mv_o; mv_o.out.indent = 2;
    makeDataOpt(mv_cmd, mv_o.data); makeParamsOpt(mv_cmd, mv_o.params);
    makeOutputOpts(mv_cmd, mv_o.out); makeReturnsOpts(mv_cmd, mv_o);

    // ── max-sharpe ────────────────────────────────────────────────────────────
    auto* ms_cmd = app.add_subcommand("max-sharpe",
                                       "Maximum-Sharpe (tangency) portfolio");
    CommonOpts ms_o; ms_o.out.indent = 2;
    makeDataOpt(ms_cmd, ms_o.data); makeParamsOpt(ms_cmd, ms_o.params);
    makeOutputOpts(ms_cmd, ms_o.out); makeReturnsOpts(ms_cmd, ms_o);
    ms_cmd->add_option("--risk-free-rate", ms_o.risk_free_rate,
                        "Risk-free rate for Sharpe calculation");

    // ── target-vol ────────────────────────────────────────────────────────────
    auto* tv_cmd = app.add_subcommand("target-vol",
                                       "Portfolio with realised volatility ≈ target");
    CommonOpts tv_o; tv_o.out.indent = 2;
    double tv_target = 0.15;
    makeDataOpt(tv_cmd, tv_o.data); makeParamsOpt(tv_cmd, tv_o.params);
    makeOutputOpts(tv_cmd, tv_o.out); makeReturnsOpts(tv_cmd, tv_o);
    tv_cmd->add_option("--target", tv_target,
                        "Target volatility (annualised, e.g. 0.15 = 15%)")
        ->required();
    tv_cmd->add_option("--risk-free-rate", tv_o.risk_free_rate, "Risk-free rate");

    // ── target-return ─────────────────────────────────────────────────────────
    auto* tr_cmd = app.add_subcommand("target-return",
                                       "Portfolio with expected return ≈ target");
    CommonOpts tr_o; tr_o.out.indent = 2;
    double tr_target = 0.10;
    makeDataOpt(tr_cmd, tr_o.data); makeParamsOpt(tr_cmd, tr_o.params);
    makeOutputOpts(tr_cmd, tr_o.out); makeReturnsOpts(tr_cmd, tr_o);
    tr_cmd->add_option("--target", tr_target,
                        "Target expected return (annualised, e.g. 0.10 = 10%)")
        ->required();
    tr_cmd->add_option("--risk-free-rate", tr_o.risk_free_rate, "Risk-free rate");

    // ── report ────────────────────────────────────────────────────────────────
    auto* rep_cmd = app.add_subcommand("report",
        "Generate a Jupyter diagnostic report");
    std::string rep_data, rep_params, rep_nb, rep_out_dir, rep_method = "both";
    CommonOpts rep_o;
    rep_cmd->add_option("-d,--data",   rep_data,    "Market data or returns CSV")->required();
    rep_cmd->add_option("-p,--params", rep_params,  "Parameters file");
    rep_cmd->add_option("-n,--notebook", rep_nb,    "Jupyter notebook template");
    rep_cmd->add_option("-o,--output-dir", rep_out_dir, "Output directory");
    rep_cmd->add_option("-m,--method", rep_method,
                         "Method: mvo|bl|both")->default_val("both");
    makeReturnsOpts(rep_cmd, rep_o);

    // ── Parse ─────────────────────────────────────────────────────────────────
    CLI11_PARSE(app, argc, argv);

    // Initialise logging
    log::Level lvl = log::Level::Info;
    std::string ll = log_level_str;
    for (auto& c : ll) c = static_cast<char>(std::tolower(c));
    if      (ll == "trace")    lvl = log::Level::Trace;
    else if (ll == "debug")    lvl = log::Level::Debug;
    else if (ll == "warn")     lvl = log::Level::Warn;
    else if (ll == "error")    lvl = log::Level::Error;
    else if (ll == "off")      lvl = log::Level::Off;
    log::init(lvl, true);
    if (!log_file.empty()) log::addFileLog(log_file);

    try {
        if (*mvo_cmd)
            return runMVO(mvo_o.data, mvo_o.params, mvo_o.out.out, mvo_o.out.fmt,
                          mvo_o.out.indent, mvo_o.out.show_zero,
                          ascii_only, mvo_o.out.total_capital, mvo_o.out.explain,
                          mvo_o.returns_mode, mvo_o.periods_per_year,
                          mvo_o.shrinkage, mvo_o.shrinkage_delta,
                          mvo_o.risk_aversion, mvo_o.risk_free_rate,
                          mvo_o.turnover_penalty, mvo_o.budget);

        if (*fr_cmd)
            return runFrontier(fr_o.data, fr_o.params, fr_o.out.out, fr_o.out.fmt,
                                fr_o.out.indent, ascii_only,
                                fr_o.returns_mode, fr_o.periods_per_year,
                                fr_o.shrinkage, fr_o.shrinkage_delta);

        if (*bl_cmd)
            return runBL(bl_o.data, bl_o.params, bl_o.out.out, bl_o.out.fmt,
                         bl_o.out.indent, bl_o.out.show_zero, bl_show_model,
                         ascii_only, bl_o.out.total_capital, bl_o.out.explain,
                         bl_o.returns_mode, bl_o.periods_per_year,
                         bl_o.shrinkage, bl_o.shrinkage_delta);

        if (*blf_cmd)
            return runBLFrontier(blf_o.data, blf_o.params, blf_o.out.out, blf_o.out.fmt,
                                  blf_o.out.indent, ascii_only,
                                  blf_o.returns_mode, blf_o.periods_per_year,
                                  blf_o.shrinkage, blf_o.shrinkage_delta);

        if (*mv_cmd)
            return runPMTarget(PMTarget::MinVariance, mv_o.data, mv_o.params,
                                mv_o.out.out, mv_o.out.fmt, mv_o.out.indent,
                                mv_o.out.show_zero, ascii_only,
                                mv_o.out.total_capital, mv_o.out.explain,
                                mv_o.returns_mode, mv_o.periods_per_year,
                                mv_o.shrinkage, mv_o.shrinkage_delta,
                                -1.0, 0.0);

        if (*ms_cmd)
            return runPMTarget(PMTarget::MaxSharpe, ms_o.data, ms_o.params,
                                ms_o.out.out, ms_o.out.fmt, ms_o.out.indent,
                                ms_o.out.show_zero, ascii_only,
                                ms_o.out.total_capital, ms_o.out.explain,
                                ms_o.returns_mode, ms_o.periods_per_year,
                                ms_o.shrinkage, ms_o.shrinkage_delta,
                                ms_o.risk_free_rate, 0.0);

        if (*tv_cmd)
            return runPMTarget(PMTarget::TargetVol, tv_o.data, tv_o.params,
                                tv_o.out.out, tv_o.out.fmt, tv_o.out.indent,
                                tv_o.out.show_zero, ascii_only,
                                tv_o.out.total_capital, tv_o.out.explain,
                                tv_o.returns_mode, tv_o.periods_per_year,
                                tv_o.shrinkage, tv_o.shrinkage_delta,
                                tv_o.risk_free_rate, tv_target);

        if (*tr_cmd)
            return runPMTarget(PMTarget::TargetReturn, tr_o.data, tr_o.params,
                                tr_o.out.out, tr_o.out.fmt, tr_o.out.indent,
                                tr_o.out.show_zero, ascii_only,
                                tr_o.out.total_capital, tr_o.out.explain,
                                tr_o.returns_mode, tr_o.periods_per_year,
                                tr_o.shrinkage, tr_o.shrinkage_delta,
                                tr_o.risk_free_rate, tr_target);

        if (*rep_cmd)
            return runReport(rep_data, rep_params, rep_nb, rep_out_dir, rep_method,
                              rep_o.returns_mode, rep_o.periods_per_year,
                              rep_o.shrinkage, rep_o.shrinkage_delta);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
