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
 * Generate Jupyter diagnostic report:
 *   portopt report -d assets.json -p params.toml -n ./notebooks/diagnostic_template.ipynb
 *
 * Run `portopt --help` for full option reference.
 */

#include <portopt/portopt.hpp>

#include <CLI/CLI.hpp>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
using namespace portopt;

// ── Helper: resolve output format ─────────────────────────────────────────────

static io::WriterConfig makeWriterConfig(const std::string& fmt_str,
                                          const std::string& out_path,
                                          int indent) {
    io::WriterConfig cfg;
    cfg.json_indent = indent;

    std::string f = fmt_str;
    for (auto& c : f) c = static_cast<char>(std::tolower(c));

    if (f == "json")    cfg.format = io::OutputFormat::JSON;
    else if (f == "csv") cfg.format = io::OutputFormat::CSV;
    else if (f == "console" || f.empty()) cfg.format = io::OutputFormat::Console;
    else if (!out_path.empty()) {
        // infer from output file extension
        fs::path p(out_path);
        auto ext = p.extension().string();
        for (auto& c : ext) c = static_cast<char>(std::tolower(c));
        if (ext == ".json") cfg.format = io::OutputFormat::JSON;
        else if (ext == ".csv") cfg.format = io::OutputFormat::CSV;
    }

    return cfg;
}

// ── Sub-command: mvo ──────────────────────────────────────────────────────────

static int runMVO(const std::string& data_path,
                  const std::string& params_path,
                  const std::string& output_path,
                  const std::string& fmt_str,
                  int json_indent,
                  bool show_zero) {
    auto data = io::readMarketData(data_path);

    MVOParameters params;
    if (!params_path.empty())
        params = io::readMVOParameters(params_path);

    // Default constraints to long-only if not set
    int n = static_cast<int>(data.assets.size());
    if (params.constraints.lower_bounds.size() != static_cast<size_t>(n))
        params.constraints = PortfolioConstraints::longOnly(n);

    MVOptimizer opt(params);
    auto result = opt.optimize(data);

    auto cfg = makeWriterConfig(fmt_str, output_path, json_indent);
    cfg.show_zero_weights = show_zero;

    if (output_path.empty()) {
        io::writeResult(result, std::cout, cfg);
    } else {
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
                       int json_indent) {
    auto data = io::readMarketData(data_path);

    MVOParameters params;
    if (!params_path.empty())
        params = io::readMVOParameters(params_path);

    int n = static_cast<int>(data.assets.size());
    if (params.constraints.lower_bounds.size() != static_cast<size_t>(n))
        params.constraints = PortfolioConstraints::longOnly(n);

    MVOptimizer opt(params);
    auto frontier = opt.efficientFrontier(data);

    auto cfg = makeWriterConfig(fmt_str, output_path, json_indent);

    if (output_path.empty()) {
        io::writeFrontier(frontier, std::cout, cfg);
    } else {
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
                 bool show_model) {
    auto data = io::readMarketData(data_path);

    BlackLittermanParameters params;
    if (!params_path.empty())
        params = io::readBLParameters(params_path);

    int n = static_cast<int>(data.assets.size());
    if (params.mvo_params.constraints.lower_bounds.size() != static_cast<size_t>(n))
        params.mvo_params.constraints = PortfolioConstraints::longOnly(n);

    BlackLittermanOptimizer bl(params);

    if (show_model) {
        auto model = bl.modelOutput(data);
        io::WriterConfig mcfg;
        mcfg.format = (fmt_str == "json") ? io::OutputFormat::JSON
                                           : io::OutputFormat::Console;
        io::writeBLModel(model, data.assets, std::cout, mcfg);
    }

    auto result = bl.optimize(data);

    auto cfg = makeWriterConfig(fmt_str, output_path, json_indent);
    cfg.show_zero_weights = show_zero;

    if (output_path.empty()) {
        io::writeResult(result, std::cout, cfg);
    } else {
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
                          int json_indent) {
    auto data = io::readMarketData(data_path);

    BlackLittermanParameters params;
    if (!params_path.empty())
        params = io::readBLParameters(params_path);

    int n = static_cast<int>(data.assets.size());
    if (params.mvo_params.constraints.lower_bounds.size() != static_cast<size_t>(n))
        params.mvo_params.constraints = PortfolioConstraints::longOnly(n);

    BlackLittermanOptimizer bl(params);
    auto frontier = bl.efficientFrontier(data);

    auto cfg = makeWriterConfig(fmt_str, output_path, json_indent);

    if (output_path.empty()) {
        io::writeFrontier(frontier, std::cout, cfg);
    } else {
        io::writeFrontier(frontier, fs::path(output_path), cfg);
        std::cout << "Frontier written to: " << output_path << "\n";
    }
    return 0;
}

// ── Sub-command: report ───────────────────────────────────────────────────────

static int runReport(const std::string& data_path,
                     const std::string& params_path,
                     const std::string& notebook_path,
                     const std::string& output_dir,
                     const std::string& method) {
    // Compute results and write JSON intermediates, then invoke nbconvert
    fs::path out_dir = output_dir.empty() ? fs::current_path()
                                           : fs::path(output_dir);
    fs::create_directories(out_dir);

    // Write data JSON
    auto data = io::readMarketData(data_path);
    auto data_out = out_dir / "_report_data.json";
    {
        auto [first_asset, rest] = std::make_pair(data.assets[0], data.assets);
        (void)first_asset; (void)rest;
        // Re-read & re-write to get normalised JSON
        // (simplest: write directly)
    }

    // MVO frontier
    MVOParameters mvo_params;
    if (!params_path.empty()) {
        try { mvo_params = io::readMVOParameters(params_path); }
        catch (...) {}
    }
    int n = static_cast<int>(data.assets.size());
    if (mvo_params.constraints.lower_bounds.size() != static_cast<size_t>(n))
        mvo_params.constraints = PortfolioConstraints::longOnly(n);

    MVOptimizer mvo(mvo_params);
    auto mvo_frontier = mvo.efficientFrontier(data);
    auto mvo_result   = mvo.optimize(data);

    io::WriterConfig json_cfg;
    json_cfg.format = io::OutputFormat::JSON;
    io::writeFrontier(mvo_frontier, out_dir / "_mvo_frontier.json", json_cfg);
    io::writeResult(mvo_result,     out_dir / "_mvo_result.json",   json_cfg);

    // BL (if method == "bl" and params available)
    if ((method == "bl" || method == "both") && !params_path.empty()) {
        try {
            BlackLittermanParameters bl_params = io::readBLParameters(params_path);
            if (bl_params.mvo_params.constraints.lower_bounds.size() !=
                static_cast<size_t>(n))
                bl_params.mvo_params.constraints = PortfolioConstraints::longOnly(n);

            BlackLittermanOptimizer bl(bl_params);
            auto bl_frontier = bl.efficientFrontier(data);
            auto bl_result   = bl.optimize(data);
            auto bl_model    = bl.modelOutput(data);

            io::writeFrontier(bl_frontier, out_dir / "_bl_frontier.json", json_cfg);
            io::writeResult(bl_result,     out_dir / "_bl_result.json",   json_cfg);
            {
                std::ofstream f(out_dir / "_bl_model.json");
                f << io::blModelToJSON(bl_model, data.assets);
            }
        } catch (const std::exception& ex) {
            std::cerr << "BL failed (skipping): " << ex.what() << "\n";
        }
    }

    // Invoke Jupyter nbconvert
    std::string nb = notebook_path.empty()
                     ? "notebooks/diagnostic_template.ipynb"
                     : notebook_path;
    std::string outfile = (out_dir / "diagnostic_report.ipynb").string();

    std::string cmd =
        "jupyter nbconvert --to notebook --execute "
        "--ExecutePreprocessor.timeout=300 "
        "--output \"" + outfile + "\" "
        "--ExecutePreprocessor.kernel_name=python3 "
        "\"" + nb + "\" "
        "2>&1";

    std::cout << "Running: " << cmd << "\n";
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::cerr << "nbconvert failed (exit " << rc << ").\n"
                  << "Ensure Jupyter is installed: pip install jupyter nbconvert\n";
        return rc;
    }

    // Optionally convert to HTML
    std::string html_cmd =
        "jupyter nbconvert --to html "
        "\"" + outfile + "\" "
        "2>&1";
    std::system(html_cmd.c_str());

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
    app.add_option("--log-level", log_level_str,
                   "Log level: trace|debug|info|warn|error|off")
       ->default_val("info");
    app.add_option("--log-file", log_file,
                   "Optional log file path");

    // ── Shared option declarations ────────────────────────────────────────────
    auto makeDataOpt = [](CLI::App* sub, std::string& path) {
        sub->add_option("-d,--data", path,
                        "Market data file (.json or .csv)")->required();
    };
    auto makeParamsOpt = [](CLI::App* sub, std::string& path) {
        sub->add_option("-p,--params", path,
                        "Parameters file (.json or .toml)");
    };
    auto makeOutputOpts = [](CLI::App* sub, std::string& out,
                              std::string& fmt, int& indent) {
        sub->add_option("-o,--output", out,
                        "Output file (default: stdout)");
        sub->add_option("-f,--format", fmt,
                        "Output format: console|json|csv (default: auto)");
        sub->add_option("--json-indent", indent,
                        "JSON indent spaces")->default_val(2);
    };

    // ── mvo sub-command ───────────────────────────────────────────────────────
    auto* mvo_cmd = app.add_subcommand("mvo", "Mean-Variance Optimisation");
    std::string mvo_data, mvo_params, mvo_out, mvo_fmt;
    bool mvo_show_zero = false;
    int  mvo_indent = 2;
    makeDataOpt(mvo_cmd, mvo_data);
    makeParamsOpt(mvo_cmd, mvo_params);
    makeOutputOpts(mvo_cmd, mvo_out, mvo_fmt, mvo_indent);
    mvo_cmd->add_flag("--show-zero", mvo_show_zero,
                       "Include near-zero weight assets in output");

    // ── frontier sub-command ──────────────────────────────────────────────────
    auto* fr_cmd = app.add_subcommand("frontier",
                                       "MVO efficient frontier");
    std::string fr_data, fr_params, fr_out, fr_fmt;
    int fr_indent = 2;
    makeDataOpt(fr_cmd, fr_data);
    makeParamsOpt(fr_cmd, fr_params);
    makeOutputOpts(fr_cmd, fr_out, fr_fmt, fr_indent);

    // ── bl sub-command ────────────────────────────────────────────────────────
    auto* bl_cmd = app.add_subcommand("bl", "Black-Litterman Optimisation");
    std::string bl_data, bl_params, bl_out, bl_fmt;
    bool bl_show_zero = false, bl_show_model = false;
    int  bl_indent = 2;
    makeDataOpt(bl_cmd, bl_data);
    makeParamsOpt(bl_cmd, bl_params);
    makeOutputOpts(bl_cmd, bl_out, bl_fmt, bl_indent);
    bl_cmd->add_flag("--show-zero",  bl_show_zero,
                      "Include near-zero weight assets in output");
    bl_cmd->add_flag("--show-model", bl_show_model,
                      "Print BL model diagnostics (prior vs posterior returns)");

    // ── bl-frontier sub-command ───────────────────────────────────────────────
    auto* blf_cmd = app.add_subcommand("bl-frontier",
                                        "Black-Litterman efficient frontier");
    std::string blf_data, blf_params, blf_out, blf_fmt;
    int blf_indent = 2;
    makeDataOpt(blf_cmd, blf_data);
    makeParamsOpt(blf_cmd, blf_params);
    makeOutputOpts(blf_cmd, blf_out, blf_fmt, blf_indent);

    // ── report sub-command ────────────────────────────────────────────────────
    auto* rep_cmd = app.add_subcommand("report",
        "Generate a Jupyter diagnostic report");
    std::string rep_data, rep_params, rep_nb, rep_out_dir, rep_method = "both";
    rep_cmd->add_option("-d,--data",   rep_data,    "Market data file")->required();
    rep_cmd->add_option("-p,--params", rep_params,  "Parameters file");
    rep_cmd->add_option("-n,--notebook", rep_nb,    "Jupyter notebook template");
    rep_cmd->add_option("-o,--output-dir", rep_out_dir,
                         "Output directory for generated files");
    rep_cmd->add_option("-m,--method", rep_method,
                         "Method: mvo|bl|both")->default_val("both");

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
    if (!log_file.empty())
        log::addFileLog(log_file);

    // Dispatch
    try {
        if (*mvo_cmd)
            return runMVO(mvo_data, mvo_params, mvo_out, mvo_fmt,
                          mvo_indent, mvo_show_zero);

        if (*fr_cmd)
            return runFrontier(fr_data, fr_params, fr_out, fr_fmt, fr_indent);

        if (*bl_cmd)
            return runBL(bl_data, bl_params, bl_out, bl_fmt,
                         bl_indent, bl_show_zero, bl_show_model);

        if (*blf_cmd)
            return runBLFrontier(blf_data, blf_params, blf_out, blf_fmt, blf_indent);

        if (*rep_cmd)
            return runReport(rep_data, rep_params, rep_nb, rep_out_dir, rep_method);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
