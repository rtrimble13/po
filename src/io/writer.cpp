#include <portopt/io/writer.hpp>
#include <portopt/logging.hpp>

#include <nlohmann/json.hpp>

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace portopt {
namespace io {

using json = nlohmann::json;

// ── Format inference for output ───────────────────────────────────────────────

static OutputFormat inferOutputFormat(const std::filesystem::path& path) {
    auto ext = path.extension().string();
    for (auto& c : ext) c = static_cast<char>(std::tolower(c));
    if (ext == ".json") return OutputFormat::JSON;
    if (ext == ".csv")  return OutputFormat::CSV;
    return OutputFormat::Console;
}

// ── JSON serialisation helpers ────────────────────────────────────────────────

static json metricsToJSON(const PortfolioMetrics& m) {
    return {
        {"expected_return", m.expected_return},
        {"volatility",      m.volatility},
        {"sharpe_ratio",    m.sharpe_ratio},
        {"variance",        m.variance}
    };
}

static json weightsToJSON(const Vector& w, const std::vector<Asset>& assets) {
    json arr = json::array();
    for (int i = 0; i < static_cast<int>(w.size()); ++i) {
        arr.push_back({
            {"ticker", assets[i].ticker},
            {"name",   assets[i].name},
            {"weight", w[i]}
        });
    }
    return arr;
}

// ── OptimizationResult ────────────────────────────────────────────────────────

std::string resultToJSON(const OptimizationResult& r, int indent) {
    json j = {
        {"method",          r.method},
        {"converged",       r.converged},
        {"iterations",      r.iterations},
        {"status",          r.status_message},
        {"metrics",         metricsToJSON(r.metrics)},
        {"weights",         weightsToJSON(r.weights, r.assets)}
    };
    return j.dump(indent);
}

std::string resultToCSV(const OptimizationResult& r) {
    std::ostringstream ss;
    ss << "ticker,name,weight\n";
    for (int i = 0; i < static_cast<int>(r.weights.size()); ++i) {
        ss << r.assets[i].ticker << ","
           << r.assets[i].name   << ","
           << std::fixed << std::setprecision(8) << r.weights[i] << "\n";
    }
    ss << "\nmetric,value\n";
    ss << "expected_return," << r.metrics.expected_return << "\n";
    ss << "volatility,"      << r.metrics.volatility      << "\n";
    ss << "sharpe_ratio,"    << r.metrics.sharpe_ratio    << "\n";
    ss << "variance,"        << r.metrics.variance        << "\n";
    return ss.str();
}

static void writeConsoleResult(const OptimizationResult& r,
                                std::ostream& out,
                                const WriterConfig& cfg) {
    out << "\n══ Portfolio Optimisation Result ══════════════════════════\n";
    out << "  Method    : " << r.method << "\n";
    out << "  Converged : " << (r.converged ? "yes" : "no") << "\n";
    if (!r.status_message.empty())
        out << "  Status    : " << r.status_message << "\n";
    out << "\n  Portfolio Metrics\n";
    out << "  ─────────────────────────────────────────────────────────\n";
    out << "  Expected Return  : "
        << std::fixed << std::setprecision(cfg.console_return_prec)
        << r.metrics.expected_return * 100.0 << " %\n";
    out << "  Volatility       : "
        << r.metrics.volatility * 100.0 << " %\n";
    out << "  Sharpe Ratio     : "
        << std::setprecision(cfg.console_return_prec)
        << r.metrics.sharpe_ratio << "\n";
    out << "  Variance         : "
        << r.metrics.variance << "\n";
    out << "\n  Asset Weights\n";
    out << "  ─────────────────────────────────────────────────────────\n";
    out << "  " << std::left << std::setw(10) << "Ticker"
        << std::setw(30) << "Name"
        << std::right << std::setw(12) << "Weight" << "\n";
    out << "  " << std::string(52, '-') << "\n";

    const int n = static_cast<int>(r.weights.size());
    for (int i = 0; i < n; ++i) {
        if (!cfg.show_zero_weights && std::abs(r.weights[i]) < cfg.weight_threshold)
            continue;
        out << "  " << std::left << std::setw(10) << r.assets[i].ticker
            << std::setw(30) << r.assets[i].name
            << std::right << std::setw(12)
            << std::fixed << std::setprecision(cfg.console_weight_prec)
            << r.weights[i] << "\n";
    }
    out << "══════════════════════════════════════════════════════════\n\n";
}

void writeResult(const OptimizationResult& result,
                 std::ostream& out,
                 const WriterConfig& cfg) {
    switch (cfg.format) {
        case OutputFormat::Console: writeConsoleResult(result, out, cfg); break;
        case OutputFormat::JSON:    out << resultToJSON(result, cfg.json_indent); break;
        case OutputFormat::CSV:     out << resultToCSV(result); break;
    }
}

void writeResult(const OptimizationResult& result,
                 const std::filesystem::path& path,
                 const WriterConfig& cfg_in) {
    WriterConfig cfg = cfg_in;
    if (cfg.format == OutputFormat::Console)
        cfg.format = inferOutputFormat(path);

    std::ofstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot write to: " + path.string());

    log::info("Writing result to '{}'", path.string());
    writeResult(result, f, cfg);
}

// ── EfficientFrontier ─────────────────────────────────────────────────────────

std::string frontierToJSON(const EfficientFrontier& frontier, int indent) {
    json j;
    j["method"] = frontier.method;
    json pts = json::array();
    for (const auto& p : frontier.points) {
        pts.push_back({
            {"risk_aversion",   p.risk_aversion},
            {"metrics",         metricsToJSON(p.metrics)},
            {"weights",         weightsToJSON(p.weights, frontier.assets)}
        });
    }
    j["frontier"] = pts;
    return j.dump(indent);
}

std::string frontierToCSV(const EfficientFrontier& frontier) {
    std::ostringstream ss;
    // Header
    ss << "risk_aversion,expected_return,volatility,sharpe_ratio";
    for (const auto& a : frontier.assets)
        ss << "," << a.ticker;
    ss << "\n";
    // Rows
    for (const auto& p : frontier.points) {
        ss << std::fixed << std::setprecision(6)
           << p.risk_aversion            << ","
           << p.metrics.expected_return  << ","
           << p.metrics.volatility       << ","
           << p.metrics.sharpe_ratio;
        for (int i = 0; i < static_cast<int>(p.weights.size()); ++i)
            ss << "," << p.weights[i];
        ss << "\n";
    }
    return ss.str();
}

static void writeConsoleFrontier(const EfficientFrontier& frontier,
                                  std::ostream& out,
                                  const WriterConfig& cfg) {
    out << "\n══ Efficient Frontier (" << frontier.method << ") ═════════════════\n";
    out << std::left
        << std::setw(12) << "Lambda"
        << std::setw(14) << "Return (%)"
        << std::setw(14) << "Volatility (%)"
        << std::setw(12) << "Sharpe" << "\n";
    out << std::string(52, '─') << "\n";
    for (const auto& p : frontier.points) {
        out << std::fixed << std::setprecision(cfg.console_return_prec)
            << std::left
            << std::setw(12) << p.risk_aversion
            << std::setw(14) << p.metrics.expected_return * 100.0
            << std::setw(14) << p.metrics.volatility * 100.0
            << std::setw(12) << p.metrics.sharpe_ratio << "\n";
    }
    out << "══════════════════════════════════════════════════════════\n\n";
    (void)cfg;
}

void writeFrontier(const EfficientFrontier& frontier,
                   std::ostream& out,
                   const WriterConfig& cfg) {
    switch (cfg.format) {
        case OutputFormat::Console: writeConsoleFrontier(frontier, out, cfg); break;
        case OutputFormat::JSON:    out << frontierToJSON(frontier, cfg.json_indent); break;
        case OutputFormat::CSV:     out << frontierToCSV(frontier); break;
    }
}

void writeFrontier(const EfficientFrontier& frontier,
                   const std::filesystem::path& path,
                   const WriterConfig& cfg_in) {
    WriterConfig cfg = cfg_in;
    if (cfg.format == OutputFormat::Console)
        cfg.format = inferOutputFormat(path);

    std::ofstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot write to: " + path.string());

    log::info("Writing frontier to '{}'", path.string());
    writeFrontier(frontier, f, cfg);
}

// ── Black-Litterman diagnostics ───────────────────────────────────────────────

std::string blModelToJSON(const BLModelOutput& bl,
                           const AssetUniverse& assets,
                           int indent) {
    const int n = static_cast<int>(bl.prior_returns.size());
    const int k = static_cast<int>(bl.view_returns.size());

    json j;
    auto toArr = [](const Vector& v) {
        json a = json::array();
        for (int i = 0; i < static_cast<int>(v.size()); ++i)
            a.push_back(v[i]);
        return a;
    };

    j["prior_returns"]     = toArr(bl.prior_returns);
    j["posterior_returns"] = toArr(bl.posterior_returns);

    json assets_arr = json::array();
    for (int i = 0; i < n; ++i) {
        assets_arr.push_back({
            {"ticker",           assets[i].ticker},
            {"prior_return",     bl.prior_returns[i]},
            {"posterior_return", bl.posterior_returns[i]}
        });
    }
    j["assets"] = assets_arr;

    json views_arr = json::array();
    for (int i = 0; i < k; ++i) {
        views_arr.push_back({
            {"view_index",      i},
            {"expected_return", bl.view_returns[i]},
            {"uncertainty",     bl.view_uncertainty(i, i)}
        });
    }
    j["views"] = views_arr;

    return j.dump(indent);
}

void writeBLModel(const BLModelOutput& bl,
                  const AssetUniverse& assets,
                  std::ostream& out,
                  const WriterConfig& cfg) {
    if (cfg.format == OutputFormat::JSON) {
        out << blModelToJSON(bl, assets, cfg.json_indent);
        return;
    }

    const int n = static_cast<int>(bl.prior_returns.size());
    out << "\n══ Black-Litterman Model Output ════════════════════════════\n";
    out << std::left
        << std::setw(10) << "Ticker"
        << std::setw(18) << "Prior Return (%)"
        << std::setw(20) << "Posterior Return (%)"
        << "\n";
    out << std::string(48, '─') << "\n";
    for (int i = 0; i < n; ++i) {
        out << std::left << std::setw(10) << assets[i].ticker
            << std::fixed << std::setprecision(4)
            << std::setw(18) << bl.prior_returns[i] * 100.0
            << std::setw(20) << bl.posterior_returns[i] * 100.0
            << "\n";
    }

    if (bl.view_returns.size() > 0) {
        out << "\n  Views (" << bl.view_returns.size() << ")\n";
        for (int i = 0; i < static_cast<int>(bl.view_returns.size()); ++i) {
            out << "  [" << i << "] q=" << bl.view_returns[i]
                << "  omega=" << bl.view_uncertainty(i, i) << "\n";
        }
    }
    out << "══════════════════════════════════════════════════════════\n\n";
}

} // namespace io
} // namespace portopt
