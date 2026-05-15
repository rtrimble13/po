#include <portopt/io/writer.hpp>
#include <portopt/logging.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace portopt {
namespace io {

using json = nlohmann::json;

namespace {

struct BoxChars {
    std::string double_h;  // ═ or =
    std::string single_h;  // ─ or -
};

BoxChars boxChars(bool ascii_only) {
    if (ascii_only) return {"=", "-"};
    return {"═", "─"};
}

std::string repeat(const std::string& s, int n) {
    std::string out;
    out.reserve(s.size() * static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) out += s;
    return out;
}

json maybeNaN(double v) {
    if (std::isnan(v) || std::isinf(v)) return nullptr;
    return v;
}

} // namespace

// ── Format inference for output ───────────────────────────────────────────────

static OutputFormat inferOutputFormat(const std::filesystem::path& path) {
    auto ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (ext == ".json") return OutputFormat::JSON;
    if (ext == ".csv")  return OutputFormat::CSV;
    return OutputFormat::Console;
}

// ── JSON serialisation helpers ────────────────────────────────────────────────

static json metricsToJSON(const PortfolioMetrics& m) {
    json j = {
        {"expected_return",       m.expected_return},
        {"volatility",            m.volatility},
        {"sharpe_ratio",          maybeNaN(m.sharpe_ratio)},
        {"variance",              m.variance},
        {"diversification_ratio", maybeNaN(m.diversification_ratio)},
        {"effective_n_assets",    maybeNaN(m.effective_n_assets)},
        {"tracking_error",        maybeNaN(m.tracking_error)},
        {"information_ratio",     maybeNaN(m.information_ratio)},
        {"active_share",          maybeNaN(m.active_share)},
        {"beta_to_benchmark",     maybeNaN(m.beta_to_benchmark)},
        {"turnover",              maybeNaN(m.turnover)}
    };
    return j;
}

static json weightsToJSON(const Vector& w,
                          const std::vector<Asset>& assets,
                          const Vector& risk_contribution) {
    json arr = json::array();
    const bool have_rc = risk_contribution.size() == w.size();
    for (int i = 0; i < static_cast<int>(w.size()); ++i) {
        json entry = {
            {"ticker", assets[i].ticker},
            {"name",   assets[i].name},
            {"weight", w[i]}
        };
        if (!assets[i].sector.empty()) entry["sector"] = assets[i].sector;
        if (have_rc) entry["risk_contribution"] = risk_contribution[i];
        arr.push_back(std::move(entry));
    }
    return arr;
}

// ── OptimizationResult ────────────────────────────────────────────────────────

std::string resultToJSON(const OptimizationResult& r, int indent) {
    json j = {
        {"method",          r.method},
        {"converged",       r.converged},
        {"iterations",      r.iterations},
        {"solve_time_ms",   r.solve_time_ms},
        {"status",          r.status_message},
        {"metrics",         metricsToJSON(r.metrics)},
        {"weights",         weightsToJSON(r.weights, r.assets, r.metrics.risk_contribution)}
    };
    if (!r.active_lower_bounds.empty() || !r.active_upper_bounds.empty()) {
        j["active_constraints"] = {
            {"lower", r.active_lower_bounds},
            {"upper", r.active_upper_bounds}
        };
    }
    return j.dump(indent);
}

std::string resultToCSV(const OptimizationResult& r) {
    std::ostringstream ss;
    const bool have_rc = r.metrics.risk_contribution.size() == r.weights.size();
    ss << "ticker,name,sector,weight";
    if (have_rc) ss << ",risk_contribution";
    ss << "\n";
    for (int i = 0; i < static_cast<int>(r.weights.size()); ++i) {
        ss << r.assets[i].ticker << ","
           << r.assets[i].name   << ","
           << r.assets[i].sector << ","
           << std::fixed << std::setprecision(10) << r.weights[i];
        if (have_rc) ss << "," << r.metrics.risk_contribution[i];
        ss << "\n";
    }
    ss << "\nmetric,value\n";
    ss << "expected_return,"       << r.metrics.expected_return       << "\n";
    ss << "volatility,"            << r.metrics.volatility            << "\n";
    ss << "sharpe_ratio,"          << r.metrics.sharpe_ratio          << "\n";
    ss << "variance,"              << r.metrics.variance              << "\n";
    ss << "diversification_ratio," << r.metrics.diversification_ratio << "\n";
    ss << "effective_n_assets,"    << r.metrics.effective_n_assets    << "\n";
    ss << "tracking_error,"        << r.metrics.tracking_error        << "\n";
    ss << "information_ratio,"     << r.metrics.information_ratio     << "\n";
    ss << "active_share,"          << r.metrics.active_share          << "\n";
    ss << "beta_to_benchmark,"     << r.metrics.beta_to_benchmark     << "\n";
    ss << "turnover,"              << r.metrics.turnover              << "\n";
    ss << "method,"                << r.method                        << "\n";
    ss << "converged,"             << (r.converged ? 1 : 0)           << "\n";
    ss << "iterations,"            << r.iterations                    << "\n";
    ss << "solve_time_ms,"         << r.solve_time_ms                 << "\n";
    return ss.str();
}

static void writeConsoleResult(const OptimizationResult& r,
                                std::ostream& out,
                                const WriterConfig& cfg) {
    const auto box = boxChars(cfg.ascii_only);
    const std::string bar  = repeat(box.double_h, 60);
    const std::string bar2 = repeat(box.single_h, 60);

    out << "\n" << bar << "\n";
    out << "  Portfolio Optimisation Result\n" << bar << "\n";
    out << "  Method        : " << r.method << "\n";
    out << "  Converged     : " << (r.converged ? "yes" : "no")
        << "  (iters=" << r.iterations
        << ", time=" << std::fixed << std::setprecision(2) << r.solve_time_ms << " ms)\n";
    if (!r.status_message.empty())
        out << "  Status        : " << r.status_message << "\n";

    out << "\n  Portfolio Metrics\n  " << bar2 << "\n";
    out << "  Expected Return  : "
        << std::fixed << std::setprecision(cfg.console_return_prec)
        << r.metrics.expected_return * 100.0 << " %\n";
    out << "  Volatility       : "
        << r.metrics.volatility * 100.0 << " %\n";
    if (!std::isnan(r.metrics.sharpe_ratio))
        out << "  Sharpe Ratio     : "
            << std::setprecision(cfg.console_return_prec)
            << r.metrics.sharpe_ratio << "\n";
    out << "  Variance         : "
        << r.metrics.variance << "\n";
    if (!std::isnan(r.metrics.diversification_ratio))
        out << "  Diversification  : "
            << r.metrics.diversification_ratio << "\n";
    if (!std::isnan(r.metrics.effective_n_assets))
        out << "  Effective N      : "
            << r.metrics.effective_n_assets << "\n";
    if (!std::isnan(r.metrics.tracking_error))
        out << "  Tracking Error   : "
            << r.metrics.tracking_error * 100.0 << " %\n";
    if (!std::isnan(r.metrics.information_ratio))
        out << "  Information Rt.  : "
            << r.metrics.information_ratio << "\n";
    if (!std::isnan(r.metrics.active_share))
        out << "  Active Share     : "
            << r.metrics.active_share * 100.0 << " %\n";
    if (!std::isnan(r.metrics.beta_to_benchmark))
        out << "  Beta vs B/M      : "
            << r.metrics.beta_to_benchmark << "\n";
    if (!std::isnan(r.metrics.turnover))
        out << "  Turnover (1-way) : "
            << r.metrics.turnover * 100.0 << " %\n";

    out << "\n  Asset Weights\n  " << bar2 << "\n";
    out << "  " << std::left << std::setw(10) << "Ticker"
        << std::setw(26) << "Name"
        << std::right << std::setw(11) << "Weight"
        << std::setw(10) << "Wt (%)";
    const bool have_rc = cfg.show_risk_contribution &&
                         r.metrics.risk_contribution.size() == r.weights.size();
    if (have_rc) out << std::setw(11) << "RC (%vol)";
    if (cfg.total_capital > 0.0) out << std::setw(14) << "Notional $";
    out << "\n  " << std::string(75, '-') << "\n";

    const int n = static_cast<int>(r.weights.size());
    for (int i = 0; i < n; ++i) {
        if (!cfg.show_zero_weights && std::abs(r.weights[i]) < cfg.weight_threshold)
            continue;
        out << "  " << std::left << std::setw(10) << r.assets[i].ticker
            << std::setw(26) << r.assets[i].name
            << std::right << std::setw(11)
            << std::fixed << std::setprecision(cfg.console_weight_prec)
            << r.weights[i]
            << std::setw(10) << std::setprecision(2)
            << r.weights[i] * 100.0;
        if (have_rc)
            out << std::setw(11) << std::setprecision(3)
                << r.metrics.risk_contribution[i] * 100.0;
        if (cfg.total_capital > 0.0)
            out << std::setw(14) << std::setprecision(0)
                << r.weights[i] * cfg.total_capital;
        out << "\n";
    }

    if (cfg.explain) {
        out << "\n  Active Constraints\n  " << bar2 << "\n";
        if (r.active_lower_bounds.empty() && r.active_upper_bounds.empty()) {
            out << "  (none — solution is interior)\n";
        } else {
            if (!r.active_lower_bounds.empty()) {
                out << "  Lower bound active: ";
                for (int idx : r.active_lower_bounds)
                    out << r.assets[idx].ticker << " ";
                out << "\n";
            }
            if (!r.active_upper_bounds.empty()) {
                out << "  Upper bound active: ";
                for (int idx : r.active_upper_bounds)
                    out << r.assets[idx].ticker << " ";
                out << "\n";
            }
        }
    }

    out << bar << "\n\n";
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
            {"risk_aversion", p.risk_aversion},
            {"metrics",       metricsToJSON(p.metrics)},
            {"weights",       weightsToJSON(p.weights, frontier.assets,
                                            p.metrics.risk_contribution)}
        });
    }
    j["frontier"] = pts;
    return j.dump(indent);
}

std::string frontierToCSV(const EfficientFrontier& frontier) {
    std::ostringstream ss;
    ss << "risk_aversion,expected_return,volatility,sharpe_ratio,"
          "diversification_ratio,effective_n_assets";
    for (const auto& a : frontier.assets) ss << "," << a.ticker;
    ss << "\n";
    for (const auto& p : frontier.points) {
        ss << std::fixed << std::setprecision(8)
           << p.risk_aversion            << ","
           << p.metrics.expected_return  << ","
           << p.metrics.volatility       << ","
           << p.metrics.sharpe_ratio     << ","
           << p.metrics.diversification_ratio << ","
           << p.metrics.effective_n_assets;
        for (int i = 0; i < static_cast<int>(p.weights.size()); ++i)
            ss << "," << p.weights[i];
        ss << "\n";
    }
    return ss.str();
}

static void writeConsoleFrontier(const EfficientFrontier& frontier,
                                  std::ostream& out,
                                  const WriterConfig& cfg) {
    const auto box = boxChars(cfg.ascii_only);
    const std::string bar = repeat(box.double_h, 60);
    const std::string bar2 = repeat(box.single_h, 60);

    out << "\n" << bar << "\n";
    out << "  Efficient Frontier (" << frontier.method << ")\n" << bar << "\n";
    out << "  " << std::left
        << std::setw(12) << "Lambda"
        << std::setw(14) << "Return (%)"
        << std::setw(16) << "Volatility (%)"
        << std::setw(10) << "Sharpe"
        << std::setw(10) << "Div.Ratio"
        << "\n  " << bar2 << "\n";
    for (const auto& p : frontier.points) {
        out << "  " << std::fixed << std::setprecision(cfg.console_return_prec)
            << std::left
            << std::setw(12) << p.risk_aversion
            << std::setw(14) << p.metrics.expected_return * 100.0
            << std::setw(16) << p.metrics.volatility * 100.0
            << std::setw(10) << p.metrics.sharpe_ratio
            << std::setw(10) << p.metrics.diversification_ratio
            << "\n";
    }
    out << bar << "\n\n";
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

    auto toArr = [](const Vector& v) {
        json a = json::array();
        for (int i = 0; i < static_cast<int>(v.size()); ++i) a.push_back(v[i]);
        return a;
    };
    auto matToArr = [](const Matrix& M) {
        json a = json::array();
        for (int i = 0; i < static_cast<int>(M.rows()); ++i) {
            json row = json::array();
            for (int j = 0; j < static_cast<int>(M.cols()); ++j)
                row.push_back(M(i, j));
            a.push_back(std::move(row));
        }
        return a;
    };

    json j;
    j["prior_returns"]     = toArr(bl.prior_returns);
    j["posterior_returns"] = toArr(bl.posterior_returns);

    json assets_arr = json::array();
    for (int i = 0; i < n; ++i) {
        assets_arr.push_back({
            {"ticker",           assets[i].ticker},
            {"prior_return",     bl.prior_returns[i]},
            {"posterior_return", bl.posterior_returns[i]},
            {"delta",            bl.posterior_returns[i] - bl.prior_returns[i]}
        });
    }
    j["assets"] = assets_arr;

    json views_arr = json::array();
    for (int i = 0; i < k; ++i) {
        json v = {
            {"view_index",      i},
            {"expected_return", bl.view_returns[i]},
            {"omega",           bl.view_uncertainty(i, i)},
        };
        if (bl.view_confidence_pct.size() == k)
            v["confidence_pct"] = bl.view_confidence_pct[i];
        views_arr.push_back(std::move(v));
    }
    j["views"] = views_arr;
    j["pick_matrix"]      = matToArr(bl.pick_matrix);
    j["posterior_cov"]    = matToArr(bl.posterior_cov);
    j["blended_cov"]      = matToArr(bl.blended_cov);

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

    const auto box = boxChars(cfg.ascii_only);
    const std::string bar  = repeat(box.double_h, 60);
    const std::string bar2 = repeat(box.single_h, 60);
    const int n = static_cast<int>(bl.prior_returns.size());

    out << "\n" << bar << "\n";
    out << "  Black-Litterman Model Output\n" << bar << "\n";
    out << "  " << std::left
        << std::setw(10) << "Ticker"
        << std::setw(18) << "Prior (%)"
        << std::setw(20) << "Posterior (%)"
        << std::setw(14) << "Delta (%)"
        << "\n  " << bar2 << "\n";
    for (int i = 0; i < n; ++i) {
        const double pr = bl.prior_returns[i] * 100.0;
        const double po = bl.posterior_returns[i] * 100.0;
        out << "  " << std::left << std::setw(10) << assets[i].ticker
            << std::fixed << std::setprecision(4)
            << std::setw(18) << pr
            << std::setw(20) << po
            << std::setw(14) << (po - pr)
            << "\n";
    }

    if (bl.view_returns.size() > 0) {
        out << "\n  Views (" << bl.view_returns.size() << ")\n  " << bar2 << "\n";
        for (int i = 0; i < static_cast<int>(bl.view_returns.size()); ++i) {
            out << "  [" << i << "] q=" << bl.view_returns[i]
                << "  omega=" << bl.view_uncertainty(i, i);
            if (bl.view_confidence_pct.size() == bl.view_returns.size())
                out << "  conf%=" << bl.view_confidence_pct[i];
            out << "\n";
        }
    }
    out << bar << "\n\n";
}

} // namespace io
} // namespace portopt
