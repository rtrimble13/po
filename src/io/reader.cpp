#include <portopt/io/reader.hpp>
#include <portopt/estimation.hpp>
#include <portopt/logging.hpp>

#include <nlohmann/json.hpp>
#include <toml++/toml.hpp>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace portopt {
namespace io {

using json = nlohmann::json;

// ── Helpers ───────────────────────────────────────────────────────────────────

Format inferFormat(const std::filesystem::path& path) {
    auto ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (ext == ".json") return Format::JSON;
    if (ext == ".csv")  return Format::CSV;
    if (ext == ".toml") return Format::TOML;
    throw std::invalid_argument("Cannot infer format from extension: " + ext);
}

static std::string readFile(const std::filesystem::path& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open file: " + path.string());
    return {std::istreambuf_iterator<char>(f), {}};
}

// CSV split with basic double-quote support. Quoted fields may contain commas.
static std::vector<std::string> splitCSVLine(const std::string& line) {
    std::vector<std::string> tokens;
    std::string cur;
    bool in_quote = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (in_quote) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    cur.push_back('"');  // escaped quote
                    ++i;
                } else {
                    in_quote = false;
                }
            } else {
                cur.push_back(c);
            }
        } else {
            if (c == '"') {
                in_quote = true;
            } else if (c == ',') {
                tokens.push_back(cur);
                cur.clear();
            } else {
                cur.push_back(c);
            }
        }
    }
    tokens.push_back(cur);

    // Trim whitespace + CR
    for (auto& t : tokens) {
        auto start = t.find_first_not_of(" \t\r\n");
        auto end   = t.find_last_not_of(" \t\r\n");
        if (start == std::string::npos) t.clear();
        else t = t.substr(start, end - start + 1);
    }
    return tokens;
}

static bool isParseableDouble(const std::string& s) {
    if (s.empty()) return false;
    try {
        std::size_t p;
        (void)std::stod(s, &p);
        return p == s.size();
    } catch (...) { return false; }
}

// ── JSON market data ──────────────────────────────────────────────────────────

MarketData readMarketDataFromJSON(const std::string& json_str) {
    json j = json::parse(json_str);
    MarketData data;

    if (!j.contains("assets"))
        throw std::runtime_error("JSON: missing 'assets' array");

    for (const auto& a : j["assets"]) {
        Asset asset;
        asset.ticker          = a.value("ticker", "");
        asset.name            = a.value("name", asset.ticker);
        asset.expected_return = a.value("expected_return", 0.0);
        asset.market_cap      = a.value("market_cap", 0.0);
        asset.sector          = a.value("sector", std::string{});
        data.assets.push_back(std::move(asset));
    }

    const int n = static_cast<int>(data.assets.size());
    if (n == 0)
        throw std::runtime_error("JSON: assets array is empty");

    if (j.contains("expected_returns")) {
        const auto& er = j["expected_returns"];
        if (static_cast<int>(er.size()) != n)
            throw std::runtime_error(
                "JSON: expected_returns length (" + std::to_string(er.size()) +
                ") != assets count (" + std::to_string(n) + ")");
        data.expected_returns = Vector(n);
        for (int i = 0; i < n; ++i)
            data.expected_returns[i] = er[i].get<double>();
    } else {
        data.expected_returns = Vector(n);
        for (int i = 0; i < n; ++i)
            data.expected_returns[i] = data.assets[i].expected_return;
    }

    if (!j.contains("covariance"))
        throw std::runtime_error("JSON: missing 'covariance' matrix");

    const auto& cov = j["covariance"];
    if (static_cast<int>(cov.size()) != n)
        throw std::runtime_error(
            "JSON: covariance rows (" + std::to_string(cov.size()) +
            ") != assets count (" + std::to_string(n) + ")");

    data.covariance = Matrix(n, n);
    for (int i = 0; i < n; ++i) {
        if (static_cast<int>(cov[i].size()) != n)
            throw std::runtime_error("JSON: covariance row " + std::to_string(i) +
                                     " has wrong length");
        for (int jj = 0; jj < n; ++jj)
            data.covariance(i, jj) = cov[i][jj].get<double>();
    }

    if (j.contains("market_weights")) {
        const auto& mw = j["market_weights"];
        if (static_cast<int>(mw.size()) != n)
            throw std::runtime_error("JSON: market_weights size mismatch");
        Vector v(n);
        for (int i = 0; i < n; ++i) v[i] = mw[i].get<double>();
        data.market_weights = v;
    }
    if (j.contains("benchmark_weights")) {
        const auto& bw = j["benchmark_weights"];
        if (static_cast<int>(bw.size()) != n)
            throw std::runtime_error("JSON: benchmark_weights size mismatch");
        Vector v(n);
        for (int i = 0; i < n; ++i) v[i] = bw[i].get<double>();
        data.benchmark_weights = v;
    }
    if (j.contains("risk_free_rate"))
        data.risk_free_rate = j["risk_free_rate"].get<double>();

    log::debug("JSON: loaded {} assets", n);
    return data;
}

// ── CSV market data ───────────────────────────────────────────────────────────

MarketData readMarketDataFromCSV(
        const std::filesystem::path& assets_csv,
        const std::filesystem::path& covariance_csv,
        const std::filesystem::path* weights_csv) {
    MarketData data;

    // ── Assets CSV ────────────────────────────────────────────────────────────
    std::vector<std::string> ticker_order;
    {
        std::ifstream f(assets_csv);
        if (!f.is_open())
            throw std::runtime_error("Cannot open: " + assets_csv.string());

        std::string header_line;
        std::getline(f, header_line);

        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            auto cols = splitCSVLine(line);
            if (cols.size() < 3)
                throw std::runtime_error("Assets CSV: expected at least 3 columns");
            Asset a;
            a.ticker          = cols[0];
            a.name            = cols.size() > 1 ? cols[1] : cols[0];
            a.expected_return = std::stod(cols[2]);
            a.market_cap      = cols.size() > 3 && !cols[3].empty()
                                ? std::stod(cols[3]) : 0.0;
            if (cols.size() > 4) a.sector = cols[4];
            data.assets.push_back(a);
            ticker_order.push_back(a.ticker);
        }
    }

    const int n = static_cast<int>(data.assets.size());
    if (n == 0)
        throw std::runtime_error("Assets CSV: no assets found");

    data.expected_returns = Vector(n);
    for (int i = 0; i < n; ++i)
        data.expected_returns[i] = data.assets[i].expected_return;

    // ── Covariance CSV ────────────────────────────────────────────────────────
    // Read all non-empty lines, then decide whether the first one is a header.
    {
        std::ifstream f(covariance_csv);
        if (!f.is_open())
            throw std::runtime_error("Cannot open: " + covariance_csv.string());

        std::vector<std::vector<std::string>> all_rows;
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            // Skip blank-only lines
            if (line.find_first_not_of(" \t\r\n,") == std::string::npos) continue;
            all_rows.push_back(splitCSVLine(line));
        }
        if (all_rows.empty())
            throw std::runtime_error("Covariance CSV: empty file");

        // Heuristic: the first row is a header if any of its cells (after
        // an optional first-column row-label) is non-parseable as double.
        bool first_is_header = false;
        const auto& first = all_rows.front();
        for (std::size_t i = 0; i < first.size(); ++i) {
            if (!first[i].empty() && !isParseableDouble(first[i])) {
                first_is_header = true; break;
            }
        }

        std::size_t data_offset = 0;
        if (first_is_header) {
            data_offset = 1;
            // Validate column header tickers if present (warn-only)
            std::unordered_set<std::string> assets_set(
                ticker_order.begin(), ticker_order.end());
            const int hdr_offset =
                (!first.empty() && !isParseableDouble(first[0])) ? 1 : 0;
            for (int i = 0; i < n && i + hdr_offset < static_cast<int>(first.size()); ++i) {
                const std::string& tk = first[i + hdr_offset];
                if (!tk.empty() && !assets_set.count(tk)) {
                    log::warn("Covariance CSV column '{}' is not in assets CSV", tk);
                }
                if (i < static_cast<int>(ticker_order.size()) &&
                    tk != ticker_order[i] && !tk.empty()) {
                    log::warn("Covariance CSV column {} = '{}' but assets[{}] = '{}'; "
                              "rows are read in assets-order — verify alignment",
                              i, tk, i, ticker_order[i]);
                }
            }
        }

        if (all_rows.size() - data_offset != static_cast<std::size_t>(n))
            throw std::runtime_error(
                "Covariance CSV: expected " + std::to_string(n) +
                " data rows, got " + std::to_string(all_rows.size() - data_offset));

        data.covariance = Matrix(n, n);
        for (int row = 0; row < n; ++row) {
            const auto& cols = all_rows[row + data_offset];
            // Strip leading ticker label if present
            int col_offset = 0;
            if (!cols.empty() && !isParseableDouble(cols[0])) col_offset = 1;
            if (static_cast<int>(cols.size()) - col_offset < n)
                throw std::runtime_error(
                    "Covariance CSV: row " + std::to_string(row) +
                    " has " + std::to_string(cols.size() - col_offset) +
                    " values, expected " + std::to_string(n));
            for (int c = 0; c < n; ++c)
                data.covariance(row, c) = std::stod(cols[col_offset + c]);
        }
    }

    // ── Optional weights CSV ──────────────────────────────────────────────────
    if (weights_csv) {
        std::ifstream f(*weights_csv);
        if (!f.is_open())
            throw std::runtime_error("Cannot open: " + weights_csv->string());
        std::string header_line;
        std::getline(f, header_line);
        Vector mw(n);
        int idx = 0;
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            if (idx >= n) throw std::runtime_error("Weights CSV: too many rows");
            auto cols = splitCSVLine(line);
            int col = cols.size() >= 2 ? 1 : 0;
            mw[idx++] = std::stod(cols[col]);
        }
        if (idx != n)
            throw std::runtime_error("Weights CSV: row count mismatch");
        data.market_weights = mw;
    }

    log::debug("CSV: loaded {} assets", n);
    return data;
}

// ── Returns CSV ──────────────────────────────────────────────────────────────

MarketData readReturnsCSV(const std::filesystem::path& returns_csv,
                          double periods_per_year,
                          const std::string& shrinkage,
                          double shrinkage_delta) {
    std::ifstream f(returns_csv);
    if (!f.is_open())
        throw std::runtime_error("Cannot open: " + returns_csv.string());

    std::string line;
    if (!std::getline(f, line))
        throw std::runtime_error("Returns CSV: empty file");
    auto header = splitCSVLine(line);
    if (header.size() < 2)
        throw std::runtime_error("Returns CSV: expected ≥ 2 columns (date + ≥ 1 asset)");

    // Drop first column (date)
    std::vector<std::string> tickers(header.begin() + 1, header.end());
    const int n = static_cast<int>(tickers.size());

    std::vector<std::vector<double>> rows;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto cols = splitCSVLine(line);
        if (static_cast<int>(cols.size()) < n + 1) continue;
        std::vector<double> r(n);
        for (int i = 0; i < n; ++i) r[i] = std::stod(cols[i + 1]);
        rows.push_back(std::move(r));
    }
    const int T = static_cast<int>(rows.size());
    if (T < 2)
        throw std::runtime_error("Returns CSV: need ≥ 2 periods (got " +
                                 std::to_string(T) + ")");

    Matrix R(T, n);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < n; ++i) R(t, i) = rows[t][i];

    log::info("Returns CSV: T={} periods, n={} assets, shrinkage={}",
              T, n, shrinkage);

    return estimation::fromReturns(tickers, R, periods_per_year,
                                   shrinkage, shrinkage_delta);
}

// ── File dispatch ─────────────────────────────────────────────────────────────

MarketData readMarketData(const std::filesystem::path& path, Format fmt) {
    if (fmt == Format::Auto) fmt = inferFormat(path);
    log::info("Reading market data from '{}'", path.string());

    if (fmt == Format::JSON)
        return readMarketDataFromJSON(readFile(path));

    auto dir = path.parent_path();
    auto stem = path.stem().string();
    auto cov_path = dir / "covariance.csv";
    if (!std::filesystem::exists(cov_path))
        cov_path = dir / (stem + "_covariance.csv");
    return readMarketDataFromCSV(path, cov_path, nullptr);
}

// ── Constraints parsing (shared between MVO and BL) ──────────────────────────

static void parseConstraintsJSON(const json& c, PortfolioConstraints& out) {
    if (c.contains("lower_bounds")) {
        const auto& lb = c["lower_bounds"];
        out.lower_bounds = Vector(static_cast<int>(lb.size()));
        for (int i = 0; i < static_cast<int>(lb.size()); ++i)
            out.lower_bounds[i] = lb[i].get<double>();
    }
    if (c.contains("upper_bounds")) {
        const auto& ub = c["upper_bounds"];
        out.upper_bounds = Vector(static_cast<int>(ub.size()));
        for (int i = 0; i < static_cast<int>(ub.size()); ++i)
            out.upper_bounds[i] = ub[i].get<double>();
    }
    out.allow_short_selling = c.value("allow_short_selling", out.allow_short_selling);
    out.budget              = c.value("budget", out.budget);
    out.turnover_penalty    = c.value("turnover_penalty", out.turnover_penalty);
    if (c.contains("current_weights")) {
        const auto& cw = c["current_weights"];
        out.current_weights = Vector(static_cast<int>(cw.size()));
        for (int i = 0; i < static_cast<int>(cw.size()); ++i)
            out.current_weights[i] = cw[i].get<double>();
    }
    if (c.contains("groups")) {
        out.groups.clear();
        for (const auto& g : c["groups"]) {
            GroupConstraint gc;
            gc.description = g.value("description", std::string{});
            if (!g.contains("coefficients"))
                throw std::runtime_error("Group constraint missing 'coefficients'");
            const auto& a = g["coefficients"];
            gc.coefficients = Vector(static_cast<int>(a.size()));
            for (int i = 0; i < static_cast<int>(a.size()); ++i)
                gc.coefficients[i] = a[i].get<double>();
            gc.lower = g.value("lower", -1e30);
            gc.upper = g.value("upper",  1e30);
            out.groups.push_back(std::move(gc));
        }
    }
}

static void parseConstraintsTOML(const toml::table& c, PortfolioConstraints& out) {
    if (auto* lb = c["lower_bounds"].as_array()) {
        out.lower_bounds = Vector(static_cast<int>(lb->size()));
        for (int i = 0; i < static_cast<int>(lb->size()); ++i)
            out.lower_bounds[i] = lb->get(i)->value_or(0.0);
    }
    if (auto* ub = c["upper_bounds"].as_array()) {
        out.upper_bounds = Vector(static_cast<int>(ub->size()));
        for (int i = 0; i < static_cast<int>(ub->size()); ++i)
            out.upper_bounds[i] = ub->get(i)->value_or(1.0);
    }
    if (auto v = c["allow_short_selling"].value<bool>())
        out.allow_short_selling = *v;
    if (auto v = c["budget"].value<double>()) out.budget = *v;
    if (auto v = c["turnover_penalty"].value<double>()) out.turnover_penalty = *v;
    if (auto* cw = c["current_weights"].as_array()) {
        out.current_weights = Vector(static_cast<int>(cw->size()));
        for (int i = 0; i < static_cast<int>(cw->size()); ++i)
            out.current_weights[i] = cw->get(i)->value_or(0.0);
    }
    if (auto* groups = c["groups"].as_array()) {
        out.groups.clear();
        for (const auto& ge : *groups) {
            const auto* gt = ge.as_table();
            if (!gt) continue;
            GroupConstraint gc;
            if (auto v = (*gt)["description"].value<std::string>()) gc.description = *v;
            if (auto v = (*gt)["lower"].value<double>()) gc.lower = *v;
            if (auto v = (*gt)["upper"].value<double>()) gc.upper = *v;
            if (auto* a = (*gt)["coefficients"].as_array()) {
                gc.coefficients = Vector(static_cast<int>(a->size()));
                for (int i = 0; i < static_cast<int>(a->size()); ++i)
                    gc.coefficients[i] = a->get(i)->value_or(0.0);
            }
            out.groups.push_back(std::move(gc));
        }
    }
}

// ── MVO parameter readers ─────────────────────────────────────────────────────

static MVOParameters parseMVOFromJSON(const json& j) {
    MVOParameters p;
    p.risk_aversion     = j.value("risk_aversion",     p.risk_aversion);
    p.frontier_points   = j.value("frontier_points",   p.frontier_points);
    p.min_risk_aversion = j.value("min_risk_aversion", p.min_risk_aversion);
    p.max_risk_aversion = j.value("max_risk_aversion", p.max_risk_aversion);
    p.risk_free_rate    = j.value("risk_free_rate",    p.risk_free_rate);
    p.group_penalty     = j.value("group_penalty",     p.group_penalty);

    if (j.contains("constraints"))
        parseConstraintsJSON(j["constraints"], p.constraints);
    return p;
}

static MVOParameters parseMVOFromTOML(const toml::table& t) {
    MVOParameters p;
    if (auto v = t["risk_aversion"].value<double>())     p.risk_aversion = *v;
    if (auto v = t["frontier_points"].value<int64_t>())  p.frontier_points = static_cast<int>(*v);
    if (auto v = t["min_risk_aversion"].value<double>()) p.min_risk_aversion = *v;
    if (auto v = t["max_risk_aversion"].value<double>()) p.max_risk_aversion = *v;
    if (auto v = t["risk_free_rate"].value<double>())    p.risk_free_rate = *v;
    if (auto v = t["group_penalty"].value<double>())     p.group_penalty = *v;
    if (auto* c = t["constraints"].as_table())
        parseConstraintsTOML(*c, p.constraints);
    return p;
}

MVOParameters readMVOParameters(const std::filesystem::path& path, Format fmt) {
    if (fmt == Format::Auto) fmt = inferFormat(path);
    log::info("Reading MVO parameters from '{}'", path.string());

    if (fmt == Format::JSON) {
        json j = json::parse(readFile(path));
        const json& node = j.contains("mvo") ? j["mvo"] : j;
        return parseMVOFromJSON(node);
    }

    if (fmt == Format::TOML) {
        auto tbl = toml::parse_file(path.string());
        if (tbl.contains("mvo") && tbl["mvo"].as_table())
            return parseMVOFromTOML(*tbl["mvo"].as_table());
        return parseMVOFromTOML(tbl);
    }

    throw std::invalid_argument("readMVOParameters: unsupported format");
}

MVOParameters readMVOParametersFromJSON(const std::string& json_str) {
    json j = json::parse(json_str);
    const json& node = j.contains("mvo") ? j["mvo"] : j;
    return parseMVOFromJSON(node);
}

// ── Black-Litterman parameter readers ────────────────────────────────────────

static ViewConfidenceMode parseConfidenceMode(const std::string& s,
                                              ViewConfidenceMode dflt) {
    std::string l = s;
    std::transform(l.begin(), l.end(), l.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (l == "idzorek")  return ViewConfidenceMode::Idzorek;
    if (l == "variance") return ViewConfidenceMode::Variance;
    return dflt;
}

static BlackLittermanParameters parseBLFromJSON(const json& bl_node,
                                                const json& root) {
    BlackLittermanParameters p;
    p.tau            = bl_node.value("tau",           p.tau);
    p.risk_aversion  = bl_node.value("risk_aversion", p.risk_aversion);
    if (bl_node.contains("confidence_mode"))
        p.confidence_mode = parseConfidenceMode(
            bl_node["confidence_mode"].get<std::string>(), p.confidence_mode);
    p.propagate_risk_aversion =
        bl_node.value("propagate_risk_aversion", p.propagate_risk_aversion);

    if (bl_node.contains("views")) {
        for (const auto& v : bl_node["views"]) {
            View view;
            view.description     = v.value("description",     "");
            view.expected_return = v.value("expected_return", 0.0);
            view.confidence      = v.value("confidence",      0.1);
            const auto& pv = v["pick_vector"];
            view.pick_vector = Vector(static_cast<int>(pv.size()));
            for (std::size_t i = 0; i < pv.size(); ++i)
                view.pick_vector[i] = pv[i].get<double>();
            p.views.push_back(std::move(view));
        }
    }

    // MVO parameters: bl_node.mvo > root.mvo > root (top-level).
    if (bl_node.contains("mvo")) {
        p.mvo_params = parseMVOFromJSON(bl_node["mvo"]);
    } else if (root.contains("mvo")) {
        p.mvo_params = parseMVOFromJSON(root["mvo"]);
    } else {
        p.mvo_params = parseMVOFromJSON(root);
    }

    // Constraints directly under [black_litterman] take precedence over MVO.
    if (bl_node.contains("constraints"))
        parseConstraintsJSON(bl_node["constraints"], p.mvo_params.constraints);

    return p;
}

BlackLittermanParameters readBLParameters(const std::filesystem::path& path,
                                          Format fmt) {
    if (fmt == Format::Auto) fmt = inferFormat(path);
    log::info("Reading Black-Litterman parameters from '{}'", path.string());

    if (fmt == Format::JSON) {
        json j = json::parse(readFile(path));
        const json& node = j.contains("black_litterman") ? j["black_litterman"] : j;
        return parseBLFromJSON(node, j);
    }

    if (fmt == Format::TOML) {
        auto tbl = toml::parse_file(path.string());
        BlackLittermanParameters p;

        const toml::table& bl_node = tbl.contains("black_litterman")
                                        ? *tbl["black_litterman"].as_table()
                                        : tbl;

        if (auto v = bl_node["tau"].value<double>())           p.tau = *v;
        if (auto v = bl_node["risk_aversion"].value<double>()) p.risk_aversion = *v;
        if (auto v = bl_node["confidence_mode"].value<std::string>())
            p.confidence_mode = parseConfidenceMode(*v, p.confidence_mode);
        if (auto v = bl_node["propagate_risk_aversion"].value<bool>())
            p.propagate_risk_aversion = *v;

        if (auto* views = bl_node["views"].as_array()) {
            for (const auto& ve : *views) {
                if (const auto* vt = ve.as_table()) {
                    View view;
                    if (auto v2 = (*vt)["description"].value<std::string>())
                        view.description = *v2;
                    if (auto v2 = (*vt)["expected_return"].value<double>())
                        view.expected_return = *v2;
                    if (auto v2 = (*vt)["confidence"].value<double>())
                        view.confidence = *v2;
                    if (auto* pv = (*vt)["pick_vector"].as_array()) {
                        view.pick_vector = Vector(static_cast<int>(pv->size()));
                        for (int i = 0; i < static_cast<int>(pv->size()); ++i)
                            view.pick_vector[i] = pv->get(i)->value_or(0.0);
                    }
                    p.views.push_back(std::move(view));
                }
            }
        }

        if (auto* m = bl_node["mvo"].as_table())
            p.mvo_params = parseMVOFromTOML(*m);
        else if (auto* m = tbl["mvo"].as_table())
            p.mvo_params = parseMVOFromTOML(*m);
        else
            p.mvo_params = parseMVOFromTOML(tbl);

        if (auto* c = bl_node["constraints"].as_table())
            parseConstraintsTOML(*c, p.mvo_params.constraints);

        return p;
    }

    throw std::invalid_argument("readBLParameters: unsupported format");
}

BlackLittermanParameters readBLParametersFromJSON(const std::string& json_str) {
    json j = json::parse(json_str);
    const json& node = j.contains("black_litterman") ? j["black_litterman"] : j;
    return parseBLFromJSON(node, j);
}

} // namespace io
} // namespace portopt
