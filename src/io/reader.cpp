#include <portopt/io/reader.hpp>
#include <portopt/logging.hpp>

#include <nlohmann/json.hpp>
#include <toml++/toml.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace portopt {
namespace io {

using json = nlohmann::json;

// ── Helpers ───────────────────────────────────────────────────────────────────

Format inferFormat(const std::filesystem::path& path) {
    auto ext = path.extension().string();
    for (auto& c : ext) c = static_cast<char>(std::tolower(c));
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

static std::vector<std::string> splitCSVLine(const std::string& line) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        // Trim whitespace
        auto start = tok.find_first_not_of(" \t\r\n");
        auto end   = tok.find_last_not_of(" \t\r\n");
        if (start == std::string::npos) tokens.push_back("");
        else tokens.push_back(tok.substr(start, end - start + 1));
    }
    return tokens;
}

// ── JSON market data ──────────────────────────────────────────────────────────

MarketData readMarketDataFromJSON(const std::string& json_str) {
    json j = json::parse(json_str);
    MarketData data;

    // Assets array
    if (!j.contains("assets"))
        throw std::runtime_error("JSON: missing 'assets' array");

    for (const auto& a : j["assets"]) {
        Asset asset;
        asset.ticker          = a.value("ticker", "");
        asset.name            = a.value("name", asset.ticker);
        asset.expected_return = a.value("expected_return", 0.0);
        asset.market_cap      = a.value("market_cap", 0.0);
        data.assets.push_back(asset);
    }

    const int n = static_cast<int>(data.assets.size());
    if (n == 0)
        throw std::runtime_error("JSON: assets array is empty");

    // Expected returns (may be embedded or inferred from assets)
    if (j.contains("expected_returns")) {
        auto& er = j["expected_returns"];
        data.expected_returns = Vector(n);
        for (int i = 0; i < n; ++i)
            data.expected_returns[i] = er[i].get<double>();
    } else {
        // Fall back to per-asset expected_return field
        data.expected_returns = Vector(n);
        for (int i = 0; i < n; ++i)
            data.expected_returns[i] = data.assets[i].expected_return;
    }

    // Covariance matrix
    if (!j.contains("covariance"))
        throw std::runtime_error("JSON: missing 'covariance' matrix");

    auto& cov = j["covariance"];
    if (static_cast<int>(cov.size()) != n)
        throw std::runtime_error("JSON: covariance row count != asset count");

    data.covariance = Matrix(n, n);
    for (int i = 0; i < n; ++i) {
        if (static_cast<int>(cov[i].size()) != n)
            throw std::runtime_error("JSON: covariance matrix is not square");
        for (int jj = 0; jj < n; ++jj)
            data.covariance(i, jj) = cov[i][jj].get<double>();
    }

    // Optional market weights
    if (j.contains("market_weights")) {
        Vector mw(n);
        for (int i = 0; i < n; ++i)
            mw[i] = j["market_weights"][i].get<double>();
        data.market_weights = mw;
    }

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
    {
        std::ifstream f(assets_csv);
        if (!f.is_open())
            throw std::runtime_error("Cannot open: " + assets_csv.string());

        std::string header_line;
        std::getline(f, header_line);
        // Expected header: ticker,name,expected_return,market_cap

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
            a.market_cap      = cols.size() > 3 ? std::stod(cols[3]) : 0.0;
            data.assets.push_back(a);
        }
    }

    const int n = static_cast<int>(data.assets.size());
    if (n == 0)
        throw std::runtime_error("Assets CSV: no assets found");

    // Build expected returns from assets
    data.expected_returns = Vector(n);
    for (int i = 0; i < n; ++i)
        data.expected_returns[i] = data.assets[i].expected_return;

    // ── Covariance CSV ────────────────────────────────────────────────────────
    {
        std::ifstream f(covariance_csv);
        if (!f.is_open())
            throw std::runtime_error("Cannot open: " + covariance_csv.string());

        // Skip header row (tickers)
        std::string header_line;
        std::getline(f, header_line);

        data.covariance = Matrix(n, n);
        std::string line;
        int row = 0;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            if (row >= n)
                throw std::runtime_error("Covariance CSV: too many rows");
            auto cols = splitCSVLine(line);
            // First column may be ticker label — detect by checking parsability
            int col_offset = 0;
            try { std::stod(cols[0]); } catch (...) { col_offset = 1; }

            if (static_cast<int>(cols.size()) - col_offset < n)
                throw std::runtime_error("Covariance CSV: not enough columns in row "
                                         + std::to_string(row));
            for (int c = 0; c < n; ++c)
                data.covariance(row, c) = std::stod(cols[col_offset + c]);
            ++row;
        }
        if (row != n)
            throw std::runtime_error("Covariance CSV: row count " +
                                     std::to_string(row) + " != " + std::to_string(n));
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
            // Expect: ticker,weight  OR just weight
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

// ── File dispatch ─────────────────────────────────────────────────────────────

MarketData readMarketData(const std::filesystem::path& path, Format fmt) {
    if (fmt == Format::Auto) fmt = inferFormat(path);
    log::info("Reading market data from '{}' ({})",
              path.string(), fmt == Format::JSON ? "JSON" : "CSV");

    if (fmt == Format::JSON)
        return readMarketDataFromJSON(readFile(path));

    // For CSV, expect the same directory to contain assets.csv + covariance.csv
    auto dir = path.parent_path();
    auto stem = path.stem().string();
    // If path is assets.csv, look for covariance.csv alongside it
    auto cov_path = dir / "covariance.csv";
    if (!std::filesystem::exists(cov_path)) {
        // Try <stem>_covariance.csv
        cov_path = dir / (stem + "_covariance.csv");
    }
    return readMarketDataFromCSV(path, cov_path, nullptr);
}

// ── MVO parameter readers ─────────────────────────────────────────────────────

static MVOParameters parseMVOFromJSON(const json& j) {
    MVOParameters p;
    p.risk_aversion     = j.value("risk_aversion",     p.risk_aversion);
    p.frontier_points   = j.value("frontier_points",   p.frontier_points);
    p.min_risk_aversion = j.value("min_risk_aversion", p.min_risk_aversion);
    p.max_risk_aversion = j.value("max_risk_aversion", p.max_risk_aversion);

    if (j.contains("constraints")) {
        auto& c = j["constraints"];
        int n = 0; // unknown until data loaded; use size from arrays if present
        if (c.contains("lower_bounds")) {
            auto& lb = c["lower_bounds"];
            n = static_cast<int>(lb.size());
            p.constraints.lower_bounds = Vector(n);
            for (int i = 0; i < n; ++i)
                p.constraints.lower_bounds[i] = lb[i].get<double>();
        }
        if (c.contains("upper_bounds")) {
            auto& ub = c["upper_bounds"];
            int ub_n = static_cast<int>(ub.size());
            p.constraints.upper_bounds = Vector(ub_n);
            for (int i = 0; i < ub_n; ++i)
                p.constraints.upper_bounds[i] = ub[i].get<double>();
        }
        p.constraints.allow_short_selling = c.value("allow_short_selling", false);
    }
    return p;
}

static MVOParameters parseMVOFromTOML(const toml::table& t) {
    MVOParameters p;
    if (auto v = t["risk_aversion"].value<double>())     p.risk_aversion = *v;
    if (auto v = t["frontier_points"].value<int64_t>())  p.frontier_points = static_cast<int>(*v);
    if (auto v = t["min_risk_aversion"].value<double>()) p.min_risk_aversion = *v;
    if (auto v = t["max_risk_aversion"].value<double>()) p.max_risk_aversion = *v;

    if (auto* c = t["constraints"].as_table()) {
        if (auto* lb = (*c)["lower_bounds"].as_array()) {
            int n = static_cast<int>(lb->size());
            p.constraints.lower_bounds = Vector(n);
            for (int i = 0; i < n; ++i)
                p.constraints.lower_bounds[i] = lb->get(i)->value_or(0.0);
        }
        if (auto* ub = (*c)["upper_bounds"].as_array()) {
            int n = static_cast<int>(ub->size());
            p.constraints.upper_bounds = Vector(n);
            for (int i = 0; i < n; ++i)
                p.constraints.upper_bounds[i] = ub->get(i)->value_or(1.0);
        }
        if (auto v = (*c)["allow_short_selling"].value<bool>())
            p.constraints.allow_short_selling = *v;
    }
    return p;
}

MVOParameters readMVOParameters(const std::filesystem::path& path, Format fmt) {
    if (fmt == Format::Auto) fmt = inferFormat(path);
    log::info("Reading MVO parameters from '{}'", path.string());

    if (fmt == Format::JSON) {
        json j = json::parse(readFile(path));
        auto& node = j.contains("mvo") ? j["mvo"] : j;
        return parseMVOFromJSON(node);
    }

    if (fmt == Format::TOML) {
        auto tbl = toml::parse_file(path.string());
        auto& node = tbl.contains("mvo") ? *tbl["mvo"].as_table() : tbl;
        return parseMVOFromTOML(node);
    }

    throw std::invalid_argument("readMVOParameters: unsupported format");
}

// ── Black-Litterman parameter readers ────────────────────────────────────────

static BlackLittermanParameters parseBLFromJSON(const json& j, const json& root) {
    BlackLittermanParameters p;
    p.tau            = j.value("tau",            p.tau);
    p.risk_aversion  = j.value("risk_aversion",  p.risk_aversion);

    if (j.contains("views")) {
        for (const auto& v : j["views"]) {
            View view;
            view.description     = v.value("description",     "");
            view.expected_return = v.value("expected_return",  0.0);
            view.confidence      = v.value("confidence",       0.1);

            auto& pv = v["pick_vector"];
            view.pick_vector = Vector(pv.size());
            for (size_t i = 0; i < pv.size(); ++i)
                view.pick_vector[i] = pv[i].get<double>();

            p.views.push_back(std::move(view));
        }
    }

    // MVO parameters can be nested or at root level
    auto& mvo_node = root.contains("mvo") ? root["mvo"] : (j.contains("mvo") ? j["mvo"] : j);
    p.mvo_params = parseMVOFromJSON(mvo_node);

    return p;
}

BlackLittermanParameters readBLParameters(const std::filesystem::path& path,
                                          Format fmt) {
    if (fmt == Format::Auto) fmt = inferFormat(path);
    log::info("Reading Black-Litterman parameters from '{}'", path.string());

    if (fmt == Format::JSON) {
        json j = json::parse(readFile(path));
        auto& node = j.contains("black_litterman") ? j["black_litterman"] : j;
        return parseBLFromJSON(node, j);
    }

    if (fmt == Format::TOML) {
        auto tbl = toml::parse_file(path.string());

        BlackLittermanParameters p;
        auto& bl_node = tbl.contains("black_litterman")
                        ? *tbl["black_litterman"].as_table() : tbl;

        if (auto v = bl_node["tau"].value<double>())           p.tau = *v;
        if (auto v = bl_node["risk_aversion"].value<double>()) p.risk_aversion = *v;

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

        auto& mvo_node = tbl.contains("mvo") ? *tbl["mvo"].as_table() : tbl;
        p.mvo_params = parseMVOFromTOML(mvo_node);

        return p;
    }

    throw std::invalid_argument("readBLParameters: unsupported format");
}

} // namespace io
} // namespace portopt
