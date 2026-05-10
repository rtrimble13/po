# portopt API Reference

## C++ API

All public types and functions are in namespace `portopt`.
Include the umbrella header: `#include <portopt/portopt.hpp>`

---

### Data types (`types.hpp`)

#### `Asset`

```cpp
struct Asset {
    std::string ticker;          // Unique identifier
    std::string name;            // Human-readable name
    double expected_return;      // Annualised expected excess return
    double market_cap;           // Market capitalisation
};
```

#### `PortfolioConstraints`

```cpp
struct PortfolioConstraints {
    Vector lower_bounds;         // Per-asset weight lower bounds
    Vector upper_bounds;         // Per-asset weight upper bounds
    bool   allow_short_selling;  // Informational flag

    // Factory methods
    static PortfolioConstraints longOnly(int n);
    static PortfolioConstraints withShorts(int n, double max_short = 1.0);

    void validate(int n) const;  // Throws std::invalid_argument on bad config
};
```

#### `MVOParameters`

```cpp
struct MVOParameters {
    double risk_aversion;        // λ (default 1.0)
    int    frontier_points;      // Number of frontier points (default 50)
    double min_risk_aversion;    // Frontier sweep lower bound (default 0.01)
    double max_risk_aversion;    // Frontier sweep upper bound (default 100.0)
    PortfolioConstraints constraints;
};
```

#### `View`

```cpp
struct View {
    std::string description;     // Human-readable label
    Vector      pick_vector;     // Row of the P matrix (length n)
    double      expected_return; // q: expected excess return for this view
    double      confidence;      // ω_i: view variance (must be > 0)
};
```

#### `BlackLittermanParameters`

```cpp
struct BlackLittermanParameters {
    double tau;                  // Prior uncertainty scaling (default 0.05)
    double risk_aversion;        // Market risk aversion δ (default 2.5)
    std::vector<View> views;     // Investor views
    MVOParameters mvo_params;    // MVO parameters for posterior optimisation
};
```

#### `PortfolioMetrics`

```cpp
struct PortfolioMetrics {
    double expected_return;      // μ'w
    double volatility;           // sqrt(w'Σw)
    double sharpe_ratio;         // expected_return / volatility
    double variance;             // w'Σw
};
```

#### `OptimizationResult`

```cpp
struct OptimizationResult {
    Vector             weights;       // Optimal weights (length n)
    PortfolioMetrics   metrics;       // Portfolio analytics
    std::vector<Asset> assets;        // Asset descriptors
    bool               converged;
    int                iterations;
    std::string        method;        // "MVO" or "Black-Litterman"
    std::string        status_message;
};
```

#### `EfficientFrontier`

```cpp
struct EfficientFrontier {
    std::vector<EfficientFrontierPoint> points;  // One per λ value
    std::vector<Asset>                  assets;
    std::string                         method;
};

struct EfficientFrontierPoint {
    double           risk_aversion;
    Vector           weights;
    PortfolioMetrics metrics;
};
```

#### `MarketData`

```cpp
struct MarketData {
    AssetUniverse         assets;            // Vector of Asset
    Vector                expected_returns;  // μ (n×1)
    Matrix                covariance;        // Σ (n×n)
    std::optional<Vector> market_weights;    // w_mkt (optional, needed for BL)
};
```

#### `BLModelOutput`

```cpp
struct BLModelOutput {
    Vector prior_returns;        // π = δ Σ w_mkt
    Vector posterior_returns;    // μ_BL
    Matrix posterior_cov;        // Σ_BL
    Matrix blended_cov;          // Σ + Σ_BL (used in MVO)
    Matrix pick_matrix;          // P
    Vector view_returns;         // q
    Matrix view_uncertainty;     // Ω (diagonal)
};
```

---

### QP Solver (`qp_solver.hpp`)

```cpp
namespace portopt::qp {

// Solve min 0.5 x'Qx + f'x  s.t. sum(x)=budget, lb≤x≤ub
SolverResult solve(const Matrix& Q, const Vector& f,
                   const Vector& lb, const Vector& ub,
                   const SolverConfig& cfg = {});

// Project v onto {x : sum(x)=s, lb≤x≤ub}
Vector projectOntoSimplex(const Vector& v, double s,
                           const Vector& lb, const Vector& ub);

// Estimate largest eigenvalue of M via power iteration
double largestEigenvalue(const Matrix& M,
                         int max_iter = 200, double tol = 1e-10);

struct SolverConfig {
    int    max_iterations{10000};
    double tolerance{1e-9};
    double budget{1.0};
    bool   use_nesterov{true};
};

struct SolverResult {
    Vector x;
    double objective;
    int    iterations;
    bool   converged;
    double primal_residual;
};

} // namespace portopt::qp
```

---

### MVOptimizer (`mvo.hpp`)

```cpp
class MVOptimizer : public IOptimizer {
public:
    explicit MVOptimizer(MVOParameters params = {});

    // Compute single optimal portfolio using params.risk_aversion
    OptimizationResult optimize(const MarketData& data) override;

    // Compute full efficient frontier (logarithmic λ sweep)
    EfficientFrontier efficientFrontier(const MarketData& data) override;

    void setParameters(const MVOParameters& params);
    const MVOParameters& parameters() const;

    // Compute portfolio metrics for given weights (static utility)
    static PortfolioMetrics computeMetrics(const Vector& weights,
                                            const Vector& mu,
                                            const Matrix& sigma);
};
```

**Exceptions**: `std::invalid_argument` for bad input dimensions or constraints.

---

### BlackLittermanOptimizer (`black_litterman.hpp`)

```cpp
class BlackLittermanOptimizer : public IOptimizer {
public:
    explicit BlackLittermanOptimizer(BlackLittermanParameters params = {});

    // Compute BL-optimal portfolio (requires data.market_weights)
    OptimizationResult optimize(const MarketData& data) override;

    // Compute BL efficient frontier
    EfficientFrontier efficientFrontier(const MarketData& data) override;

    // Return BL model internals (prior, posterior, views)
    BLModelOutput modelOutput(const MarketData& data) const;

    void setParameters(const BlackLittermanParameters& params);
    const BlackLittermanParameters& parameters() const;
};
```

**Exceptions**: `std::invalid_argument` for missing market weights, dimension
mismatches, zero confidence values, or non-unit-sum market weights.
`std::runtime_error` if the posterior covariance is not positive definite.

---

### IO — Reader (`io/reader.hpp`)

```cpp
namespace portopt::io {

enum class Format { JSON, CSV, TOML, Auto };

// Infer format from file extension
Format inferFormat(const std::filesystem::path& path);

// Read market data from JSON or CSV file (Auto infers from extension)
MarketData readMarketData(const std::filesystem::path& path,
                          Format fmt = Format::Auto);

// Parse market data from a JSON string
MarketData readMarketDataFromJSON(const std::string& json_str);

// Read from separate assets.csv + covariance.csv
MarketData readMarketDataFromCSV(
    const std::filesystem::path& assets_csv,
    const std::filesystem::path& covariance_csv,
    const std::filesystem::path* weights_csv = nullptr);

// Read MVO parameters from JSON or TOML
MVOParameters readMVOParameters(const std::filesystem::path& path,
                                Format fmt = Format::Auto);

// Read BL parameters from JSON or TOML
BlackLittermanParameters readBLParameters(const std::filesystem::path& path,
                                          Format fmt = Format::Auto);

} // namespace portopt::io
```

---

### IO — Writer (`io/writer.hpp`)

```cpp
namespace portopt::io {

enum class OutputFormat { Console, JSON, CSV };

struct WriterConfig {
    OutputFormat format{OutputFormat::Console};
    int          json_indent{2};
    int          console_weight_prec{4};
    int          console_return_prec{4};
    bool         show_zero_weights{false};
    double       weight_threshold{1e-6};
};

// Write OptimizationResult
void writeResult(const OptimizationResult& result,
                 std::ostream& out,
                 const WriterConfig& cfg = {});
void writeResult(const OptimizationResult& result,
                 const std::filesystem::path& path,
                 const WriterConfig& cfg = {});
std::string resultToJSON(const OptimizationResult& result, int indent = 2);
std::string resultToCSV(const OptimizationResult& result);

// Write EfficientFrontier
void writeFrontier(const EfficientFrontier& frontier,
                   std::ostream& out,
                   const WriterConfig& cfg = {});
void writeFrontier(const EfficientFrontier& frontier,
                   const std::filesystem::path& path,
                   const WriterConfig& cfg = {});
std::string frontierToJSON(const EfficientFrontier& frontier, int indent = 2);
std::string frontierToCSV(const EfficientFrontier& frontier);

// Write BLModelOutput
void writeBLModel(const BLModelOutput& bl, const AssetUniverse& assets,
                  std::ostream& out, const WriterConfig& cfg = {});
std::string blModelToJSON(const BLModelOutput& bl,
                           const AssetUniverse& assets, int indent = 2);

} // namespace portopt::io
```

---

### Logging (`logging.hpp`)

```cpp
namespace portopt::log {

enum class Level { Trace, Debug, Info, Warn, Error, Critical, Off };

void init(Level level = Level::Info,
          bool  console = true,
          bool  force_reinit = false);

void setLevel(Level level);

void addFileLog(const std::string& filename,
                Level level = Level::Trace);

std::shared_ptr<spdlog::logger> getLogger();

// Templated logging functions (same interface as spdlog)
template<typename... Args>
void trace(spdlog::format_string_t<Args...> fmt, Args&&... args);
void debug(...);
void info(...);
void warn(...);
void error(...);

} // namespace portopt::log
```

---

## Python API

The Python module (`portopt`) exposes the same types and functions with
Pythonic naming conventions (snake_case). All Eigen matrices/vectors are
automatically converted to/from NumPy arrays.

### Module-level functions

| Function | Description |
|---|---|
| `init_logging(level, console, force)` | Initialise the library logger |
| `set_log_level(level)` | Change log level at runtime |
| `add_file_log(filename, level)` | Add rotating file sink |
| `read_market_data(path)` | Read market data from file |
| `read_market_data_json(json_str)` | Parse market data from string |
| `read_mvo_parameters(path)` | Read MVO params from file |
| `read_bl_parameters(path)` | Read BL params from file |
| `result_to_json(result, indent)` | Serialise result to JSON string |
| `frontier_to_json(frontier, indent)` | Serialise frontier to JSON string |
| `bl_model_to_json(model, assets, indent)` | Serialise BL model to JSON |

### `portopt.LogLevel` enum

`Trace`, `Debug`, `Info`, `Warn`, `Error`, `Critical`, `Off`

### `portopt.MarketData`

```python
data = portopt.MarketData.from_arrays(
    assets=list_of_assets,
    expected_returns=np.ndarray,
    covariance=np.ndarray,
    market_weights=np.ndarray,  # optional
)
data.assets           # list[Asset]
data.expected_returns # np.ndarray
data.covariance       # np.ndarray
data.market_weights   # Optional[np.ndarray]
```

### `portopt.MVOptimizer`

```python
opt = portopt.MVOptimizer(params: MVOParameters = MVOParameters())
result   = opt.optimize(data: MarketData) -> OptimizationResult
frontier = opt.efficient_frontier(data: MarketData) -> EfficientFrontier
opt.set_parameters(params: MVOParameters)
opt.parameters  # property
portopt.MVOptimizer.compute_metrics(weights, mu, sigma) -> PortfolioMetrics
```

### `portopt.BlackLittermanOptimizer`

```python
bl = portopt.BlackLittermanOptimizer(params: BlackLittermanParameters = ...)
result   = bl.optimize(data: MarketData) -> OptimizationResult
frontier = bl.efficient_frontier(data: MarketData) -> EfficientFrontier
model    = bl.model_output(data: MarketData) -> BLModelOutput
```

### `portopt.EfficientFrontier`

```python
frontier.points  # list[EfficientFrontierPoint]
frontier.assets  # list[Asset]
frontier.method  # str
frontier.to_dataframe()  # -> pd.DataFrame (requires pandas)
```

### `portopt.OptimizationResult`

```python
result.weights          # np.ndarray
result.metrics          # PortfolioMetrics
result.assets           # list[Asset]
result.converged        # bool
result.iterations       # int
result.method           # str
result.status_message   # str
result.to_dict()        # -> dict
```

---

## Helper functions (`portopt.helpers`)

```python
from portopt.helpers import (
    make_sample_data,
    plot_efficient_frontier,
    plot_weights,
    plot_bl_comparison,
    frontier_to_dataframe,
)

# Generate synthetic data
data = make_sample_data(n_assets=5, seed=42)

# Matplotlib plots (return ax for further customisation)
ax = plot_efficient_frontier(frontier, ax=None, label=None, show_sharpe=True)
ax = plot_weights(result, ax=None, threshold=1e-4)
ax = plot_bl_comparison(bl_model, assets, ax=None)

# pandas DataFrame with all frontier data including per-asset weights
df = frontier_to_dataframe(frontier)
```
