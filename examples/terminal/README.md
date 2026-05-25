# `po` command-line examples

This directory ships a runnable, walkthrough-style introduction to the `po`
CLI. Every numbered script under [`scripts/`](scripts/) demonstrates one
optimisation feature, has a matching `.md` walkthrough next to it, and writes
its output to [`var/`](var/) (gitignored).

> Looking for the Python API? See the scripts at the level above —
> [`examples/example_mvo.py`](../example_mvo.py),
> [`examples/example_bl.py`](../example_bl.py),
> [`examples/example_from_json.py`](../example_from_json.py),
> [`examples/example_from_returns.py`](../example_from_returns.py).

## Layout

```
examples/terminal/
├── _common.sh        — shared helper sourced by every script
├── data/             — dummy market data, returns, and parameter files
├── scripts/          — 01_*.sh through 16_*.sh + run_all.sh + .md walkthroughs
└── var/              — generated output, one sub-directory per script (gitignored)
```

## Prerequisites

You need the `po` binary on disk. Build it once:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel --target portopt_cli
```

The example scripts find `po` automatically by checking, in order:

1. `$PO_BIN` (explicit override)
2. `examples/terminal/bin/po` (staged by the `po-examples` CMake target — see below)
3. `build/cli/po` (default out-of-source CMake layout)
4. `build/bin/po`
5. `command -v po` (system-wide install)

### Optional: stage `po` at a stable path

If you want the binary at a known location that survives clean builds or that
your shell scripts can reference without env vars:

```bash
cmake --build build --target po-examples
# → examples/terminal/bin/po
```

`po-examples` is a custom target in [`cli/CMakeLists.txt`](../../cli/CMakeLists.txt)
that depends on `portopt_cli` and copies it under `examples/terminal/bin/`.
The staged file is gitignored.

## Running

Run a single example:

```bash
cd examples/terminal/scripts
./01_mvo_basic.sh
```

Or smoke-test the whole suite:

```bash
./run_all.sh
```

Each script writes everything it generates to `var/<script-name>/`. Re-running
overwrites that subdirectory.

## Example index

| # | Script | Feature |
|---|---|---|
| 01 | [scripts/01_mvo_basic.sh](scripts/01_mvo_basic.sh) ([md](scripts/01_mvo_basic.md))             | Baseline MVO portfolio |
| 02 | [scripts/02_mvo_frontier.sh](scripts/02_mvo_frontier.sh) ([md](scripts/02_mvo_frontier.md))       | MVO efficient frontier |
| 03 | [scripts/03_bl_basic.sh](scripts/03_bl_basic.sh) ([md](scripts/03_bl_basic.md))                 | Black-Litterman with views |
| 04 | [scripts/04_bl_frontier.sh](scripts/04_bl_frontier.sh) ([md](scripts/04_bl_frontier.md))         | Black-Litterman frontier |
| 05 | [scripts/05_min_variance.sh](scripts/05_min_variance.sh) ([md](scripts/05_min_variance.md))       | Minimum-variance portfolio |
| 06 | [scripts/06_max_sharpe.sh](scripts/06_max_sharpe.sh) ([md](scripts/06_max_sharpe.md))           | Max-Sharpe (tangency) portfolio |
| 07 | [scripts/07_target_vol.sh](scripts/07_target_vol.sh) ([md](scripts/07_target_vol.md))           | Target-volatility portfolio |
| 08 | [scripts/08_target_return.sh](scripts/08_target_return.sh) ([md](scripts/08_target_return.md)) | Target-return portfolio |
| 09 | [scripts/09_report.sh](scripts/09_report.sh) ([md](scripts/09_report.md))                       | Jupyter diagnostic report |
| 10 | [scripts/10_returns_shrinkage.sh](scripts/10_returns_shrinkage.sh) ([md](scripts/10_returns_shrinkage.md)) | Returns CSV + Ledoit-Wolf shrinkage |
| 11 | [scripts/11_csv_inputs.sh](scripts/11_csv_inputs.sh) ([md](scripts/11_csv_inputs.md))           | Market data via CSV pair |
| 12 | [scripts/12_group_constraints.sh](scripts/12_group_constraints.sh) ([md](scripts/12_group_constraints.md)) | Sector group caps |
| 13 | [scripts/13_turnover_penalty.sh](scripts/13_turnover_penalty.sh) ([md](scripts/13_turnover_penalty.md)) | L2 turnover penalty |
| 14 | [scripts/14_tracking_error.sh](scripts/14_tracking_error.sh) ([md](scripts/14_tracking_error.md))   | Tracking-error budget |
| 15 | [scripts/15_long_short.sh](scripts/15_long_short.sh) ([md](scripts/15_long_short.md))           | Dollar-neutral long/short |
| 16 | [scripts/16_output_formats.sh](scripts/16_output_formats.sh) ([md](scripts/16_output_formats.md)) | Console / JSON / CSV + notional $ |

## Out of scope (Python-only features)

A handful of features ship in the C++/Python libraries but are not exposed
through the parameter-file parser, so the CLI cannot reach them. See the
top-level [README.md](../../README.md) for examples in Python:

- Minimum-CVaR portfolios
- Factor-neutral / beta-neutral constraints (parameterised programmatically)
- Currency-exposure aggregation and FX hedging
- Linear / quadratic per-asset transaction costs
- Walk-forward backtesting and Brinson attribution
