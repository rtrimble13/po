#!/usr/bin/env python3
"""
generate_report.py — CLI wrapper to execute the portopt diagnostic notebook.

Usage:
    python notebooks/generate_report.py \\
        --data    tests/data/assets.json \\
        --params  tests/data/params.json \\
        --method  both \\
        --output  reports/

This script:
  1. Runs the optimisations (MVO and/or Black-Litterman).
  2. Serialises intermediate results to JSON in --output.
  3. Executes diagnostic_template.ipynb via nbconvert, injecting
     the output directory as a parameter.
  4. Optionally exports an HTML version of the executed notebook.

Requirements:
    pip install jupyter nbconvert nbformat portopt matplotlib pandas
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _check_import(pkg: str) -> bool:
    import importlib
    try:
        importlib.import_module(pkg)
        return True
    except ImportError:
        return False


def run_optimisations(data_path: str, params_path: str | None,
                      method: str, output_dir: Path) -> dict:
    """Run optimisations and write JSON intermediates. Returns paths dict."""
    try:
        import portopt
    except ImportError:
        sys.exit(
            "ERROR: portopt Python module not found.\n"
            "Build with CMake (PORTOPT_BUILD_PYTHON=ON) and add the "
            "build/python directory to PYTHONPATH."
        )

    portopt.init_logging(portopt.LogLevel.Info)

    print(f"[portopt] Reading market data from {data_path}")
    data = portopt.read_market_data(data_path)
    n = len(data.assets)
    print(f"[portopt] Loaded {n} assets: {[a.ticker for a in data.assets]}")

    # ── MVO ───────────────────────────────────────────────────────────────────
    mvo_params = portopt.MVOParameters()
    if params_path:
        try:
            mvo_params = portopt.read_mvo_parameters(params_path)
        except Exception as e:
            print(f"[portopt] Warning: could not read MVO params ({e}), using defaults")

    if (mvo_params.constraints.lower_bounds is None or
            len(mvo_params.constraints.lower_bounds) != n):
        mvo_params.constraints = portopt.PortfolioConstraints.long_only(n)

    mvo = portopt.MVOptimizer(mvo_params)
    mvo_result   = mvo.optimize(data)
    mvo_frontier = mvo.efficient_frontier(data)

    paths = {}
    with open(output_dir / "mvo_result.json", "w") as f:
        f.write(portopt.result_to_json(mvo_result))
    with open(output_dir / "mvo_frontier.json", "w") as f:
        f.write(portopt.frontier_to_json(mvo_frontier))
    paths["mvo_result"]   = str(output_dir / "mvo_result.json")
    paths["mvo_frontier"] = str(output_dir / "mvo_frontier.json")
    print(f"[portopt] MVO: return={mvo_result.metrics.expected_return:.4f} "
          f"vol={mvo_result.metrics.volatility:.4f} "
          f"sharpe={mvo_result.metrics.sharpe_ratio:.4f}")

    # ── Black-Litterman ───────────────────────────────────────────────────────
    if method in ("bl", "both") and data.market_weights is not None:
        bl_params = portopt.BlackLittermanParameters()
        if params_path:
            try:
                bl_params = portopt.read_bl_parameters(params_path)
            except Exception as e:
                print(f"[portopt] Warning: could not read BL params ({e}), using defaults")

        if (bl_params.mvo_params.constraints.lower_bounds is None or
                len(bl_params.mvo_params.constraints.lower_bounds) != n):
            bl_params.mvo_params.constraints = portopt.PortfolioConstraints.long_only(n)

        bl = portopt.BlackLittermanOptimizer(bl_params)
        try:
            bl_result   = bl.optimize(data)
            bl_frontier = bl.efficient_frontier(data)
            bl_model    = bl.model_output(data)

            with open(output_dir / "bl_result.json", "w") as f:
                f.write(portopt.result_to_json(bl_result))
            with open(output_dir / "bl_frontier.json", "w") as f:
                f.write(portopt.frontier_to_json(bl_frontier))
            with open(output_dir / "bl_model.json", "w") as f:
                f.write(portopt.bl_model_to_json(bl_model, data.assets))
            paths["bl_result"]   = str(output_dir / "bl_result.json")
            paths["bl_frontier"] = str(output_dir / "bl_frontier.json")
            paths["bl_model"]    = str(output_dir / "bl_model.json")
            print(f"[portopt] BL: return={bl_result.metrics.expected_return:.4f} "
                  f"vol={bl_result.metrics.volatility:.4f}")
        except Exception as e:
            print(f"[portopt] BL optimisation failed: {e}")
    elif method in ("bl", "both") and data.market_weights is None:
        print("[portopt] Warning: market_weights missing — skipping BL")

    # Write manifest
    manifest = {
        "data_path":   str(Path(data_path).resolve()),
        "params_path": str(Path(params_path).resolve()) if params_path else None,
        "method":      method,
        "output_dir":  str(output_dir.resolve()),
        "assets":      [{"ticker": a.ticker, "name": a.name}
                        for a in data.assets],
        **paths,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def execute_notebook(notebook_path: str, output_dir: Path,
                     html: bool = True) -> Path:
    """Execute the Jupyter notebook, injecting output_dir as a parameter."""
    nb_in  = Path(notebook_path).resolve()
    nb_out = output_dir / "diagnostic_report.ipynb"

    env = os.environ.copy()
    env["PORTOPT_OUTPUT_DIR"] = str(output_dir.resolve())

    cmd = [
        sys.executable, "-m", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=600",
        "--ExecutePreprocessor.kernel_name=python3",
        "--output", str(nb_out),
        str(nb_in),
    ]

    print(f"[notebook] Executing: {nb_in.name}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print("nbconvert stderr:\n", result.stderr)
        sys.exit(f"ERROR: nbconvert failed (exit {result.returncode})")

    print(f"[notebook] Executed notebook: {nb_out}")

    if html:
        html_out = output_dir / "diagnostic_report.html"
        html_cmd = [
            sys.executable, "-m", "nbconvert",
            "--to", "html",
            "--output", str(html_out),
            str(nb_out),
        ]
        subprocess.run(html_cmd, check=False)
        if html_out.exists():
            print(f"[notebook] HTML report: {html_out}")

    return nb_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate portopt diagnostic Jupyter report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-d", "--data",     required=True,
                        help="Market data file (.json or .csv)")
    parser.add_argument("-p", "--params",   default=None,
                        help="Parameters file (.json or .toml)")
    parser.add_argument("-n", "--notebook",
                        default=str(Path(__file__).parent / "diagnostic_template.ipynb"),
                        help="Jupyter notebook template")
    parser.add_argument("-o", "--output",   default="reports",
                        help="Output directory")
    parser.add_argument("-m", "--method",   default="both",
                        choices=["mvo", "bl", "both"],
                        help="Optimisation method")
    parser.add_argument("--no-html",        action="store_true",
                        help="Skip HTML export")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = run_optimisations(args.data, args.params, args.method, output_dir)
    execute_notebook(args.notebook, output_dir, html=not args.no_html)

    print("\n[portopt] Report complete.")
    print(f"  Outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
