#!/usr/bin/env python3
"""
example_from_json.py — Load market data and parameters from files, run MVO + BL.

Demonstrates the file-driven workflow that pairs naturally with the CLI:
  - `tests/data/assets.json`  — market data
  - `tests/data/params.json`  — MVO + BL parameters

Useful as a copy-paste template.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import portopt
except ImportError:
    sys.exit("portopt not found. Build with CMake (PORTOPT_BUILD_PYTHON=ON).")

ROOT = Path(__file__).resolve().parent.parent
data_path   = ROOT / "tests" / "data" / "assets.json"
params_path = ROOT / "tests" / "data" / "params.json"

portopt.init_logging(portopt.LogLevel.Info)

# ── Load ─────────────────────────────────────────────────────────────────────
data = portopt.read_market_data(str(data_path))
print(f"Loaded {len(data.assets)} assets from {data_path.name}: "
      f"{[a.ticker for a in data.assets]}")

# ── MVO from params file ─────────────────────────────────────────────────────
mvo_params = portopt.read_mvo_parameters(str(params_path))
mvo = portopt.MVOptimizer(mvo_params)
mvo_result = mvo.optimize(data)
print(f"\nMVO (λ = {mvo_params.risk_aversion}):")
print(f"  return     : {mvo_result.metrics.expected_return * 100:.2f}%")
print(f"  volatility : {mvo_result.metrics.volatility * 100:.2f}%")
print(f"  Sharpe     : {mvo_result.metrics.sharpe_ratio:.3f}")

# ── BL from params file ──────────────────────────────────────────────────────
bl_params = portopt.read_bl_parameters(str(params_path))
print(f"\nBL has {len(bl_params.views)} views, confidence_mode="
      f"{bl_params.confidence_mode.name}")
bl = portopt.BlackLittermanOptimizer(bl_params)
bl_result = bl.optimize(data)
print(f"\nBL (δ = {bl_params.risk_aversion}, λ = "
      f"{bl_params.mvo_params.risk_aversion}):")
print(f"  return     : {bl_result.metrics.expected_return * 100:.2f}%")
print(f"  volatility : {bl_result.metrics.volatility * 100:.2f}%")
print(f"  Sharpe     : {bl_result.metrics.sharpe_ratio:.3f}")

# ── PM-friendly targets ──────────────────────────────────────────────────────
print("\nPM-friendly portfolios:")
for label, fn in [
    ("min-variance",  lambda: mvo.min_variance_portfolio(data)),
    ("max-Sharpe",    lambda: mvo.max_sharpe_portfolio(data)),
    ("target vol 15%", lambda: mvo.optimize_for_target_volatility(data, 0.15)),
    ("target ret 12%", lambda: mvo.optimize_for_target_return(data, 0.12)),
]:
    r = fn()
    print(f"  {label:18s}  ret={r.metrics.expected_return * 100:5.2f}%  "
          f"vol={r.metrics.volatility * 100:5.2f}%  "
          f"Sharpe={r.metrics.sharpe_ratio:5.3f}")
