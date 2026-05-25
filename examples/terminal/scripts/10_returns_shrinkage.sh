#!/usr/bin/env bash
# 10_returns_shrinkage.sh — Estimate μ, Σ from a daily-returns CSV with
# Ledoit-Wolf shrinkage, then run MVO.
source "$(dirname "$0")/../_common.sh"
banner "10 — Returns CSV + Ledoit-Wolf shrinkage"

# Plain sample covariance.
"$PO_BIN" mvo \
    --data             "$DATA_DIR/returns_daily.csv" \
    --returns \
    --periods-per-year 252 \
    --shrinkage        none \
    --output           "$OUT_DIR/mvo_sample.json" \
    --format           json

# Ledoit-Wolf shrunk covariance.
"$PO_BIN" mvo \
    --data             "$DATA_DIR/returns_daily.csv" \
    --returns \
    --periods-per-year 252 \
    --shrinkage        ledoit-wolf \
    --output           "$OUT_DIR/mvo_ledoit_wolf.json" \
    --format           json

# Console view for the shrunk version so the user can compare.
"$PO_BIN" mvo \
    --data             "$DATA_DIR/returns_daily.csv" \
    --returns \
    --periods-per-year 252 \
    --shrinkage        ledoit-wolf \
    | tee "$OUT_DIR/mvo_ledoit_wolf.console.txt"
