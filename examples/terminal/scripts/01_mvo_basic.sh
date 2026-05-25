#!/usr/bin/env bash
# 01_mvo_basic.sh — Baseline MVO: one optimal portfolio with bounds and λ.
source "$(dirname "$0")/../_common.sh"
banner "01 — MVO basic"

"$PO_BIN" mvo \
    --data   "$DATA_DIR/assets.json" \
    --params "$DATA_DIR/params_mvo.toml" \
    --output "$OUT_DIR/result.json" \
    --format json

# Also print the human-readable console view so the user can eyeball it.
"$PO_BIN" mvo \
    --data   "$DATA_DIR/assets.json" \
    --params "$DATA_DIR/params_mvo.toml" \
    | tee "$OUT_DIR/result.console.txt"
