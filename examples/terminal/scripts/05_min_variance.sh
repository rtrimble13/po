#!/usr/bin/env bash
# 05_min_variance.sh — Minimum-variance portfolio (ignores expected returns).
source "$(dirname "$0")/../_common.sh"
banner "05 — Minimum-variance portfolio"

"$PO_BIN" min-variance \
    --data   "$DATA_DIR/assets.json" \
    --params "$DATA_DIR/params_mvo.toml" \
    | tee "$OUT_DIR/min_variance.console.txt"

"$PO_BIN" min-variance \
    --data   "$DATA_DIR/assets.json" \
    --params "$DATA_DIR/params_mvo.toml" \
    --output "$OUT_DIR/min_variance.json" \
    --format json
