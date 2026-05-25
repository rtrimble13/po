#!/usr/bin/env bash
# 15_long_short.sh — Dollar-neutral long/short portfolio (budget = 0).
source "$(dirname "$0")/../_common.sh"
banner "15 — Dollar-neutral long/short"

"$PO_BIN" mvo \
    --data      "$DATA_DIR/assets.json" \
    --params    "$DATA_DIR/params_mvo_long_short.toml" \
    --show-zero \
    --explain \
    | tee "$OUT_DIR/result.console.txt"
