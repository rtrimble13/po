#!/usr/bin/env bash
# 03_bl_basic.sh — Black-Litterman with two views, variance-confidence mode.
source "$(dirname "$0")/../_common.sh"
banner "03 — Black-Litterman"

"$PO_BIN" bl \
    --data        "$DATA_DIR/assets.json" \
    --params      "$DATA_DIR/params_bl.toml" \
    --show-model \
    --output      "$OUT_DIR/bl_result.json" \
    | tee "$OUT_DIR/bl_model.console.txt"
