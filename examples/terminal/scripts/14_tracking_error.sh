#!/usr/bin/env bash
# 14_tracking_error.sh — Constrain MVO so ex-ante tracking error vs the
# benchmark in assets.json stays under 5%.
source "$(dirname "$0")/../_common.sh"
banner "14 — Tracking-error budget vs benchmark"

"$PO_BIN" mvo \
    --data    "$DATA_DIR/assets.json" \
    --params  "$DATA_DIR/params_mvo_tracking.toml" \
    --explain \
    | tee "$OUT_DIR/result.console.txt"
