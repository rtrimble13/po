#!/usr/bin/env bash
# 11_csv_inputs.sh — Market data via the two-file CSV format
# (assets.csv + covariance.csv) instead of JSON.
source "$(dirname "$0")/../_common.sh"
banner "11 — CSV market-data input"

# The CLI auto-discovers covariance.csv next to assets.csv.
"$PO_BIN" mvo \
    --data   "$DATA_DIR/assets.csv" \
    --params "$DATA_DIR/params_mvo.toml" \
    | tee "$OUT_DIR/result.console.txt"
