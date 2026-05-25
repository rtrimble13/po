#!/usr/bin/env bash
# 16_output_formats.sh — Demonstrate the three output formats: console, JSON,
# and CSV (with notional dollar amounts via --total-capital).
source "$(dirname "$0")/../_common.sh"
banner "16 — Output formats + notional capital"

# Console (default).
"$PO_BIN" mvo \
    --data           "$DATA_DIR/assets.json" \
    --params         "$DATA_DIR/params_mvo.toml" \
    --total-capital  10000000 \
    | tee "$OUT_DIR/result.console.txt"

# JSON.
"$PO_BIN" mvo \
    --data    "$DATA_DIR/assets.json" \
    --params  "$DATA_DIR/params_mvo.toml" \
    --output  "$OUT_DIR/result.json" \
    --format  json
echo "Wrote $OUT_DIR/result.json"

# CSV.
"$PO_BIN" mvo \
    --data    "$DATA_DIR/assets.json" \
    --params  "$DATA_DIR/params_mvo.toml" \
    --output  "$OUT_DIR/result.csv" \
    --format  csv
echo "Wrote $OUT_DIR/result.csv"
