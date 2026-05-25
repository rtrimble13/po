#!/usr/bin/env bash
# 02_mvo_frontier.sh — MVO efficient frontier (return vs risk sweep).
source "$(dirname "$0")/../_common.sh"
banner "02 — MVO efficient frontier"

"$PO_BIN" frontier \
    --data   "$DATA_DIR/assets.json" \
    --params "$DATA_DIR/params_mvo.toml" \
    --output "$OUT_DIR/frontier.csv"

# Show the first few rows so the user can see the (return, vol, sharpe) sweep.
head -n 6 "$OUT_DIR/frontier.csv"
echo "..."
wc -l "$OUT_DIR/frontier.csv"
