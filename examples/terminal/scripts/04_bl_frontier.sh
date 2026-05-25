#!/usr/bin/env bash
# 04_bl_frontier.sh — Black-Litterman efficient frontier on the posterior.
source "$(dirname "$0")/../_common.sh"
banner "04 — Black-Litterman frontier"

"$PO_BIN" bl-frontier \
    --data   "$DATA_DIR/assets.json" \
    --params "$DATA_DIR/params_bl.toml" \
    --output "$OUT_DIR/bl_frontier.csv"

head -n 6 "$OUT_DIR/bl_frontier.csv"
echo "..."
wc -l "$OUT_DIR/bl_frontier.csv"
