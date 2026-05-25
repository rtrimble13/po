#!/usr/bin/env bash
# 13_turnover_penalty.sh — Rebalance from an equal-weight current book with
# an L2 turnover penalty.
source "$(dirname "$0")/../_common.sh"
banner "13 — Turnover penalty (rebalance)"

# Run with the penalty from the params file.
"$PO_BIN" mvo \
    --data   "$DATA_DIR/assets.json" \
    --params "$DATA_DIR/params_mvo_turnover.toml" \
    | tee "$OUT_DIR/with_penalty.console.txt"

# Same data, but the CLI flag drops the penalty to zero so you can see how
# far weights move when nothing penalises trading.
"$PO_BIN" mvo \
    --data             "$DATA_DIR/assets.json" \
    --params           "$DATA_DIR/params_mvo_turnover.toml" \
    --turnover-penalty 0.0 \
    | tee "$OUT_DIR/no_penalty.console.txt"
