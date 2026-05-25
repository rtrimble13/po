#!/usr/bin/env bash
# 08_target_return.sh — Portfolio with expected return ≈ target.
source "$(dirname "$0")/../_common.sh"
banner "08 — Target-return portfolio (10%)"

"$PO_BIN" target-return \
    --data    "$DATA_DIR/assets.json" \
    --params  "$DATA_DIR/params_mvo.toml" \
    --target  0.10 \
    | tee "$OUT_DIR/target_return.console.txt"
