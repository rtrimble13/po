#!/usr/bin/env bash
# 07_target_vol.sh — Portfolio whose realised volatility ≈ target.
source "$(dirname "$0")/../_common.sh"
banner "07 — Target-volatility portfolio (15%)"

"$PO_BIN" target-vol \
    --data    "$DATA_DIR/assets.json" \
    --params  "$DATA_DIR/params_mvo.toml" \
    --target  0.15 \
    | tee "$OUT_DIR/target_vol.console.txt"
