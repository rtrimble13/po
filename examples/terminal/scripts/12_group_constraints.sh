#!/usr/bin/env bash
# 12_group_constraints.sh — Sector group caps (Tech ≤ 50 %, Energy ≤ 15 %).
source "$(dirname "$0")/../_common.sh"
banner "12 — Group (sector) constraints"

"$PO_BIN" mvo \
    --data    "$DATA_DIR/assets.json" \
    --params  "$DATA_DIR/params_mvo_groups.toml" \
    --explain \
    | tee "$OUT_DIR/result.console.txt"
