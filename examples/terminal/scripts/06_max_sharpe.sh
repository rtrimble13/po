#!/usr/bin/env bash
# 06_max_sharpe.sh — Maximum-Sharpe (tangency) portfolio.
source "$(dirname "$0")/../_common.sh"
banner "06 — Max-Sharpe (tangency) portfolio"

"$PO_BIN" max-sharpe \
    --data           "$DATA_DIR/assets.json" \
    --params         "$DATA_DIR/params_mvo.toml" \
    --risk-free-rate 0.04 \
    | tee "$OUT_DIR/max_sharpe.console.txt"

"$PO_BIN" max-sharpe \
    --data           "$DATA_DIR/assets.json" \
    --params         "$DATA_DIR/params_mvo.toml" \
    --risk-free-rate 0.04 \
    --output         "$OUT_DIR/max_sharpe.json" \
    --format         json
