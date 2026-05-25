#!/usr/bin/env bash
# 09_report.sh — Jupyter diagnostic report.
#
# The `po report` subcommand drives a Jupyter notebook end-to-end and emits an
# executed notebook + an HTML report. It requires `jupyter` (and `nbconvert`)
# on PATH. If they're missing, this script prints how to install them and
# exits with a non-zero status that run_all.sh skips over.
source "$(dirname "$0")/../_common.sh"
banner "09 — Jupyter diagnostic report"

if ! command -v jupyter >/dev/null 2>&1; then
    cat <<'EOF'
Skipping: `jupyter` is not on PATH.
Install once with:
    pip install jupyter nbconvert matplotlib pandas
…then re-run this script.
EOF
    exit 0
fi

"$PO_BIN" report \
    --data       "$DATA_DIR/assets.json" \
    --params     "$DATA_DIR/params_bl.toml" \
    --method     both \
    --output-dir "$OUT_DIR"

echo
echo "Generated:"
ls -1 "$OUT_DIR"
