#!/usr/bin/env bash
# 09_report.sh — Jupyter diagnostic report.
#
# The `po report` subcommand drives a Jupyter notebook end-to-end and emits an
# executed notebook + an HTML report. It needs `nbconvert` installed in a
# Python environment that also has `portopt` importable.
#
# Preferred setup (one command from the repo root):
#     cmake --build build --target python-venv
#     source .venv/bin/activate
#
# If the project venv exists but isn't activated, this script picks it up via
# PORTOPT_PYTHON so `po report` calls the right interpreter.
source "$(dirname "$0")/../_common.sh"
banner "09 — Jupyter diagnostic report"

# Prefer the project venv unless the caller has explicitly set PORTOPT_PYTHON.
# (We deliberately ignore any ambient $VIRTUAL_ENV — e.g. a conda base env —
# because the example scripts want the .venv produced by `python-venv`.)
if [[ -z "${PORTOPT_PYTHON:-}" && -x "$PROJ_ROOT/.venv/bin/python" ]]; then
    export PORTOPT_PYTHON="$PROJ_ROOT/.venv/bin/python"
    echo "Using project venv Python: $PORTOPT_PYTHON"
fi

probe_python="${PORTOPT_PYTHON:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}"
probe_python="${probe_python:-python}"

if ! "$probe_python" -m nbconvert --version >/dev/null 2>&1; then
    cat <<EOF
Skipping: \`nbconvert\` is not available via $probe_python.
Install once with:
    cmake --build build --target python-venv
or, into an existing environment:
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
