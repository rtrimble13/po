# shellcheck shell=bash
# Shared helpers for the terminal example scripts.
# Source this file from any example with:   source "$(dirname "$0")/../_common.sh"

set -euo pipefail

# Path layout — derived from the location of the *calling* script (BASH_SOURCE[1]),
# so every example resolves the same DATA_DIR / VAR_DIR regardless of cwd.
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
EX_ROOT="$(cd "$SCRIPTS_DIR/.." && pwd)"
PROJ_ROOT="$(cd "$EX_ROOT/../.." && pwd)"
DATA_DIR="$EX_ROOT/data"
VAR_DIR="$EX_ROOT/var"

mkdir -p "$VAR_DIR"

# Locate the `po` binary. Search order:
#   1. $PO_BIN env var (explicit override)
#   2. examples/terminal/bin/po  (staged by `cmake --build build --target po-examples`)
#   3. build/cli/po              (default CMake out-of-source build)
#   4. build/bin/po              (alternative layout)
#   5. `command -v po`           (installed system-wide)
locate_po() {
    if [[ -n "${PO_BIN:-}" ]]; then
        if [[ -x "$PO_BIN" ]]; then echo "$PO_BIN"; return; fi
        echo "PO_BIN is set to '$PO_BIN' but that path is not executable." >&2
        return 1
    fi
    local candidates=(
        "$EX_ROOT/bin/po"
        "$PROJ_ROOT/build/cli/po"
        "$PROJ_ROOT/build/bin/po"
    )
    for c in "${candidates[@]}"; do
        if [[ -x "$c" ]]; then echo "$c"; return; fi
    done
    if command -v po >/dev/null 2>&1; then
        command -v po
        return
    fi
    cat >&2 <<'EOF'
ERROR: could not find the `po` binary.

Build it with one of:
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --parallel --target portopt_cli

Then re-run, or stage a stable copy with:
    cmake --build build --target po-examples

…or set PO_BIN to an explicit path before running the script.
EOF
    return 1
}

PO_BIN="$(locate_po)"
export PO_BIN

# Per-script output directory: var/<calling-script-name-without-.sh>/
SCRIPT_STEM="$(basename "${BASH_SOURCE[1]}" .sh)"
OUT_DIR="$VAR_DIR/$SCRIPT_STEM"
mkdir -p "$OUT_DIR"

# Pretty header.
banner() {
    local title="$1"
    printf '\n==== %s ====\n' "$title"
    printf 'po binary : %s\n' "$PO_BIN"
    printf 'output dir: %s\n\n' "$OUT_DIR"
}
