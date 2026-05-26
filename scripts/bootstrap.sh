#!/usr/bin/env bash
# bootstrap.sh — Configure the project, build the C++ side, and create a
# Python venv with all extras installed in editable mode. Idempotent: rerun
# any time pyproject.toml changes or you want to refresh the build.
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$PROJ_ROOT/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

echo "==> Configuring CMake ($BUILD_TYPE) in $BUILD_DIR"
cmake -S "$PROJ_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

echo "==> Building C++ targets + python-venv (parallel jobs: $JOBS)"
cmake --build "$BUILD_DIR" --target python-venv --parallel "$JOBS"

cat <<EOF

Bootstrap complete.

To activate the Python environment:

    source "$PROJ_ROOT/.venv/bin/activate"

Then:

    python -c "import portopt; print(portopt.__version__)"
    po --help

To rebuild after editing C++ or Python sources, rerun this script (or
'cmake --build $BUILD_DIR --target python-venv').
EOF
