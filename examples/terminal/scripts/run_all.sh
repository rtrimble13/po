#!/usr/bin/env bash
# run_all.sh — Run every numbered example in order. Used as a smoke test.
set -u  # do NOT set -e: keep going past individual failures so we report all.

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPTS_DIR"

fail=0
for s in $(ls [0-9][0-9]_*.sh | sort); do
    echo
    echo "############################################################"
    echo "# Running $s"
    echo "############################################################"
    if bash "$s"; then
        echo "## OK: $s"
    else
        rc=$?
        echo "## FAILED ($rc): $s"
        fail=$((fail + 1))
    fi
done

echo
echo "============================================================"
if [[ $fail -eq 0 ]]; then
    echo "All examples completed successfully."
else
    echo "$fail example(s) failed."
fi
echo "============================================================"
exit $fail
