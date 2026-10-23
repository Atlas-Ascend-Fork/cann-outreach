#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -ne 0 && $# -ne 4 ]]; then
    echo 'Usage: bash run.sh [N4 N3 N2 N1]' >&2
    exit 2
fi
if [[ $# -eq 0 ]]; then set -- 1 2 3 17; fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
bash "$PROJECT_DIR/build.sh"
DATA_DIR="$PROJECT_DIR/data"
"$PYTHON_BIN" "$PROJECT_DIR/rsqrt_tools.py" generate --shape "$@" --out "$DATA_DIR"
if [[ -f "$DATA_DIR/output.bin" ]]; then rm -- "$DATA_DIR/output.bin"; fi
"$PROJECT_DIR/build/inplace_rsqrt_demo" "$@" \
    "$DATA_DIR/input.bin" "$DATA_DIR/output.bin" "${DEVICE_ID:-0}"
"$PYTHON_BIN" "$PROJECT_DIR/rsqrt_tools.py" verify --data "$DATA_DIR"
