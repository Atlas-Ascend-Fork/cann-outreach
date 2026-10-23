#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo 'Load the CANN 9.0.0 set_env.sh before running this script.' >&2
    exit 2
fi
cmake -S "$PROJECT_DIR" -B "$PROJECT_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release -DNPU_ARCH="${NPU_ARCH:-dav-2201}" "$@"
cmake --build "$PROJECT_DIR/build" --parallel "${BUILD_JOBS:-4}"
