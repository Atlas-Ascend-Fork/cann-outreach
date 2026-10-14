#!/bin/bash
set -euo pipefail

: "${ASCEND_TOOLKIT_HOME:?ASCEND_TOOLKIT_HOME is not set}"
source "$ASCEND_TOOLKIT_HOME/set_env.sh"
: "${ASCEND_HOME_PATH:?ASCEND_HOME_PATH is not set}"
export ASC_DIR="$ASCEND_HOME_PATH/aarch64-linux/tikcpp/ascendc_kernel_cmake/"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cmake -S "$script_dir" -B "$script_dir/build"
cmake --build "$script_dir/build" --parallel
"$script_dir/build/mul_test"
