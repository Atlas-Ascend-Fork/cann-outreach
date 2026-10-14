#!/bin/bash
set -e

# 定位脚本所在目录，确保任意路径下执行都能找到工程文件
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 加载 CANN 环境（支持 ASCEND_TOOLKIT_HOME / ASCEND_HOME_PATH，缺失时报错退出）
if [ -n "$ASCEND_TOOLKIT_HOME" ] && [ -f "$ASCEND_TOOLKIT_HOME/set_env.sh" ]; then
    source "$ASCEND_TOOLKIT_HOME/set_env.sh"
elif [ -n "$ASCEND_HOME_PATH" ] && [ -f "$ASCEND_HOME_PATH/set_env.sh" ]; then
    source "$ASCEND_HOME_PATH/set_env.sh"
else
    echo "Error: CANN environment not found (need ASCEND_TOOLKIT_HOME or ASCEND_HOME_PATH with set_env.sh)."
    exit 1
fi

mkdir -p build
export ASC_DIR="${ASCEND_HOME_PATH:-$ASCEND_TOOLKIT_HOME}/aarch64-linux/tikcpp/ascendc_kernel_cmake/"
cd build && cmake .. && make && ./mul_test
