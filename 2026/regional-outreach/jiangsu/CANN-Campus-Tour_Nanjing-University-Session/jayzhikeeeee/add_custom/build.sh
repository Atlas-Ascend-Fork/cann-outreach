#!/bin/bash
# add_custom 算子构建脚本
# 前提：已安装 CANN 包，且 export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
SOC_VERSION="${SOC_VERSION:-Ascend910B1}"
RUN_MODE="${RUN_MODE:-npu}"   # npu / cpu

if [ -z "$ASCEND_HOME_PATH" ]; then
    echo "错误：请先 source CANN 环境变量，如："
    echo "  source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    exit 1
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake .. -DSOC_VERSION=${SOC_VERSION} -DRUN_MODE=${RUN_MODE} \
         -DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
make -j$(nproc)

echo "=============================================="
echo "构建完成，产物位于 ${BUILD_DIR}"
echo "=============================================="
