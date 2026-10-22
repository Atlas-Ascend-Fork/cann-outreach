#!/bin/bash
# add_custom 算子 UT 测试执行脚本（CPU 仿真，不依赖 NPU 设备）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
CLEAN_BUILD=true

echo "========================================"
echo "add_custom 算子 UT 测试"
echo "========================================"

if [ -z "$ASCEND_HOME_PATH" ]; then
    export ASCEND_HOME_PATH=/usr/local/Ascend/cann
fi

# tikicpulib（CPU 仿真）运行依赖的库路径：
#   libpem_davinci.so 等仿真库在 9.0 x86 布局下位于 x86_64-linux/simulator/dav_2201（910B）/dav_1001（310）
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/compiler/lib64:${ASCEND_HOME_PATH}/tools/tikicpulib/lib:${ASCEND_HOME_PATH}/tools/tikicpulib/lib/Ascend910B1:${ASCEND_HOME_PATH}/x86_64-linux/simulator/dav_2201/lib:${ASCEND_HOME_PATH}/x86_64-linux/simulator/dav_1001/lib:${ASCEND_HOME_PATH}/aarch64-linux/simulator/Ascend910B1/lib:${ASCEND_HOME_PATH}/x86_64-linux/lib64:${ASCEND_HOME_PATH}/aarch64-linux/lib64:${LD_LIBRARY_PATH}

if [ "$CLEAN_BUILD" = true ]; then
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake ..
make -j$(nproc)

echo "========================================"
echo "执行 UT 测试"
echo "========================================"

FAILED_TESTS=()
PASSED_TESTS=()

cd "${BUILD_DIR}/op_host"
if ./add_custom_op_host_ut; then
    PASSED_TESTS+=("op_host")
else
    FAILED_TESTS+=("op_host")
fi

cd "${BUILD_DIR}/op_kernel"
if ./add_custom_op_kernel_ut; then
    PASSED_TESTS+=("op_kernel")
else
    FAILED_TESTS+=("op_kernel")
fi

if [ -d "${BUILD_DIR}/../../../op_api" ]; then
    cd "${BUILD_DIR}/op_api"
    if ./add_custom_op_api_ut; then
        PASSED_TESTS+=("op_api")
    else
        FAILED_TESTS+=("op_api")
    fi
fi

echo "========================================"
echo "测试结果汇总"
echo "========================================"

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo "失败的测试: ${FAILED_TESTS[@]}"
    exit 1
else
    echo "所有测试通过: ${PASSED_TESTS[@]}"
    exit 0
fi
