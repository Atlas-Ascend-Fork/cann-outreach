#!/usr/bin/env bash
# =============================================================================
# build.sh — 构建 add_custom 算子内核与主机端测试
#
# 前置条件（Linux + CANN，Windows 上无法构建）：
#   1. 已安装 CANN（建议 7.0+），例如 /usr/local/Ascend/ascend-toolkit/latest
#   2. 已通过 asennv / source set_env.bash 加载环境变量，或手动设置 CANN_PATH
#
# 用法：
#   ./build.sh            # 默认编译
#   SOC=Ascend910B ./build.sh
# =============================================================================
set -euo pipefail

# --- 环境变量 ----------------------------------------------------------------
CANN_PATH="${CANN_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
SOC="${SOC:-Ascend910B}"          # 目标芯片，按实际设备修改（Ascend310P / Ascend910B / Atlas 800 等）
BUILD_DIR="build"

export PATH="${CANN_PATH}/tools/ccec_compiler/bin:${PATH}"
export LD_LIBRARY_PATH="${CANN_PATH}/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "${BUILD_DIR}"

# --- 1. 编译 Ascend C 内核（.cpp -> .o 自定义内核） ----------------------------
echo "[build] ccec_compiler: src/add_custom.cpp -> ${BUILD_DIR}/add_custom.o"
ccec_compiler -x AscendC --soc-version="${SOC}" \
    src/add_custom.cpp \
    -o "${BUILD_DIR}/add_custom.o"

# --- 2. 编译主机端测试 ---------------------------------------------------------
echo "[build] g++: test/main.cpp -> ${BUILD_DIR}/add_custom_test"
g++ -std=c++17 -O2 \
    -I"${CANN_PATH}/include" \
    test/main.cpp \
    -o "${BUILD_DIR}/add_custom_test" \
    -L"${CANN_PATH}/lib64" -lacl_rt -Wl,-rpath,"${CANN_PATH}/lib64"

echo "[build] 完成。运行测试："
echo "  cd ${BUILD_DIR} && ./add_custom_test"
