#!/usr/bin/env bash
# =============================================================================
# add_custom 算子构建脚本（参考入口）
#
# 重要：CANN 自定义算子的构建链路（build.sh / config / CMakePresets.json）
#       会随 CANN 版本与芯片型号变化。生产环境请优先使用 msOpGen 生成对应
#       版本的标准工程与构建脚本；本脚本用于快速演示 kernel / host 两侧的编译。
#
# 用法：bash build.sh
# 前置：已 source 目标 CANN 的 set_env.sh（使 ASCEND_HOME 生效）
# =============================================================================
set -euo pipefail

# ------------------------- 可修改配置区 -------------------------
# CANN 安装路径（如未 source set_env.sh，则在此显式指定）
ASCEND_HOME="${ASCEND_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
# 目标芯片（如 ascend910b3 / ascend910_93 / 310b3），需与 op_host 里 AddConfig 一致
SOC_VERSION="${SOC_VERSION:-ascend910b3}"
# 算子工程根目录
OP_PATH="$(cd "$(dirname "$0")" && pwd)"
# 输出目录
BUILD_DIR="${OP_PATH}/build"
# ----------------------------------------------------------------

mkdir -p "${BUILD_DIR}/kernel" "${BUILD_DIR}/host"

echo ">>> [1/2] 构建 kernel (.so)  SOC=${SOC_VERSION}"
cmake -S "${OP_PATH}/op_kernel" -B "${BUILD_DIR}/kernel" \
    -DCMAKE_BUILD_TYPE=Release \
    -DASCEND_HOME="${ASCEND_HOME}" \
    -DOP_KERNEL_OUTPUT_DIR="${BUILD_DIR}/kernel"
cmake --build "${BUILD_DIR}/kernel" -j

echo ">>> [2/2] 构建 host/tiling (.so)  SOC=${SOC_VERSION}"
cmake -S "${OP_PATH}/op_host" -B "${BUILD_DIR}/host" \
    -DCMAKE_BUILD_TYPE=Release \
    -DASCEND_HOME="${ASCEND_HOME}" \
    -DASCEND_CUSTOM_OP_PATH="${OP_PATH}" \
    -DOP_HOST_OUTPUT_DIR="${BUILD_DIR}/host"
cmake --build "${BUILD_DIR}/host" -j

echo ">>> 构建完成，产物位于：${BUILD_DIR}"
