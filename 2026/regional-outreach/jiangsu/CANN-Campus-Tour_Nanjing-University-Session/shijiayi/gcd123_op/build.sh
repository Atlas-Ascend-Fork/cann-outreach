#!/bin/bash
# Gcd123 构建脚本
#
# 说明：
#  - 推荐流程是先用 msOpGen 生成工程骨架（会自动生成 CMakeLists / CMakePresets /
#    完整的 build.sh），然后把本仓库的 op_host/gcd123.cpp、op_kernel/* 覆盖进去再编译。
#  - 若手工编译，请确保已加载 CANN 环境变量，并按需指定 compute unit。
#
set -e

# 1) 加载 CANN 环境变量（路径按实际安装位置调整；每次新开终端都需重新执行）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2) 生成构建配置并编译
#    ASCEND_COMPUTE_UNIT 取目标芯片 soc_version，例如 ascend910b / ascend310p3 等。
#    若使用 msOpGen 生成的 CMakePresets.json，可改为：
#        cmake --preset <preset_name> && cmake --build build
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DASCEND_COMPUTE_UNIT=${ASCEND_COMPUTE_UNIT:-ascend910b}
cmake --build build -j

echo "==== 构建完成 ===="
echo "算子安装包(.run)通常位于 build/ 或 build_out/ 目录，执行安装："
echo "  ./build_out/custom_opp_*.run   # 安装到 \$ASCEND_OPP_PATH/vendors/customize/"
