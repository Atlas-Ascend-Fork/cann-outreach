#!/bin/bash
# add_custom 算子一键编译脚本
# 用法：./build.sh（在 add_custom 目录下执行）
set -e

# CANN 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

BUILD_DIR=build
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

# 生成 NPU 目标
cmake .. -DCMAKE_BUILD_TYPE=Release
# 编译算子（仅编译 kernel 侧可加：--target=add_custom_npu）
make -j8

echo "编译完成，产物位于 add_custom/build 下"
