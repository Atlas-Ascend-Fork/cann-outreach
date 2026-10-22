#!/usr/bin/env bash
# add_custom 算子编译运行脚本（需在装有 CANN Toolkit 的昇腾环境执行）
set -e

# 加载 CANN 环境变量（按实际安装路径调整）
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

# 方式一：使用 CANN 自带的自定义算子编译工具
# 将 kernel 编译为 NPU 二进制（run 包方式）
# ascendc add_custom.cpp --core_type=AI_CORE -o ./add_custom_npu

# 方式二：CMake 构建（kernel + host 联合编译）
mkdir -p build output
cd build
cmake .. -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest
make -j
cd ..

# 运行校验
./build/add_custom_host
