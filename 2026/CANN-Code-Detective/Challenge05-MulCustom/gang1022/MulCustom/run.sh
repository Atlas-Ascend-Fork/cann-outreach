#!/bin/bash
# 激活 CANN 环境；如环境变量已由镜像配置，可按实际情况调整。
source "$ASCEND_TOOLKIT_HOME/set_env.sh"
mkdir -p build
export ASC_DIR="$ASCEND_HOME_PATH/aarch64-linux/tikcpp/ascendc_kernel_cmake/"
cd build/ && \
cmake .. && \
make && \
./mul_test
