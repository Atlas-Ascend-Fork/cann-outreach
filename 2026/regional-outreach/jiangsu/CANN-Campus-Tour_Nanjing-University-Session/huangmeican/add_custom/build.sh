#!/bin/bash
# 一键编译脚本。使用前请先 source CANN 环境：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 然后执行： ./build.sh
set -e

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

mkdir -p build output
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DASCEND_CANN_PACKAGE_PATH=${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}
make -j
cd ..

echo "build done -> build/add_custom_npu"
