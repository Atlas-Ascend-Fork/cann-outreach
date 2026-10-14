#!/bin/bash
set -e

# 激活 CANN 环境（CANNLab 中一般为 /home/developer/Ascend/cann-9.0.0 或 $ASCEND_TOOLKIT_HOME）
if [ -z "$ASCEND_TOOLKIT_HOME" ]; then
    for d in /home/developer/Ascend/ascend-toolkit/latest /usr/local/Ascend/ascend-toolkit/latest; do
        if [ -d "$d" ]; then export ASCEND_TOOLKIT_HOME="$d"; break; fi
    done
fi
source "$ASCEND_TOOLKIT_HOME/set_env.sh"

rm -rf build
mkdir -p build && cd build
cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 .. && make -j
cd ..

# 运行验证：输出 Output 与 Golden 均为 2.76 时打印 [Success] Case accuracy is verification passed.
./build/mul_test
