#!/bin/bash
# add_custom 编译脚本
# 用法：./build.sh            （同时编译 NPU 核函数 .o 与 CPU 调试程序）
#       ./build.sh npu        （仅编译 NPU 核函数 .o）
#       ./build.sh cpu        （仅编译 CPU 调试程序）

set -e

# CANN 包路径（按实际安装位置修改）
export DDK_PATH=${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}

ARCH=${ARCH:-ascend910b1}     # 按实际芯片型号修改

rm -rf build && mkdir -p build && cd build

build_npu() {
    $DDK_PATH/x86_64-linux/bin/cce -g -O2 -std=c++17 \
        --cce-aicore-arch=$ARCH \
        -I$DDK_PATH/x86_64-linux/include \
        -I.. \
        ../add_custom.cpp -o add_custom.o
    echo "[OK] NPU kernel: $(pwd)/add_custom.o (arch=$ARCH)"
}

build_cpu() {
    g++ -std=c++17 -O2 -DASCEND_CPU_DEBUG \
        -I$DDK_PATH/x86_64-linux/include \
        -I$DDK_PATH/x86_64-linux/include/aicpu \
        -I$DDK_PATH/x86_64-linux/include/profiling \
        -I.. \
        ../add_custom.cpp ../main_cpu.cpp \
        -L$DDK_PATH/x86_64-linux/lib64 \
        -lascendcl -lacl_cblas -lc_sec -lslog -lmmpa \
        -Wl,-rpath,$DDK_PATH/x86_64-linux/lib64 \
        -o add_custom_cpu
    echo "[OK] CPU debug binary: $(pwd)/add_custom_cpu"
    echo "     运行验证：./add_custom_cpu"
}

case "${1:-all}" in
    npu) build_npu ;;
    cpu) build_cpu ;;
    all) build_npu; build_cpu ;;
    *) echo "usage: $0 [npu|cpu|all]"; exit 1 ;;
esac
