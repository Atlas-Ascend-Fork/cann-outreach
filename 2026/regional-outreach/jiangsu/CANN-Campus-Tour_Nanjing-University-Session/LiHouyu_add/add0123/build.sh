#!/bin/bash
# add0123 算子工程编译脚本
# 用法: bash build.sh [SOC_VERSION] [CANN路径]
#   SOC_VERSION: 昇腾处理器型号, 默认 Ascend910B1
#   CANN路径:    默认取环境变量 ASCEND_CANN_PACKAGE_PATH, 否则 /usr/local/Ascend/ascend-toolkit/latest
set -e
script_dir=$(cd "$(dirname "$0")" && pwd)

SOC_VERSION=${1:-${SOC_VERSION:-Ascend910B1}}
ASCEND_CANN_PACKAGE_PATH=${2:-${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}

if [ -f "${ASCEND_CANN_PACKAGE_PATH}/bin/setenv.bash" ]; then
    source "${ASCEND_CANN_PACKAGE_PATH}/bin/setenv.bash"
fi

mkdir -p build_out
cd build_out
cmake .. -DASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH}" -DSOC_VERSION="${SOC_VERSION}"
cmake --build . -j16

echo "=============================================="
echo "构建完成, 产物位于 ${script_dir}/build_out/"
ls -l libcust_opsproto_rt2.0.so libcust_opmaster_rt2.0.so libadd0123_kernels.so 2>/dev/null || true
