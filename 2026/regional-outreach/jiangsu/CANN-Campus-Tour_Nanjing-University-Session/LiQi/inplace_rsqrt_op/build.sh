#!/bin/bash
# Build script for InplaceRsqrt AscendC operator
set -e

# Source CANN environment
if [ -f "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh" ]; then
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
fi

# Detect chip
CHIP=${ASCEND_AITUBE_CHIP:-"ascend910"}

echo "=== Building InplaceRsqrt Operator ==="
echo "Chip: ${CHIP}"
echo "CANN_HOME: ${CANN_HOME}"

mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DASCEND_AITUBE_CHIP=${CHIP} \
    -DCANN_HOME=${CANN_HOME:-/usr/local/Ascend/ascend-toolkit/latest}

make -j$(nproc)

echo ""
echo "=== Build Complete ==="
echo "Output: $(pwd)/libinplace_rsqrt_op.so"
echo ""
echo "To install to CANN op library:"
echo "  cp libinplace_rsqrt_op.so ${CANN_HOME}/opp/built-in/op_impl/ai_core/tbe/op_kernel/ops_kernel/math/"
