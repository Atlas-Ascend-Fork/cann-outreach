#!/usr/bin/env bash
# add_custom 算子构建与运行脚本
# 用法：
#   bash run.sh -v <soc_version>            # 默认 shape 64x64
#   bash run.sh -v Ascend910B4 32 128       # 指定 N2 N1
# 说明：需先 source CANN 环境变量，例如
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -e

SOC_VERSION=""
N2=64
N1=64
while getopts "v:" opt; do
    case "$opt" in
        v) SOC_VERSION="$OPTARG" ;;
        *) echo "unknown option"; exit 1 ;;
    esac
done
shift $((OPTIND - 1))
if [ $# -ge 2 ]; then N2=$1; N1=$2; fi
if [ -z "$SOC_VERSION" ]; then
    echo "usage: bash run.sh -v <soc_version> [N2 N1]"
    exit 1
fi

TOTAL=$((N2 * N1))
UNIT=$((8 * 2 * 256))  # BLOCK_DIM * BUFFER_NUM * TILE_LENGTH
if [ $((TOTAL % UNIT)) -ne 0 ]; then
    echo "shape [$N2, $N1] 不满足 tiling 约束：N2*N1 必须能被 $UNIT 整除"
    exit 1
fi

echo "==> 生成测试数据 [$N2, $N1] float16"
python3 gen_data.py "$N2" "$N1"

echo "==> 构建（SOC_VERSION=$SOC_VERSION）"
mkdir -p build
cmake -B build -DASCEND_CANN_PACKAGE_PATH="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}" -DSOC_VERSION="$SOC_VERSION"
cmake --build build -j8

echo "==> 运行单算子调用"
cd build/bin && ./main "$N2" "$N1"
cd ../..

echo "==> 结果校验"
python3 - <<'EOF'
import numpy as np
z = np.fromfile("build/bin/z_out.bin", dtype=np.float16)
g = np.fromfile("z_golden.bin", dtype=np.float16)
if z.shape != g.shape:
    raise SystemExit("size mismatch")
diff = np.abs(z.astype(np.float32) - g.astype(np.float32))
print(f"max diff = {diff.max()}, result {'PASS' if diff.max() == 0 else 'FAIL'}")
EOF
