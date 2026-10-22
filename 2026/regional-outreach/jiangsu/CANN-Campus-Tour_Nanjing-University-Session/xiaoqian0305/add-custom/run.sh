#!/bin/bash
# 一键运行 add_custom 算子（NPU 模式）
set -e

ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export ASCEND_HOME_PATH

mkdir -p build
cd build
if [ ! -f Makefile ]; then
    cmake .. -DCMAKE_BUILD_TYPE=Release
fi
make -j8
cd ..

# 生成输入数据
python3 gen_data.py

# 运行
./build/add_custom_host

# 精度比对
python3 - <<'EOF'
import numpy as np
out = np.fromfile("./output/output_z.bin", dtype=np.float16)
golden = np.fromfile("./output/golden.bin", dtype=np.float16)
if out.shape != golden.shape:
    print("FAIL: shape mismatch", out.shape, golden.shape)
    exit(1)
diff = np.abs(out.astype(np.float32) - golden.astype(np.float32))
print("max diff:", diff.max())
print("PASS" if diff.max() == 0 else "FAIL")
EOF
