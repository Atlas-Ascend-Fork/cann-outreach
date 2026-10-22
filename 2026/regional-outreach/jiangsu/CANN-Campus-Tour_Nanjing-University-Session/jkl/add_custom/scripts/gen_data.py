# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# 测试数据生成脚本（直调通路）
# ============================================================================
#
# 生成内容（均写入当前工作目录 build/ 下）：
#   input/input_x.bin   float16，shape [DIM0, DIM1]，行优先连续
#   input/input_y.bin   float16，shape [DIM0, DIM1]，行优先连续
#   output/golden.bin   float32 参考结果（见 golden.py 的精度策略说明）
#
# 用法（在 build/ 目录下）：python3 ../scripts/gen_data.py
# ============================================================================

import os

import numpy as np

from case_config import DIM0, DIM1
from golden import compute_golden

# 固定随机种子，保证多次运行数据可复现
SEED = 2026
DTYPE = np.float16

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

rng = np.random.default_rng(SEED)
x = rng.standard_normal((DIM0, DIM1)).astype(DTYPE)
y = rng.standard_normal((DIM0, DIM1)).astype(DTYPE)

x.tofile("input/input_x.bin")
y.tofile("input/input_y.bin")

golden = compute_golden(x, y)  # float32
golden.tofile("output/golden.bin")

print(f"Generated test data: shape=[{DIM0}, {DIM1}], dtype={np.dtype(DTYPE)}, seed={SEED}")
print(f"  input/input_x.bin : {x.shape}, {x.dtype}, {x.nbytes} bytes")
print(f"  input/input_y.bin : {y.shape}, {y.dtype}, {y.nbytes} bytes")
print(f"  output/golden.bin : {golden.shape}, {golden.dtype}, {golden.nbytes} bytes")
