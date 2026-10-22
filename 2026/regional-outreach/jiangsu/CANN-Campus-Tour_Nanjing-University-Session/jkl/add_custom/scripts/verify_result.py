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
# 结果验证脚本（直调通路）
# ============================================================================
#
# 比较对象：
#   output.bin  —— 算子输出，float16（NPU 写入的原始字节）
#   golden.bin  —— 参考结果，float32（见 golden.py）
#
# 精度判据：把 float16 输出升到 float32 后与参考值比较，
#   |got - golden| <= ATOL + RTOL * |golden|
# float16 机器精度 eps = 2^-10 ≈ 9.77e-4，故 rtol/atol 取 1e-3，
# 即允许「一次 float16 舍入」的误差，不掩盖真实的计算错误。
#
# 用法（在 build/ 目录下）：python3 ../scripts/verify_result.py output/output.bin output/golden.bin
# ============================================================================

import sys

import numpy as np

from case_config import DIM0, DIM1

OUTPUT_DTYPE = np.float16   # 算子输出的数据类型
GOLDEN_DTYPE = np.float32   # golden 的数据类型
RTOL = 1e-3
ATOL = 1e-3
MAX_PRINT_MISMATCH = 5


def verify_result(output_path, golden_path):
    output = np.fromfile(output_path, dtype=OUTPUT_DTYPE).astype(np.float32)
    golden = np.fromfile(golden_path, dtype=GOLDEN_DTYPE)

    if output.shape != golden.shape:
        print(f"[FAIL] Shape mismatch: output {output.shape} vs golden {golden.shape}")
        return False

    expected = DIM0 * DIM1
    if golden.size != expected:
        print(f"[FAIL] Element count mismatch: golden {golden.size} vs expected {expected} "
              f"(case_config.py != op_host/add_custom.asc?)")
        return False

    diff = np.abs(output - golden)
    tol = ATOL + RTOL * np.abs(golden)
    bad = diff > tol
    bad_count = int(np.count_nonzero(bad))

    print(f"shape=[{DIM0}, {DIM1}], dtype=float16, elements={golden.size}")
    print(f"max |diff| = {diff.max():.6e}, mean |diff| = {diff.mean():.6e}, "
          f"rtol={RTOL}, atol={ATOL}")

    if bad_count == 0:
        print("[PASS] Verification PASSED")
        return True

    print(f"[FAIL] Verification FAILED: {bad_count} / {golden.size} "
          f"({bad_count / golden.size:.4%}) elements out of tolerance")
    idx = np.where(bad)[0][:MAX_PRINT_MISMATCH]
    for i in idx:
        print(f"  idx={int(i)}: got={output[i]:.6f}, golden={golden[i]:.6f}, "
              f"diff={diff[i]:.6e}, tol={tol[i]:.6e}")
    return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_result.py <output.bin> <golden.bin>")
        sys.exit(1)

    success = verify_result(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
