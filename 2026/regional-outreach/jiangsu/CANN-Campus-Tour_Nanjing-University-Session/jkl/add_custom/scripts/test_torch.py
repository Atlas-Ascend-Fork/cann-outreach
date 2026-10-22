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
# PyTorch 通路测试脚本
# ============================================================================
#
# 覆盖点：
#   - 主用例（元素数为 TILE_LENGTH 整数倍，多核切分无尾块）
#   - 非对齐尾块（第 1 维为奇数，最后一个核/最后一个 tile 不是 32B 整数倍）
#   - 极小 shape（单核单 tile）
#   - 单行 shape
#   - 全零、正负抵消（结果精确为 0）
#   - 大 shape（多核多 tile）
#
# 用法（在 build/ 目录下）：python3 ../scripts/test_torch.py
# ============================================================================

import os
import sys

import torch
import torch_npu

from golden import compute_golden

SO_NAME = "libadd_custom_ops.so"   # 需与 CMakeLists.txt 中 target 名一致
OP_NAME = "add_custom"             # 需与 register.cpp 中注册名一致
DTYPE = torch.float16              # 算子支持的数据类型
RTOL = 1e-3                        # float16 machine eps ≈ 9.77e-4
ATOL = 1e-3


def run_case(name, x, y):
    """运行单个用例，返回 (name, shape, passed, max_diff, note)"""
    op_fn = getattr(torch.ops.npu, OP_NAME)
    out = op_fn(x.npu(), y.npu())
    golden = compute_golden(x, y)  # float32 参考

    got = out.cpu().float()
    diff = (got - golden).abs()
    tol = ATOL + RTOL * golden.abs()
    max_diff = diff.max().item()

    notes = []
    if out.dtype != DTYPE:
        notes.append(f"dtype={out.dtype}")
    if tuple(out.shape) != tuple(x.shape):
        notes.append(f"shape={tuple(out.shape)}")
    bad = int(torch.count_nonzero(diff > tol))
    if bad != 0:
        notes.append(f"{bad} elem out of tol")

    return name, tuple(x.shape), len(notes) == 0, max_diff, "; ".join(notes)


def main():
    # .so 位于工程根目录下的 build/，与当前工作目录无关
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    so_path = os.path.join(project_root, "build", SO_NAME)
    if not os.path.exists(so_path):
        print(f"ERROR: {so_path} not found. Run 'cd build && cmake .. && make' first.")
        sys.exit(1)
    torch.ops.load_library(so_path)

    torch.manual_seed(2026)
    cases = []

    # ---- 功能正确性 ----------------------------------------------------------
    # 主用例：512*1024 = 524288 元素，多核 + 整 tile 切分
    cases.append(("T1 random 2D main", torch.randn(512, 1024), torch.randn(512, 1024)))

    # 非对齐尾块：127*33 = 4191，最后一个核的最后一个 tile 不是 32B 整数倍
    cases.append(("T2 unaligned tail", torch.randn(127, 33), torch.randn(127, 33)))

    # 极小 shape：8*16 = 128 元素，单核单 tile
    cases.append(("T3 tiny 8x16", torch.randn(8, 16), torch.randn(8, 16)))

    # 单行
    cases.append(("T4 single row", torch.randn(1, 8192), torch.randn(1, 8192)))

    # 单元素
    cases.append(("T5 single elem", torch.randn(1, 1), torch.randn(1, 1)))

    # 全零
    cases.append(("T6 zeros", torch.zeros(512, 1024), torch.zeros(512, 1024)))

    # 正负抵消：x + (-x) 应精确为 0
    x = torch.randn(256, 512)
    cases.append(("T7 pos_neg", x, -x))

    # 大 shape：2048*2048 = 4M 元素，多核多 tile
    cases.append(("T8 large 2048x2048", torch.randn(2048, 2048), torch.randn(2048, 2048)))

    # 数值量级覆盖：大值
    cases.append(("T9 large values", torch.randn(64, 64) * 100.0, torch.randn(64, 64) * 100.0))

    # ---- 执行 ----------------------------------------------------------------
    results = []
    for name, x, y in cases:
        x = x.to(DTYPE).contiguous()
        y = y.to(DTYPE).contiguous()
        results.append(run_case(name, x, y))

    # ---- 汇总 ----------------------------------------------------------------
    total = len(results)
    passed = sum(1 for r in results if r[2])
    failed = total - passed

    print()
    print("=" * 78)
    print(f"PyTorch test results ({OP_NAME}, {DTYPE})")
    print("=" * 78)
    print(f"{'case':<24}{'shape':<18}{'status':<8}{'max diff':<14}note")
    print("-" * 78)
    for name, shape, ok, max_diff, note in results:
        print(f"{name:<24}{str(shape):<18}{'PASS' if ok else 'FAIL':<8}{max_diff:<14.6e}{note}")
    print("=" * 78)
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    print(f"Status: {'PASSED' if failed == 0 else 'FAILED'}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
