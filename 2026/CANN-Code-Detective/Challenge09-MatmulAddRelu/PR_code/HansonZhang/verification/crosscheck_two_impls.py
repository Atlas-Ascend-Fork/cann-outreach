"""两份**独立编写**的核函数复刻器交叉比对（bit-for-bit）。

`algosim.simulate_kernel`（第一实现，随 PR 一并提交，随补全代码一起演进）
与 `verification/second_emulator.py`（第二实现，只依据任务书与提交代码独立写出）
在同一组输入上必须给出**逐位相同**的输出。

两者的区别是刻意的：第二实现把 FP32 的 L0C 累加器按硬件的 rank-1 方式逐 k 累加，
第一实现用一次矩阵乘完成。因此「逐位相同」说明结论对 FP32 求和顺序不敏感，
而不是同一段代码被执行了两次。

用法：
    python3 verification/crosscheck_two_impls.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))      # algosim.py
sys.path.insert(0, HERE)                        # second_emulator.py

import second_emulator  # noqa: E402
from algosim import simulate_kernel  # noqa: E402


def main() -> int:
    inputs = os.path.join(HERE, "inputs.npz")
    if not os.path.exists(inputs):
        # 复用 second_emulator 的确定性输入生成逻辑
        second_emulator.main()
    data = np.load(inputs)
    a, b, x = data["a"], data["b"], data["x"]

    y1 = simulate_kernel(a, b, x)
    y2, _, _ = second_emulator.emulate(a, b, x)

    same = np.array_equal(y1, y2)
    diff = np.abs(y1.astype(np.float64) - y2.astype(np.float64))
    expected = np.maximum(a.astype(np.float32) @ b.astype(np.float32) + x, 0.0)
    tol = np.maximum(np.ones_like(expected), np.abs(expected)) / 256.0

    print("=" * 78)
    print("两份独立复刻器交叉比对")
    print("=" * 78)
    print("输入：%s  a%s %s  b%s %s  x%s %s"
          % (os.path.basename(inputs), a.shape, a.dtype, b.shape, b.dtype, x.shape, x.dtype))
    print("实现 1：algosim.simulate_kernel        （BLAS 矩阵乘累加）")
    print("实现 2：second_emulator.emulate        （rank-1 逐 k 累加，同硬件）")
    print("-" * 78)
    print("逐位相同 (np.array_equal)      : %s" % same)
    print("最大绝对差                    : %.6g" % diff.max())
    print("存在差异的元素数              : %d / %d" % (int((diff > 0).sum()), diff.size))
    print("实现 1 vs 官方参考：超差元素  : %d / %d"
          % (int((np.abs(y1 - expected) > tol).sum()), expected.size))
    print("实现 2 vs 官方参考：超差元素  : %d / %d"
          % (int((np.abs(y2 - expected) > tol).sum()), expected.size))
    print("-" * 78)
    ok = same and int((np.abs(y1 - expected) > tol).sum()) == 0 \
        and int((np.abs(y2 - expected) > tol).sum()) == 0
    print("交叉比对结果：%s" % ("一致" if ok else "不一致"))
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
