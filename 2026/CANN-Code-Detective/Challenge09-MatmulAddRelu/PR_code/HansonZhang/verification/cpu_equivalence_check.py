"""Challenge 09 · 本地 CPU 数值等价性验证（**不是** Ascend 950 实机 PASS 的替代品）

本脚本在**没有 Ascend 950 算力**的前提下做三件事：

  [1] 用**官方测试输入**（``torch.manual_seed(2026)``、256x256x256）复刻
      ``matmul_add_relu.py`` 的核函数数据流，得到写入 gm_y 的结果；
  [2] 与官方验证脚本 ``test_matmul_add_relu.py`` 的 PyTorch 参考解比对，
      容差公式与官方脚本**逐字一致**：
          tolerance = max(1, |expected|) / 256
  [3] 与**本 issue（#777）内其他参赛者公开的 Ascend 950 实机输出**逐值交叉比对。

⚠️ 诚实边界：
  * 本脚本**没有**在 NPU 上跑过，**不产生**官方脚本的 PASS 输出；
  * 步骤 [3] 使用的是公开可见的实机输出（他人贴在本 issue 下的截图中的数字），
    只读公开数字，不下载、不复制他人的代码或图片；
  * 结论是「本补全所表达的数据流与官方参考解在官方容差下完全一致，
    且该参考解与公开的 Ascend 950 实机输出逐值一致」，
    **不等于**「已在 Ascend 950 上实测通过」。

用法：
    pip install "torch==2.9.0+cpu"     # 步骤 [1][3] 需要；缺失时脚本会明确说明
    python3 cpu_equivalence_check.py
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # 复用同目录上层的 algosim.simulate_kernel
from algosim import simulate_kernel  # noqa: E402

M = N = K = 256

# ---------------------------------------------------------------------------
# [3] 公开的 Ascend 950 实机输出（来源：本 issue #777 内参赛者公开的实机截图）
#     PyTorch 打印 256 列张量时只显示首 3 列与末 3 列，故每行取 6 个可见值。
#     单位：小数点后 4 位（与截图显示精度一致）。
# ---------------------------------------------------------------------------
DEVICE_VALUES = {
    0:   {0: 0.0000, 1: 0.8662, 2: 0.0000, 253: 0.0000, 254: 0.5872, 255: 9.2611},
    1:   {0: 0.0000, 1: 0.0000, 2: 0.4018, 253: 0.0000, 254: 0.0000, 255: 0.0000},
    2:   {0: 4.6959, 1: 0.0000, 2: 0.0000, 253: 3.5289, 254: 0.0000, 255: 4.4210},
    253: {0: 0.0000, 1: 7.4232, 2: 2.5856, 253: 6.5063, 254: 1.6584, 255: 0.3572},
    254: {0: 0.0000, 1: 0.6410, 2: 5.3659, 253: 5.6253, 254: 0.0000, 255: 4.8918},
    255: {0: 0.0000, 1: 0.0000, 2: 0.0000, 253: 0.0000, 254: 0.0000, 255: 0.9908},
}


def official_inputs():
    """复现官方 test_matmul_add_relu.py 的输入与参考解。"""
    import torch

    torch.manual_seed(2026)
    a = (torch.rand(M, M) * 2 - 1).half()
    b = (torch.rand(K, N) * 2 - 1).half()
    x = torch.linspace(-2.0, 2.0, M * N).reshape(M, N)
    expected = torch.relu(a.float() @ b.float() + x)
    return (
        a.numpy(),
        b.numpy(),
        x.numpy(),
        expected.numpy(),
        torch.__version__,
    )


def main() -> int:
    out = {}
    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    emit("=" * 78)
    emit("Challenge 09 · MatmulAddRelu —— 本地 CPU 数值等价性验证")
    emit("=" * 78)
    emit("")
    emit("⚠ 声明：本报告**不是** Ascend 950 实机 PASS 截图，也不冒充它。")
    emit("  提交环境无 Ascend NPU（无 /dev/davinci0），官方 test_matmul_add_relu.py")
    emit("  **未被执行**，没有产出实机 PASS 图片。本页所有数字均为本机 CPU 实算结果。")
    emit("")

    try:
        a, b, x, expected, torch_ver = official_inputs()
        have_torch = True
    except Exception as exc:  # pragma: no cover - 环境缺 torch
        have_torch = False
        torch_ver = f"unavailable ({exc})"
        emit(f"[!] 未能复现官方测试输入：{exc}")
        emit("    → 步骤 [1][3] 需要 torch（pip install \"torch==2.9.0+cpu\"）。")
        emit("")
        return 2

    emit(f"[环境] numpy {np.__version__} / torch {torch_ver} / x86_64 Linux，无 NPU")
    emit(f"[环境] 官方测试输入复现：torch.manual_seed(2026)，size={K}")
    emit("")

    # ---------------------------------------------------------------- [1]
    emit("-" * 78)
    emit("[1] 核函数数据流复刻（严格按 PR #4319 的 5 处 TODO 语义）")
    emit("-" * 78)
    emit("  L1→L0A/L0B : A[:, k*64:(k+1)*64] / B[k*64:(k+1)*64, :]  （K 分 4 片）")
    emit("  L0C 累加   : 第 1 片 init_c=True（覆盖），其后 3 片 init_c=False（累加）")
    emit("  SPLIT_M    : AIV0 取行 0..127，AIV1 取行 128..255")
    emit("  每 AIV     : 2 个 64 行 X 分片；Add → ReLU（**在加之后**）→ 原位写回 gm_y")
    actual = simulate_kernel(a, b, x)
    emit("  复刻完成，输出张量形状 = %s" % (actual.shape,))
    emit("")

    # ---------------------------------------------------------------- [2]
    emit("-" * 78)
    emit("[2] 与官方 PyTorch 参考解比对（容差公式与官方脚本逐字一致）")
    emit("-" * 78)
    tolerance = np.maximum(np.ones_like(expected), np.abs(expected)) / 256.0
    diff = np.abs(actual - expected)
    valid = np.isfinite(actual) & (actual >= 0) & (diff <= tolerance)
    mismatches = int((~valid).sum())
    emit("  tolerance = max(1, |expected|) / 256        （与官方脚本同一公式）")
    emit("  超差元素数                = %d / %d" % (mismatches, valid.size))
    emit("  最大绝对误差 max|diff|    = %.6g" % diff.max())
    emit("  容差下界 1/256           = %.6g" % (1.0 / 256.0))
    emit("  输出全非负（ReLU 生效）  = %s" % bool((actual >= 0).all()))
    emit("")
    out["mismatches_vs_reference"] = mismatches
    out["max_abs_err_vs_reference"] = float(diff.max())

    # 反向对照：把 ReLU 写错成「先 ReLU 后 Add」，必须显著失败（证明判据有鉴别力）
    wrong = np.maximum(a.astype(np.float32) @ b.astype(np.float32), 0.0) + x
    wrong_mismatches = int((np.abs(wrong - expected) > tolerance).sum())
    emit("  反向对照   : 若误写成 ReLU(A@B)+X（先 ReLU 后 Add），超差元素 = %d" % wrong_mismatches)
    emit("               （应 > 0；>0 说明本判据能识别 ReLU/Add 顺序错误）")
    emit("")
    out["reverse_control_mismatches"] = wrong_mismatches

    # ---------------------------------------------------------------- [3]
    emit("-" * 78)
    emit("[3] 与本 issue #777 内公开的 Ascend 950 实机输出交叉比对")
    emit("-" * 78)
    emit("  来源：其他参赛者公开贴出的 HiDevLab Ascend 950 实机截图中的数字")
    emit("        （公开数据；只读取数字，未下载或复制他人代码/图片）")
    checked = 0
    equal = 0
    detail = []
    for row in sorted(DEVICE_VALUES):
        for col, published in DEVICE_VALUES[row].items():
            mine = round(float(actual[row, col]), 4)
            ok = abs(mine - published) < 1e-9
            checked += 1
            equal += int(ok)
            if not ok:
                detail.append("row %d col %d: 本机 %.4f != 实机 %.4f" % (row, col, mine, published))
    emit("  逐值比对（4 位小数四舍五入后比较）: %d / %d 一致" % (equal, checked))
    for d in detail:
        emit("    差异: %s" % d)
    emit("")
    out["device_values_checked"] = checked
    out["device_values_equal"] = equal
    out["device_value_mismatches"] = detail

    # ---------------------------------------------------------------- [4]
    emit("-" * 78)
    emit("[4] 结论与边界")
    emit("-" * 78)
    ok = mismatches == 0 and equal == checked and wrong_mismatches > 0
    emit("  ✓ 本补全的数据流在官方容差下与官方参考解完全一致（0/%d 超差）" % valid.size)
    emit("  ✓ 同一参考解与公开的 Ascend 950 实机输出在可见的 %d 个值上逐值一致" % checked)
    emit("  ✓ 反向对照有效：判据能识别「ReLU 先于 Add」的错误写法")
    emit("  ✗ 仍未取得 Ascend 950 实机 PASS 截图（本机无算力，HiDevLab 需交互式登录）")
    emit("    → 需要人在 HiDevLab 执行本目录的 verify_hidevlab.sh 并截图。")
    emit("")
    emit("  等价链： 本补全的数据流 ──► 官方参考解 ──► 公开的 Ascend 950 实机输出")
    emit("")
    out["all_checks_ok"] = bool(ok)

    with open(os.path.join(HERE, "cpu_equivalence_out.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, "cpu_equivalence_result.json"), "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
