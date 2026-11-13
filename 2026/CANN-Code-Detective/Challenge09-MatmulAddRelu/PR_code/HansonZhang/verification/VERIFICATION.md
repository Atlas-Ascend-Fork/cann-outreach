# Challenge 09 · 自验证材料说明（本地 CPU 数值等价性验证）

> ## ⚠️ 边界声明（请先读）
>
> **本目录的材料不是 Ascend 950 实机 PASS 截图，也不冒充它。**
>
> 提交者环境无 Ascend NPU（无 `/dev/davinci0` 设备节点），官方验证脚本
> `test_matmul_add_relu.py` **未被执行**，因此**没有**实机 PASS 图片。
>
> 本目录中每一个数字都来自**本机 CPU 上实际执行过的脚本**，输出文件即其原样留存。
> 未使用任何占位图，未下载或复制他人的代码与验证图片。
> 凡引用他人公开数据之处，均只读取**公开可见的数字**并注明出处。

---

## 一、这次验证证明了什么

在没有算力的前提下，用**可复现的证据链**回答「这份补全在实机上会算出什么」：

```
本补全的核函数数据流  ──►  官方 PyTorch 参考解  ──►  公开的 Ascend 950 实机输出
   （本地 CPU 复刻）          （seed 2026 复现）        （本 issue 内公开截图中的数字）
```

| 环节 | 结论 | 证据 |
|:--|:--|:--|
| 复刻 vs 官方参考解 | **0 / 65536** 元素超出官方容差；最大绝对误差 1.34e-05（容差下界 3.9e-03） | `cpu_equivalence_out.txt` |
| 两份独立复刻器交叉比对 | **逐位相同**（0 / 65536 元素有差异） | `crosscheck_out.txt` |
| 官方参考解 vs 公开实机输出 | 公开可见的 **36 / 36** 个值逐值一致（4 位小数） | `cpu_equivalence_out.txt` 第 [3] 节 |
| 判据鉴别力（反向对照） | 故意写成「先 ReLU 后 Add」→ 超差 35006 / 65536 | `cpu_equivalence_out.txt` 第 [2] 节 |

**等价链的关键一环**：官方验证脚本的输入是确定性的
（`torch.manual_seed(2026)`，`size=256`），参考解 `relu(a.float() @ b.float() + x)`
只依赖 CPU 上可复现的 `torch` 运算。本机复现出的参考解，与**本 issue 内其他参赛者
公开贴出的 Ascend 950 实机输出**在全部 36 个可见值上逐值一致 —— 说明该参考解就是
真机上的结果；再结合「本补全的数据流 ≡ 该参考解」，即说明本补全在实机上应产生同样
的结果。

⚠️ 这**不**等于「已在 Ascend 950 上实测通过」。硬件侧的编译产物、DSL 版本差异、
SPLIT_M 的实际下发行为等仍只有实机能覆盖。

---

## 二、文件清单

| 文件 | 内容 |
|:--|:--|
| `cpu_equivalence_check.py` | 主验证脚本：官方输入复现 → 核函数数据流复刻 → 官方容差比对 → 公开实机输出交叉比对 |
| `cpu_equivalence_out.txt` | 上述脚本的**原样输出** |
| `cpu_equivalence_result.json` | 上述结果的机器可读形式 |
| `second_emulator.py` | **第二份独立复刻器**（只依据任务书与提交代码写出，rank-1 逐 k 累加，同硬件） |
| `crosscheck_two_impls.py` | 驱动两份复刻器逐位交叉比对 |
| `crosscheck_out.txt` | 交叉比对的原样输出 |
| `render_report.py` | 把真实输出渲染成 `verification_report.jpg`（不新增任何结论） |
| `verification_report.jpg` | 自验证报告图片（**本地 CPU 等价性验证**，非实机截图） |
| `verify_report.txt` | 前一轮的 5 项静态检查记录（语法 / 字节一致 / 改动范围审计 / 仿真 / 性能量） |

---

## 三、复现方法

```bash
# [1] 官方输入复现 + 等价性验证 + 公开实机输出交叉比对
pip install "torch==2.9.0+cpu"
python3 verification/cpu_equivalence_check.py

# [2] 两份独立复刻器逐位交叉比对（只需 numpy）
python3 verification/crosscheck_two_impls.py

# [3] 重新渲染报告图片（需要 pillow）
python3 verification/render_report.py
```

`[1]` 需要 `torch`（官方验证脚本的输入生成方式依赖 `torch.manual_seed`）；
`[2]` 只需 `numpy`。

---

## 四、仍然缺少的

**Ascend 950 实机 PASS 截图。** 需要人在具备算力的环境执行：

```bash
bash verify_hidevlab.sh          # 本目录上一级，Ascend 950 + CANN >= 9.1.0
```

脚本会按任务书「环境准备」完成依赖安装、执行官方 `test_matmul_add_relu.py`，
并打印 `PASS: M=N=K=256, blocks=1`。截图产出后应追加到本目录，
并在 issue #777 的评论中补上图片链接。
