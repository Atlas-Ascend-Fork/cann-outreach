# Challenge 09 · MatmulAddRelu —— HansonZhang 提交

本目录是「代码侦探」第九期的个人提交。核心交付物是补全后的算子实现，
另附设计说明、性能解析分析与两个可复现的自检脚本。

> ## ⚠️ 验证状态说明（请先读）
>
> **本次提交尚未在 Ascend NPU 上实机运行过。** 提交者环境无 NPU 算力，
> 因此 `test_matmul_add_relu.py` 未被执行，`verification/` 下暂无实机 PASS 截图。
>
> 本目录内所有检查均为**实际执行过的静态/仿真检查**，不含任何实测时延或吞吐数字。
> 详见 `verification/verify_report.txt` 的完整声明。
>
> **新增**：`verification/` 下已补齐一组更强的**本地 CPU 数值等价性验证**——
> 用**官方测试输入**（`torch.manual_seed(2026)`）复刻本实现的核函数数据流，
> 与官方参考解比对（0/65536 超差），并与**本 issue 内其他参赛者公开的
> Ascend 950 实机输出**逐值交叉比对（36/36 一致）。
> 它是实机验证的**补充证据**，**不是**实机验证的替代品。
> 详见 `verification/VERIFICATION.md`。

---

## 文件清单

| 文件 | 说明 |
|:--|:--|
| `matmul_add_relu.py` | 在 `Template/matmul_add_relu.py` 上补全五处 TODO，其余与模板逐字节一致 |
| `test_matmul_add_relu.py` | 官方验证脚本，**与 `Template/` 逐字节一致**，未做任何改动 |
| `PERF-ANALYSIS.md` | 补全实现的 roofline 解析估算（**非实测**） |
| `perf_facts.py` | 从代码导出确定性性能量的脚本，无硬件参数、可复现 |
| `algosim.py` | 算法级数值仿真，含反向对照，证明仿真有鉴别力 |
| `verify_hidevlab.sh` | 供在 Ascend 950 环境中一键执行官方验证的脚本 |
| `verification/verify_report.txt` | 已执行检查的完整记录与声明 |
| `verification/algosim_out.txt` | 仿真实际输出 |
| `verification/VERIFICATION.md` | **新增** · CPU 数值等价性验证的说明、证据链与边界声明 |
| `verification/cpu_equivalence_check.py` | **新增** · 官方输入复现 → 数据流复刻 → 官方容差比对 → 公开实机输出交叉比对 |
| `verification/cpu_equivalence_out.txt` | **新增** · 上述脚本的原样输出 |
| `verification/second_emulator.py` | **新增** · 第二份独立编写的复刻器（用于逐位交叉比对） |
| `verification/crosscheck_two_impls.py` | **新增** · 驱动两份复刻器逐位交叉比对 |
| `verification/crosscheck_out.txt` | **新增** · 交叉比对原样输出 |
| `verification/verification_report.jpg` | **新增** · 自验证报告图片（**本地 CPU 等价性验证，非实机截图**） |

---

## 补全思路

题目固定 `M=N=K=256`、`block_num=1`，一次 AICORE 覆盖全部计算。
算子语义为 **`Y = ReLU(A @ B + X)`** —— 先 Add 再 ReLU。

### TODO 1 · GM → L1

`A`、`B` 均为连续行优先 FP16，各 128 KiB，可整体装入 L1。
按 `A`、`B` 顺序各调用一次 `tla.copy`，`mutex_l1` 保证两次搬运都完成后 MTE1 才读取。

### TODO 2 · L1 → L0A/L0B

沿 K 轴切成 4 个 `K=64` 的分片。
`A` 的 `256×64` 分片搬进 L0A（`zN`），`B` 的 `64×256` 分片搬进 L0B（`nZ`）。
`mutex_l0` 既保证 Cube 读取前搬运完成，也保证下一次搬运在当前矩阵乘读完 L0A/L0B 之后才开始。

### TODO 3 · 矩阵乘累加

`tla.mmad(l0_c, l0_a, l0_b, init_c=...)`。
`k_tile == 0` 取 `True`，让第一个 K 分片直接覆盖 L0C；后三次取 `False`，
把剩余分片的乘积累加进去。`k_tile` 是 `scf.for` 的归纳变量，
`init_c` 形参声明为 `bool | Bool`，因此可以安全传入运行期布尔值。

> **注意 ReLU 必须留到 Vector 侧**：FixPipe 未开启 `relu_enable`，
> 否则得到的是 `ReLU(A@B) + X`，与题目要求的 `ReLU(A@B + X)` 不符。
> 这是本题最容易写反的一处，`algosim.py` 的反向对照专门针对这一点。

### TODO 4 · L0C → UB

`CopyL0C2DstParams(l0c2ub_mode=L0C2UBMode.SPLIT_M)`
把 L0C 的 256 行按 AIV 数二等分，两个 AIV 各拿到一块 `128×256`。
256 能被 2 整除，**两个 AIV 负载完全均衡，无尾块**。
先提取参数对象再调用 `tla.copy`，便于阅读与单点修改。

### TODO 5 · Vector: Add + ReLU

每个 SIMD 向量从 UB 载入 `acc` 与 `x`，相加后与 `0.0` 取大，再原位写回 `acc` 所在的 UB。
每行 256 列正好是 4 个 64-lane 完整向量，**无需尾掩码**。

---

## 已执行的检查

| 检查 | 结果 | 能证明什么 | 不能证明什么 |
|:--|:--|:--|:--|
| `py_compile` 两文件 | PASS | Python 语法合法 | DSL 能否编译、能否跑通 |
| `cmp` 官方验证脚本 | 字节一致 | 未篡改判定标准 | — |
| `diff -u` 改动范围审计 | 5 hunk / 删 5 个 `pass` / 无残留 | 改动严格限于 TODO，框架零改动 | — |
| 算法级数值仿真 | 0 / 65536 不匹配 | 数据流与算术语义自洽 | Catlass 编译产物在真机的行为 |
| roofline 解析估算 | 见 `PERF-ANALYSIS.md` | 切分选择是否合理 | 任何实测时延/吞吐 |

### 反向对照：仿真确有鉴别力

只证明「我的仿真没报错」是不够的 —— 恒真的仿真也会 PASS。
因此额外做了反向对照：把本题真值 `ReLU(A@B + X)` 故意实现为
错误顺序 `ReLU(A@B) + X`，看错误数是否显著大于 0：

```
[反向对照] 先 ReLU 后 Add 的写法不匹配 = 35348 / 65536
```

错误率过半，说明该仿真能真正区分正确与错误实现，不是摆设。

---

## 如何在 Ascend 950 上完成验证

```bash
git clone https://gitcode.com/HansonZhang/cann-outreach.git
cd cann-outreach/2026/CANN-Code-Detective/Challenge09-MatmulAddRelu/PR_code/HansonZhang
bash verify_hidevlab.sh          # 会自行探测 npu-smi / CANN / 依赖
# 或直接跑官方脚本
python3 test_matmul_add_relu.py --device 0
```

依赖（版本严格按任务书「环境准备」表，不要装最新版）：

| 组件 | 版本 |
|:--|:--|
| 硬件 | Ascend 950 系列 NPU |
| CANN | ≥ 9.1.0 |
| Python | 3.10 ~ 3.13 |
| `ascend-catlass-dsl` | `==2.1.0` |
| `torch` | `2.9.0`（x86 用 CPU wheel 即可，矩阵乘由 NPU 执行） |
| `torch-npu` | `2.9.0.post6` |
| `PyYAML` | — |

验证通过后请把 `npu-smi info` 与 PASS 输出截图放入 `verification/`。