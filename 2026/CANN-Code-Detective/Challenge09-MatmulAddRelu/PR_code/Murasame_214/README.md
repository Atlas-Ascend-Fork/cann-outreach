# Challenge 09 . MatmulAddRelu --- Murasame_214 提交

本目录是「代码侦探」第九期的个人提交。除补全后的算子实现与官方验证脚本外，
额外附带一份设计说明和一份补充自检脚本。

## 文件清单

| 文件 | 说明 |
| :--- | :--- |
| `matmul_add_relu.py` | 在 `Template/matmul_add_relu.py` 上补全五处 `TODO`，其余内容与模板逐字节一致 |
| `test_matmul_add_relu.py` | 与 `Template/test_matmul_add_relu.py` 完全一致的官方验证脚本，未做任何改动 |
| `verify_extra.py` | 补充自检脚本：在官方用例之外增加多随机种子、饱和输入和输入不变性检查 |
| `README.md` | 本文件 |

## 补全思路

题目固定 `M=N=K=256`、`block_num=1`，一次 AICORE 覆盖全部计算。

1. **TODO 1 --- GM -> L1**：`A`、`B` 均为连续行优先 FP16，各 128 KiB，可整体装入 L1。
   按 `A`、`B` 顺序各调用一次 `tla.copy`，`mutex_l1` 保证两次搬运都完成后 MTE1 才读取。
2. **TODO 2 --- L1 -> L0**：沿 K 轴切成 4 个 `K=64` 的分片。
   `A` 的 `256x64` 分片搬进 L0A（`zN`），`B` 的 `64x256` 分片搬进 L0B（`nZ`）。
   `mutex_l0` 既保证 Cube 读取前搬运完成，也保证下一次搬运在当前矩阵乘读完
   L0A/L0B 之后才开始。
3. **TODO 3 --- 矩阵乘累加**：`tla.mmad(l0_c, l0_a, l0_b, init_c=...)`。
   `k_tile == 0` 取 `True`，让第一个 K 分片直接覆盖 L0C；后三次取 `False`，
   把剩余分片的乘积累加进去。`k_tile` 是 `scf.for` 的归纳变量，`k_tile == 0`
   求值得到的是运行期 `i1`；`tla.mmad` 的 `init_c` 形参声明为 `bool | Bool`
   （见 `catlass/core_api.py`），因此可以安全传入。
   注意 ReLU 必须留到 Vector 侧：FixPipe 未开启 `relu_enable`，
   否则会得到 `ReLU(A @ B) + X`，与题目要求的 `ReLU(A @ B + X)` 不符。
4. **TODO 4 --- L0C -> UB**：`CopyL0C2DstParams(l0c2ub_mode=L0C2UBMode.SPLIT_M)`
   把 L0C 的 256 行按 AIV 数二等分，两个 AIV 各拿到一块 `128x256` 的累加结果。
   先把参数对象提取成局部变量再调用 `tla.copy`，便于阅读与单点修改。
5. **TODO 5 --- Vector: Add + ReLU**：每个 SIMD 向量从 UB 载入 `acc` 与 `x`，
   相加后与 `0.0` 取大，再原位写回 `acc` 所在的 UB。
   每行 256 列正好是 4 个 64 lane 的完整向量，无需尾掩码。

## 运行验证

在 Ascend 950 环境的题目目录下执行：

```bash
python PR_code/Murasame_214/test_matmul_add_relu.py --device 0
python PR_code/Murasame_214/verify_extra.py --device 0
```

两个脚本都以 `torch.relu(A.float() @ B.float() + X)` 为参考，通过时打印 `PASS`，
失败时打印 `FAIL` 并返回非零状态。
