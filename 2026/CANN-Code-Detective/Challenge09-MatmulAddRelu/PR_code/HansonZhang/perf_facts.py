#!/usr/bin/env python3
"""从补全代码中**精确导出**性能相关的确定性量（不依赖任何硬件参数）。

只输出两类东西：
  1. 可直接从算子代码算出的确定量（访存字节数、FLOPs、切分次数、分片规模）
  2. 以硬件参数为自变量的解析表达式（不代入猜测的硬件数值）

这样 PERF-ANALYSIS.md 里不会出现「拍脑袋的 TFLOPS」。
"""

M = N = K = 256
L0_K = 64                      # 任务书：L0 按 K=64 切分
AIV_COUNT = 2                  # SPLIT_M → 两个 AIV
AIV_M = M // AIV_COUNT         # 128
X_CHUNK_M = 64                 # 每个 AIV 的 X 分片行数
SIMD_LANES = 64

FP16 = 2
FP32 = 4


def row(name, value, unit=""):
    print(f"  {name:<42} {value:>14,} {unit}")


def main():
    print("=" * 72)
    print(" Challenge 09 · MatmulAddRelu —— 确定性性能量（纯代码导出）")
    print("=" * 72)

    n_k = K // L0_K
    print(f"\n[几何]  M={M}  N={N}  K={K}  block_num=1（单 AICORE）")
    print(f"[切分]  K 轴分片数 = {K}/{L0_K} = {n_k}")
    print(f"[切分]  SPLIT_M 把 {M} 行分给 {AIV_COUNT} 个 AIV，每 AIV {AIV_M} 行")
    print(f"        → 整除余数 = {M % AIV_COUNT}（0 表示负载完全均衡，无尾块）")
    print(f"[切分]  每 AIV 的 X 分片 = {AIV_M}/{X_CHUNK_M} = {AIV_M // X_CHUNK_M} 块 × {X_CHUNK_M}×{N}")
    print(f"[向量]  每行 SIMD 向量数 = {N}/{SIMD_LANES} = {N // SIMD_LANES}"
          f"（整除，尾掩码开销 = 0）")

    # ---- GM 访存（唯一真正打到 HBM 的流量）----
    gm_a = M * K * FP16
    gm_b = K * N * FP16
    gm_x = M * N * FP32
    gm_y = M * N * FP32
    gm_total = gm_a + gm_b + gm_x + gm_y
    print(f"\n[GM 访存] 仅此部分进入 HBM；片上 L1/L0/UB 的流量不计入")
    row("读 gm_a  (256×256 FP16)", gm_a, "B")
    row("读 gm_b  (256×256 FP16)", gm_b, "B")
    row("读 gm_x  (256×256 FP32)", gm_x, "B")
    row("写 gm_y  (256×256 FP32)", gm_y, "B")
    row("GM 合计", gm_total, "B")
    row("GM 合计", gm_total / 1024, "KiB")

    # ---- 算力 ----
    flops = 2 * M * N * K
    print(f"\n[算力]")
    row("矩阵乘 FLOPs = 2·M·N·K", flops, "FLOP")
    row("  （X 相加 + ReLU：向量侧 2·M·N 次）", 2 * M * N, "op")

    # ---- 算术强度 ----
    intensity = flops / gm_total
    print(f"\n[算术强度]  roofline 自变量，完全由上面两项决定")
    row("AI = FLOPs / GM 字节", round(intensity, 2), "FLOP/B")
    print(f"       → 任何「Cube 吞吐 : HBM 带宽」比值 > {intensity:.0f} 的硬件")
    print(f"         本算子都是 compute-bound。这是解析结论，不依赖具体型号。")

    # ---- 片上流量（与 HBM 无关，但决定 Cube 利用率）----
    l1_fill = gm_a + gm_b
    l1_to_l0 = gm_a + gm_b           # 4 个分片合起来 = 整份 A、B
    l0c_acc = M * N * FP32 * n_k     # 4 次 mmad，每次读改写整块 L0C
    l0c_to_ub = M * N * FP32
    # 每 AIV 的 UB 工作集 = 累 acc 的**完整 AIV_M 行** + **一个** X 分片（X_CHUNK_M 行）。
    # 注意不是 2 份 acc：任务书明确「128 KiB 存放累加结果、64 KiB 存放 X 分片，合计 192 KiB」。
    vec_ub = AIV_M * N * FP32 + X_CHUNK_M * N * FP32
    print(f"\n[片上流量]  不落 HBM，但决定 Cube/Vector 能否被喂满")
    row("GM→L1 填充", l1_fill, "B")
    row("L1→L0A/L0B（4 分片累计）", l1_to_l0, "B")
    row(f"L0C 累加读写（{n_k} 次 mmad × 256KiB）", l0c_acc, "B")
    row("L0C→UB（SPLIT_M 分发）", l0c_to_ub, "B")
    row("每 AIV UB：acc(AIV_M行)", AIV_M * N * FP32, "B")
    row("每 AIV UB：X 分片(X_CHUNK_M行)", X_CHUNK_M * N * FP32, "B")
    row("每 AIV UB 工作集合计", vec_ub, "B")
    row("  → 占 256KiB UB 上限", round(vec_ub / (256 * 1024) * 100, 1), "%")
    print("       任务书原话：128 KiB 存累加结果 + 64 KiB 存 X 分片 = 192 KiB ✓")

    # ---- 同步结构 ----
    print(f"\n[同步结构]  决定 Cube 与 Vector 能否重叠")
    row("Cube↔Vector 跨核通道数", 1, "条 (cross_flag fix_done)")
    row("Cube 段内 K 分片串行轮次", n_k, "轮")
    row("每轮 mutex lock/unlock 对数", 2, "对 (mutex_l0)")
    row("Cube 段 mutex 对总数（估算）", n_k * 2 + 2, "对")
    print("       → block_num=1 且 FixPipe 之后才有 Vector 工作，")
    print("         两段**严格串行**，不存在 Cube/Vector 重叠。")

    # ---- Vector 指令数 ----
    vec_instr_per_aiv = (AIV_M * N) // SIMD_LANES
    print(f"\n[Vector 侧指令量]  每 AIV")
    row("元素数（AIV_M × N）", AIV_M * N, "")
    row("每元素算子（add + max）", 2, "op")
    row("SIMD 指令数（64 lane/条）", vec_instr_per_aiv * 2, "条")
    print(f"       → {AIV_M*N} / {SIMD_LANES} = {AIV_M*N//SIMD_LANES} 条向量指令"
          f"（每条含 add+max 两步）")

    # ---- 解析式（保留符号）----
    print(f"\n[解析式]  设 P = Cube 实测 FP16 吞吐(FLOP/s), BW = HBM 实测带宽(B/s)")
    print(f"  t_compute  = {flops:,} / P")
    print(f"  t_memory   = {gm_total:,} / BW")
    print(f"  t_compute ≤ t_memory  ⟺  P / BW ≥ {intensity:.2f} FLOP/B")
    print(f"  ⇒ compute_time / memory_time = {intensity:.4f} × (BW / P)")
    print("  上式左边系数已由代码确定；代入实测 P、BW 即得时延比。")
    print("  实测时应替换为 npu-smi / msprof 得到的 P 与 BW，勿用公开宣传值。")

    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())