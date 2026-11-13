"""Challenge 09 补全逻辑的**算法级数值仿真**（纯 NumPy/CPU，不依赖 NPU）。

用途：在没有 Ascend 950 算力的前提下，验证补全代码所表达的数据流与算术语义
（K 分片累加 → SPLIT_M 分行 → Add → ReLU → 写回）是否正确。

⚠️ 本脚本**不是** HiDevLab 实机验证，不能替代 test_matmul_add_relu.py 的 PASS 输出。
   它只证明「补全的算法逻辑自洽」，不证明 Catlass DSL 编译产物在真机上行为一致。
"""

import numpy as np

M = N = K = 256
L0_K = 64
AIV_M = M // 2
X_CHUNK_M = 64
SIMD_LANES = 64


def simulate_kernel(a, b, x):
    """严格按 matmul_add_relu.py 的控制流逐步执行，返回写入 gm_y 的结果。"""
    gm_y = np.full((M, N), -9999.0, dtype=np.float32)

    # ---- Cube 段 ----
    l1_a = np.zeros((M, K), dtype=np.float16)  # zN，仿真里等价于行优先存储
    l1_b = np.zeros((K, N), dtype=np.float16)
    # TODO 1: tla.copy(l1_a, gm_a) / tla.copy(l1_b, gm_b)
    l1_a[...] = a
    l1_b[...] = b

    l0_c = np.zeros((M, N), dtype=np.float32)  # 单缓冲 L0C，四轮全写同一块
    for k_tile in range(K // L0_K):
        l1_a_k = l1_a[:, k_tile * L0_K : (k_tile + 1) * L0_K]  # (256, 64)
        l1_b_k = l1_b[k_tile * L0_K : (k_tile + 1) * L0_K, :]  # (64, 256)
        l0_a = l1_a_k  # zN
        l0_b = l1_b_k  # nZ（SIMD 不做转置）
        # TODO 3: init_c = True if k_tile == 0 else False
        init_c = True if k_tile == 0 else False
        partial = l0_a.astype(np.float32) @ l0_b.astype(np.float32)
        l0_c = partial.copy() if init_c else (l0_c + partial)

    # TODO 4: CopyL0C2DstParams(l0c2ub_mode=SPLIT_M) → 按 M 行均分给 2 个 AIV
    ub_acc_by_aiv = np.split(l0_c, 2, axis=0)  # 两个 (128, 256) 视图
    assert all(v.shape == (AIV_M, N) for v in ub_acc_by_aiv)

    # ---- Vector 段 ----
    for aiv_id in range(2):
        gm_x_aiv = x[aiv_id * AIV_M : (aiv_id + 1) * AIV_M, :]  # (128, 256)
        gm_y_aiv = gm_y[aiv_id * AIV_M : (aiv_id + 1) * AIV_M, :]
        ub_acc_aiv = ub_acc_by_aiv[aiv_id]

        for chunk in range(AIV_M // X_CHUNK_M):
            rows = slice(chunk * X_CHUNK_M, (chunk + 1) * X_CHUNK_M)
            ub_x = gm_x_aiv[rows, :]  # (64, 256)
            ub_acc_chunk = ub_acc_aiv[rows, :]

            # TODO 5: 每个 1×64 SIMD 向量 load → add → max(·,0) → 原位 store
            assert N % SIMD_LANES == 0, "每行应恰好 4 个完整 SIMD 向量，无需尾掩码"
            out = np.zeros_like(ub_acc_chunk)
            for row in range(X_CHUNK_M):
                for col in range(N // SIMD_LANES):
                    c = slice(col * SIMD_LANES, (col + 1) * SIMD_LANES)
                    acc_v = ub_acc_chunk[row, c]  # load()
                    x_v = ub_x[row, c]  # load()
                    sum_v = acc_v + x_v  # tla.add(acc_v, x_v)
                    out[row, c] = np.maximum(sum_v, 0.0)  # tla.max(sum_v, 0.0) → store()

            ub_acc_chunk[...] = out  # 原位写回 acc_reg
            gm_y_aiv[rows, :] = ub_acc_chunk  # tla.copy(gm_y_chunk, ub_acc_chunk)

    return gm_y


def main() -> int:
    rng = np.random.default_rng(2026)
    a = (rng.random((M, N), dtype=np.float32) * 2 - 1).astype(np.float16)
    b = (rng.random((K, N), dtype=np.float32) * 2 - 1).astype(np.float16)
    x = np.linspace(-2.0, 2.0, M * N, dtype=np.float32).reshape(M, N)

    actual = simulate_kernel(a, b, x)
    expected = np.maximum(a.astype(np.float32) @ b.astype(np.float32) + x, 0.0)

    tolerance = np.maximum(np.ones_like(expected), np.abs(expected)) / 256.0
    valid = np.isfinite(actual) & (actual >= 0) & (np.abs(actual - expected) <= tolerance)

    print(f"[算法仿真] max|diff| = {np.abs(actual - expected).max():.6g}")
    print(f"[算法仿真] 不匹配元素 = {int((~valid).sum())} / {valid.size}")
    print(f"[算法仿真] 输出全非负 = {bool((actual >= 0).all())}")

    # 反向对照：若误写成 ReLU(A@B) + X（先 ReLU 后 Add），应当失败
    wrong = np.maximum(a.astype(np.float32) @ b.astype(np.float32), 0.0) + x
    wrong_valid = np.abs(wrong - expected) <= tolerance
    print(f"[反向对照] 先 ReLU 后 Add 的写法不匹配 = {int((~wrong_valid).sum())} （应 > 0）")

    ok = bool(valid.all()) and int((~wrong_valid).sum()) > 0
    print("ALGO-SIM PASS" if ok else "ALGO-SIM FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())