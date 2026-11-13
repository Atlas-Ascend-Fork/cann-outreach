#!/usr/bin/env python3
"""Independent CPU emulation of the Challenge-09 Catlass kernel `matmul_add_relu_kernel`.

Target: Y = ReLU(A @ B + X), M = N = K = 256, A/B FP16 row-major, X/Y FP32 row-major.

This is a SECOND, independent implementation, written only from the task book and
the submitted kernel, used to cross-check ``algosim.simulate_kernel`` bit for bit.
It deliberately accumulates the FP32 L0C register one k at a time (rank-1 updates,
like the hardware accumulator) instead of calling a BLAS matmul, so that the two
implementations differ in summation order: agreement therefore shows the result is
insensitive to FP32 rounding, not merely that one code path was copied.

Written from (a) the official task book
    2026/CANN-Code-Detective/Challenge09-MatmulAddRelu/README.md
(b) the submitted implementation
    PR_code/HansonZhang/matmul_add_relu.py
and the official validation script Template/test_matmul_add_relu.py.

numpy only -- no torch, no NPU, no network.  Every step below mirrors the
submitted kernel's dataflow *in the same order*:

    Cube:  GM -> L1 (A, B)  ->  [L1 -> L0A/L0B per K slice]  ->  mmad into L0C
                                ->  L0C -> UB (SPLIT_M, 2 AIVs)
    Vector: per AIV, per 64-row chunk:  UB -> add X -> ReLU -> store to UB -> GM

The only "interpretation" points (SPLIT_M row assignment, ReLU placement) are
documented in emulator_B_notes.md.
"""

from __future__ import annotations

import os

import numpy as np

# ---------------------------------------------------------------- kernel consts
# Mirrors the constants in PR_code/HansonZhang/matmul_add_relu.py
M = N = K = 256
L0_K = 64  # K slice length handled by one L0A/L0B tile pair
NUM_AIV = 2  # two Vector sub-cores
AIV_M = M // NUM_AIV  # 128 rows of L0C per AIV under SPLIT_M
X_CHUNK_M = 64  # rows of X handled per Vector iteration
SIMD_LANES = 64  # 256-byte SIMD register holds 64 FP32 elements

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(HERE, "inputs.npz")
OUTPUT = os.path.join(HERE, "second_emulator_out.npy")


def emulate(a_h: np.ndarray, b_h: np.ndarray, x_f32: np.ndarray):
    """Run the emulated kernel; return (Y_gm, l0_c_before_epilogue, y_wrong_order)."""

    # =====================================================================
    # CUBE SECTION
    # =====================================================================

    # -- TODO 1: GM -> L1 -----------------------------------------------
    # `tla.copy(l1_a, gm_a)` / `tla.copy(l1_b, gm_b)`: the *whole* matrices
    # are staged in L1 (A and B are moved as a pair under mutex_l1).
    # l1_a: A[:, 0:256] fp16, l1_b: B[0:256, :] fp16  (row-major, full 256x256)
    l1_a = np.ascontiguousarray(a_h, dtype=np.float16)  # (256, 256) fp16
    l1_b = np.ascontiguousarray(b_h, dtype=np.float16)  # (256, 256) fp16

    # L0C: the FP32 accumulator tile, M x N = 256 x 256.
    l0_c = np.zeros((M, N), dtype=np.float32)

    for k_tile in range(K // L0_K):
        # -- TODO 2: L1 -> L0A / L0B (current K slice) ------------------
        # l1_a_k = tile_view(l1_a, shape(M, L0_K), coord(0, k_tile))
        #        -> A[:, k_tile*64 : (k_tile+1)*64]           (256 x 64)
        # l1_b_k = tile_view(l1_b, shape(L0_K, N), coord(k_tile, 0))
        #        -> B[k_tile*64 : (k_tile+1)*64, :]           ( 64 x 256)
        # (tile_view coords are tile-granularity: element offset = coord * shape)
        c0 = k_tile * L0_K
        l0_a = l1_a[:, c0 : c0 + L0_K].astype(np.float32)  # (256, 64) fp32 view of fp16 data
        l0_b = l1_b[c0 : c0 + L0_K, :].astype(np.float32)  # ( 64, 256)

        # -- TODO 3: mmad into L0C --------------------------------------
        # `tla.mmad(l0_c, l0_a, l0_b, init_c=True if k_tile == 0 else False)`
        #   init_c=True  -> the accumulator is CLEARED first (overwrite)
        #   init_c=False -> the slice product is ACCUMULATED into L0C
        # Operands are FP16, the accumulator is FP32; emulate the FP32
        # accumulator one k at a time (rank-1 FP32 updates).
        init_c = k_tile == 0
        slice_acc = np.zeros((M, N), dtype=np.float32)
        for kk in range(L0_K):
            slice_acc += l0_a[:, kk : kk + 1] * l0_b[kk : kk + 1, :]

        if init_c:
            l0_c = slice_acc  # FIRST slice overwrites L0C
        else:
            l0_c += slice_acc  # the other three slices ACCUMULATE into L0C

    # -- TODO 4: L0C -> UB with SPLIT_M ----------------------------------
    # `tla.copy(ub_acc_full, l0_c, CopyL0C2DstParams(l0c2ub_mode=SPLIT_M))`
    # SPLIT_M splits the 256 rows of L0C over the 2 AIVs: AIV 0 receives rows
    # 0..127, AIV 1 receives rows 128..255.  Each AIV's rows land at offset 0
    # of that AIV's own UB accumulator buffer (128 x 256 fp32).
    # NOTE: no ReLU is applied here -- the FixPipe copy leaves relu_enable=False
    # (X has not been added yet).
    ub_acc = []
    for aiv_id in range(NUM_AIV):
        r0 = aiv_id * AIV_M
        ub_acc.append(l0_c[r0 : r0 + AIV_M, :].copy())  # (128, 256) fp32

    # =====================================================================
    # VECTOR SECTION  (aiv_id = tla.arch.sub_block_idx())
    # =====================================================================
    y_gm = np.zeros((M, N), dtype=np.float32)  # GM Y
    y_wrong = np.zeros((M, N), dtype=np.float32)  # diagnostic only: ReLU BEFORE add

    for aiv_id in range(NUM_AIV):
        # gm_x_aiv / gm_y_aiv = tile_view(gm_x/gm_y, shape(128, 256), coord(aiv_id, 0))
        #   -> global rows [aiv_id*128, aiv_id*128+128)
        aiv_row0 = aiv_id * AIV_M
        gm_x_aiv = x_f32[aiv_row0 : aiv_row0 + AIV_M, :]  # (128, 256) fp32

        for chunk in range(AIV_M // X_CHUNK_M):
            # gm_x_chunk / gm_y_chunk / ub_acc_chunk = tile_view(..., shape(64,256),
            # coord(chunk, 0)) -> rows [chunk*64, chunk*64+64) of the AIV view.
            r0 = chunk * X_CHUNK_M
            ub_x = gm_x_aiv[r0 : r0 + X_CHUNK_M, :]  # (64, 256) fp32, via MTE2
            acc_chunk = ub_acc[aiv_id][r0 : r0 + X_CHUNK_M, :]  # (64, 256) fp32 view of UB

            out_chunk = np.empty((X_CHUNK_M, N), dtype=np.float32)
            out_wrong = np.empty((X_CHUNK_M, N), dtype=np.float32)

            for row in range(X_CHUNK_M):
                for col0 in range(0, N, SIMD_LANES):
                    # tile_view(ub_acc_chunk, shape(1, 64), coord(row, col))
                    #   -> elements [row, row+1) x [col*64, col*64+64)
                    acc_v = acc_chunk[row, col0 : col0 + SIMD_LANES]
                    x_v = ub_x[row, col0 : col0 + SIMD_LANES]

                    # -- TODO 5: Vector Add then ReLU, in place ------------
                    #   sum_v = tla.add(acc_v, x_v)      # add X FIRST
                    #   store tla.max(sum_v, 0.0)        # ReLU AFTER the add
                    sum_v = acc_v + x_v
                    relu_v = np.maximum(sum_v, 0.0)
                    out_chunk[row, col0 : col0 + SIMD_LANES] = relu_v

                    # diagnostic only (NOT part of the result): ReLU before add
                    out_wrong[row, col0 : col0 + SIMD_LANES] = np.maximum(acc_v, 0.0) + x_v

            # store back into the AIV's UB accumulator (in place)
            ub_acc[aiv_id][r0 : r0 + X_CHUNK_M, :] = out_chunk
            # MTE3: UB -> GM Y at rows [aiv_row0+r0, aiv_row0+r0+64)
            y_gm[aiv_row0 + r0 : aiv_row0 + r0 + X_CHUNK_M, :] = out_chunk
            y_wrong[aiv_row0 + r0 : aiv_row0 + r0 + X_CHUNK_M, :] = out_wrong

    return y_gm, l0_c, y_wrong


def main() -> int:
    if not os.path.exists(INPUTS):
        # 自生成一组确定性输入（numpy PCG64，种子 20261027），与官方测试输入无关，
        # 仅用于两份独立复刻器之间的交叉比对。
        rng = np.random.default_rng(20261027)
        a0 = (rng.random((M, M), dtype=np.float32) * 2 - 1).astype(np.float16)
        b0 = (rng.random((K, N), dtype=np.float32) * 2 - 1).astype(np.float16)
        x0 = np.linspace(-2.0, 2.0, M * N, dtype=np.float32).reshape(M, N)
        np.savez(INPUTS, a=a0, b=b0, x=x0)
    data = np.load(INPUTS)
    a_h = data["a"]  # (256,256) float16
    b_h = data["b"]  # (256,256) float16
    x_f32 = data["x"]  # (256,256) float32

    y, l0_c, y_wrong = emulate(a_h, b_h, x_f32)

    # ---- save the emulated result -------------------------------------
    np.save(OUTPUT, y.astype(np.float32))

    # ---- reference + tolerance (official rule from the test script) ----
    expected = np.maximum(a_h.astype(np.float32) @ b_h.astype(np.float32) + x_f32, 0.0)
    tolerance = np.maximum(np.ones_like(expected), np.abs(expected)) / 256.0

    diff = np.abs(y - expected)
    max_abs_err = float(diff.max())
    mismatches = int(np.count_nonzero(diff > tolerance))
    invalid = int(np.count_nonzero(~np.isfinite(y))) + int(np.count_nonzero(y < 0.0))

    # diagnostic only: same emulation but with ReLU applied BEFORE the add,
    # used to show that the tolerance check actually discriminates.
    diff_wrong = np.abs(y_wrong - expected)
    mismatches_wrong = int(np.count_nonzero(diff_wrong > tolerance))

    print(f"second_emulator: CPU emulation of Challenge-09 MatmulAddRelu "
          f"(Y = ReLU(A@B + X), M=N=K=256)")
    print(f"max_abs_err : {max_abs_err:.9g}")
    print(f"mismatches  : {mismatches}  (elements with |actual-expected| > max(1,|expected|)/256)")
    print(f"invalid     : {invalid}  (non-finite or negative elements)")
    print(f"first row [0:4]: {y[0, 0:4].tolist()}")
    print(f"[diagnostic only, not part of the result] mismatches if ReLU were applied "
          f"BEFORE the add: {mismatches_wrong}")
    print(f"saved: {OUTPUT}  shape={y.shape} dtype={y.dtype}")
    return 0 if (mismatches == 0 and invalid == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
