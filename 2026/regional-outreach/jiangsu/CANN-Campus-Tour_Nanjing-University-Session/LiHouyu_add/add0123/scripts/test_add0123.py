#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add0123 算子 Python 单元测试
覆盖:
  1. tiling 分核策略: 与 op_host(BLOCK_DIM=8)/op_kernel(TILE_LENGTH=128) 的实现逻辑
     保持镜像, 校验分片不重叠、无遗漏, 尾块正确兜底
  2. 数学正确性: golden 数据生成与校验 (z = x + y, float16 精确可表示用例)
pytest 可直接收集; 也可直接执行: python3 test_add0123.py
"""
import os
import struct
import numpy as np

BLOCK_DIM = 8       # 与 op_host/add0123.cpp 中 BLOCK_DIM 一致
TILE_LENGTH = 128   # 与 op_kernel/add0123.cpp 中 TILE_LENGTH 一致
SHAPE = (8, 2048)   # 默认测试 shape [N2, N1]


# ---------- 镜像 kernel Init 的分核逻辑 ----------
def split_cores(total_length, block_dim=BLOCK_DIM):
    """按 kernel Init 逻辑返回每核的 (start, length); 最后一核兜底余量"""
    block_per_core = total_length // block_dim
    cores = []
    for i in range(block_dim):
        start = i * block_per_core
        length = block_per_core
        if i == block_dim - 1:
            length = total_length - start
        cores.append((start, length))
    return cores


def split_tiles(block_length, tile_length=TILE_LENGTH):
    """按 kernel Process 逻辑返回每片的 (start, length)"""
    tiles = []
    offset = 0
    while offset < block_length:
        length = min(tile_length, block_length - offset)
        tiles.append((offset, length))
        offset += tile_length
    return tiles


# ---------- tiling 单元测试 ----------
def test_core_split_exact():
    """8*2048 可整除: 每核 2048, 连续无缝"""
    cores = split_cores(8 * 2048)
    assert all(length == 2048 for _, length in cores)
    assert [start for start, _ in cores] == [i * 2048 for i in range(8)]


def test_core_split_with_tail():
    """不能整除时: 前核均分, 最后一核兜底, 总量不丢不重"""
    total = 8 * 2048 + 5
    cores = split_cores(total)
    assert sum(length for _, length in cores) == total
    assert cores[0][1] == total // BLOCK_DIM
    assert cores[-1][1] == total - (BLOCK_DIM - 1) * (total // BLOCK_DIM)


def test_core_split_coverage():
    """随机总量: 分片不重叠、无遗漏且连续"""
    for total in [8, 16, 100, 4096, 16384, 20001, 123457]:
        cores = split_cores(total)
        assert sum(length for _, length in cores) == total, f"total={total}"
        for i in range(1, len(cores)):
            prev_start, prev_len = cores[i - 1]
            start, _ = cores[i]
            assert start == prev_start + prev_len, f"gap/overlap at total={total}"


def test_tile_loop_covers_block():
    """每核内分片(128/片)覆盖完整, 尾片长度正确"""
    for block_length in [128, 2048, 2053, 1, 129]:
        tiles = split_tiles(block_length)
        assert sum(length for _, length in tiles) == block_length
        for i in range(1, len(tiles)):
            prev_start, prev_len = tiles[i - 1]
            start, _ = tiles[i]
            assert start == prev_start + prev_len
        # 每片不超过 TILE_LENGTH, 整片恰为 TILE_LENGTH
        assert all(0 < length <= TILE_LENGTH for _, length in tiles)


def test_tiling_struct_size():
    """tiling 结构字段宽度: uint32 可表达的总元素上限"""
    max_len = np.iinfo(np.uint32).max
    assert 8 * 2048 < max_len


# ---------- golden 数据生成与校验 ----------
def make_golden(shape=SHAPE, seed=42):
    """生成 float16 精确可表示的随机数据 (n/4, n<100), 保证 z == x+y 无舍入"""
    rng = np.random.RandomState(seed)
    x = (rng.randint(0, 100, size=shape) / 4.0).astype(np.float16)
    y = (rng.randint(0, 100, size=shape) / 4.0).astype(np.float16)
    z = (x + y).astype(np.float16)
    return x, y, z


def test_golden_math():
    x, y, z = make_golden()
    assert np.array_equal(z, (x + y).astype(np.float16))
    assert x.dtype == y.dtype == z.dtype == np.float16
    assert x.shape == y.shape == z.shape == SHAPE


def test_golden_bin_roundtrip():
    """二进制落盘/回读一致性 (little-endian float16)"""
    x, _, _ = make_golden(seed=7)
    path = os.path.join(os.path.dirname(__file__), "golden_x.bin")
    x.tofile(path)
    try:
        loaded = np.fromfile(path, dtype=np.float16).reshape(SHAPE)
        assert np.array_equal(loaded, x)
    finally:
        os.remove(path)


def test_golden_bin_layout():
    """ND 2维数据按行主序展平, 与 kernel 的一维分片处理一致"""
    x, _, _ = make_golden(shape=(3, 5), seed=3)
    flat = x.reshape(-1)
    assert flat[0] == x[0, 0] and flat[5] == x[1, 0]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} tests passed")
    raise SystemExit(1 if failed else 0)
