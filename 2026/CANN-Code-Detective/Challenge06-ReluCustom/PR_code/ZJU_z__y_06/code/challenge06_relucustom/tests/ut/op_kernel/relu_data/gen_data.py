#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os

import numpy as np


def impl(x):
    return np.maximum(np.zeros((), dtype=x.dtype), x)


if __name__ == "__main__":
    for filename in glob.glob("*.bin"):
        os.remove(filename)

    # 与 test_relu.cpp 使用相同的确定性序列，覆盖负数、零和正数。
    indices = np.arange(45 * 2048, dtype=np.int64)
    input_x = ((indices % 17 - 8).astype(np.float32) * np.float32(0.25)).reshape(45, 2048)
    golden = impl(input_x)
    input_x.tofile("float32_input_relu_x.bin")
    golden.tofile("float32_golden_relu.bin")
    print("生成完成: dtype=float32，包含负数、零和正数")
