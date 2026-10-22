#!/usr/bin/env python3
# 生成 input/x_gm.bin, input/y_gm.bin 以及 golden 输出 golden/z_gm.bin（float16, ND, shape [N2, N1]）
import os
import numpy as np

N2, N1 = 8, 128

def gen_data():
    x = np.random.uniform(-10, 10, (N2, N1)).astype(np.float16)
    y = np.random.uniform(-10, 10, (N2, N1)).astype(np.float16)
    z = (x + y).astype(np.float16)  # golden

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("golden", exist_ok=True)
    x.tofile("input/x_gm.bin")
    y.tofile("input/y_gm.bin")
    z.tofile("golden/z_gm.bin")
    print("shape:", x.shape, "dtype:", x.dtype)
    print("input generated: input/x_gm.bin, input/y_gm.bin")
    print("golden generated: golden/z_gm.bin")

if __name__ == "__main__":
    gen_data()
