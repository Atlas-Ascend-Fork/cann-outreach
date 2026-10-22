import os
import numpy as np


def gen_data_and_golden():
    n2, n1 = 8, 128
    total = n2 * n1
    x = np.random.uniform(-100, 100, [n2, n1]).astype(np.float16)
    y = np.random.uniform(-100, 100, [n2, n1]).astype(np.float16)
    golden = (x + y).astype(np.float16)

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    x.tofile("./input/input_x.bin")
    y.tofile("./input/input_y.bin")
    golden.tofile("./output/golden.bin")
    print("input x shape:", x.shape, "dtype:", x.dtype)
    print("generated input_x.bin / input_y.bin / golden.bin")


if __name__ == "__main__":
    gen_data_and_golden()
