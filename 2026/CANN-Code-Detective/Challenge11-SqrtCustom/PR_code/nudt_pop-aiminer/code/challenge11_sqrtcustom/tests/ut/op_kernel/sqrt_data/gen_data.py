#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np

SHAPE = (8, 2048)
SEED = 202611

def make_input(dtype=np.float32):
    dtype = np.dtype(dtype)
    rng = np.random.default_rng(SEED)
    # Log-uniform positive values cover a broad dynamic range reproducibly.
    low, high = (-4.0, 4.0) if dtype == np.float16 else (-20.0, 20.0)
    values = np.power(10.0, rng.uniform(low, high, np.prod(SHAPE))).astype(dtype)
    limits = np.finfo(dtype)
    boundaries = np.array([
        0.0, 0.25, 1.0, 2.0, 4.0, 9.0, 16.0, 100.0,
        10.0 ** low, 10.0 ** high, limits.tiny, limits.max,
    ], dtype=dtype)
    values[:boundaries.size] = boundaries
    return values.reshape(SHAPE)

def impl(x):
    return np.sqrt(x.astype(np.float64)).astype(x.dtype)

if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    for dtype in (np.float32, np.float16):
        name = np.dtype(dtype).name
        # Remove only the outputs consumed by this test, so stale results cannot pass.
        Path(f"{name}_output_sqrt_0.bin").unlink(missing_ok=True)
        x = make_input(dtype)
        x.tofile(f"{name}_input_sqrt_x.bin")
        impl(x).tofile(f"{name}_golden_sqrt.bin")
        print(f"Generated Sqrt input/golden: shape={SHAPE}, dtype={name}, seed={SEED}")
