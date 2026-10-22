#!/usr/bin/env python3
# 比对 output/z_gm.bin 与 golden/z_gm.bin
import numpy as np

out = np.fromfile("output/z_gm.bin", dtype=np.float16)
gold = np.fromfile("golden/z_gm.bin", dtype=np.float16)

if out.shape != gold.shape:
    print("FAIL: shape mismatch", out.shape, gold.shape)
    raise SystemExit(1)

err = np.abs(out.astype(np.float32) - gold.astype(np.float32))
max_err = err.max()
rel_ok = np.all(err <= 0.001 * np.abs(gold.astype(np.float32)) + 0.001)

print("max error:", max_err)
print("PASS" if rel_ok else "FAIL")
raise SystemExit(0 if rel_ok else 1)
