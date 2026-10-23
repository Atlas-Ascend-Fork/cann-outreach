# -*- coding: utf-8 -*-
"""
atanh_cpu.py — CPU reference + test-data generator for the AscendC atanh operator.

Purpose
-------
Provide a float16 reference implementation of atanh so that the NPU kernel
output can be checked for correctness, plus helpers to generate random ND test
tensors of the shape the operator targets: [N4, N3, N2, N1].

The reference computes atanh(x) = 0.5 * ln((1 + x) / (1 - x)) in float32 and
rounds to float16, mirroring exactly what the AscendC kernel does (fp32
transcendental, fp16 output).  This is the same identity the kernel uses, so a
correct kernel output should match within float16 rounding.

Usage
-----
    # generate random input + reference output, save to .npy for a NPU test
    python atanh_cpu.py gen   --shape 2,4,8,16 --out-dir ./data

    # compare a kernel output .npy against the reference computed from input
    python atanh_cpu.py check --input ./data/input.npy --output ./data/output.npy
"""

import argparse
import os
import sys

import numpy as np


def atanh_ref(x_fp16: np.ndarray) -> np.ndarray:
    """float16 reference atanh: fp32 compute, fp16 result (matches the kernel)."""
    x = x_fp16.astype(np.float32)
    # Guard the domain: atanh is defined on (-1, 1).  Values exactly at +/-1
    # produce inf in fp32 as well, which is the expected behavior; we still clip
    # to avoid NaN from a 0 denominator being turned into a warning.
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (1.0 + x) / (1.0 - x)
        y = 0.5 * np.log(r)
    return y.astype(np.float16)


def gen_random(shape, dtype=np.float16, seed=42, lo=-0.99, hi=0.99):
    """Generate a random ND tensor within the atanh domain (lo, hi)."""
    rng = np.random.default_rng(seed)
    # Draw (float64 default), round to float32, then to the target dtype so the
    # distribution is realistic for float16.
    data = rng.uniform(lo, hi, size=shape).astype(np.float32).astype(dtype)
    return data


def save_array(path, arr):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.save(path, arr)
    print(f"  saved {path}  shape={arr.shape}  dtype={arr.dtype}")


def cmd_gen(args):
    shape = tuple(int(v) for v in args.shape.split(","))
    print(f"[gen] shape={shape}  dtype={args.dtype}")
    inp = gen_random(shape, dtype=np.float16, seed=args.seed)
    ref = atanh_ref(inp)
    save_array(os.path.join(args.out_dir, "input.npy"), inp)
    save_array(os.path.join(args.out_dir, "expected.npy"), ref)


def cmd_check(args):
    inp = np.load(args.input).astype(np.float16)
    out = np.load(args.output).astype(np.float16)
    expected = atanh_ref(inp)

    out32 = out.astype(np.float32)
    exp32 = expected.astype(np.float32)
    diff = np.abs(out32 - exp32)
    max_diff = float(diff.max()) if diff.size else 0.0
    mean_diff = float(diff.mean()) if diff.size else 0.0
    # float16 has ~3 significant decimal digits; a generous tolerance for a
    # transcendental chained in fp32 then rounded to fp16.
    ok = bool(np.allclose(out32, exp32, rtol=1e-2, atol=1e-2, equal_nan=True))

    print(f"[check] shape={out.shape}")
    print(f"        max_abs_diff = {max_diff:.6e}")
    print(f"        mean_abs_diff = {mean_diff:.6e}")
    print(f"        result = {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description="CPU reference for the AscendC atanh op")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="generate input + expected output")
    g.add_argument("--shape", default="2,4,8,16", help="comma-separated 4-D shape [N4,N3,N2,N1]")
    g.add_argument("--out-dir", default="./data")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--dtype", default="float16")
    g.set_defaults(func=cmd_gen)

    c = sub.add_parser("check", help="compare kernel output to reference")
    c.add_argument("--input", required=True)
    c.add_argument("--output", required=True)
    c.set_defaults(func=cmd_check)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
