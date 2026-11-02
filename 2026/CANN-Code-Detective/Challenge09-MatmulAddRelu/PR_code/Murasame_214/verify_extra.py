"""在官方验证脚本之外补充自检：多随机种子、饱和输入与输入不变性。

用法与 test_matmul_add_relu.py 一致：

    python PR_code/Murasame_214/verify_extra.py --device 0
"""

import argparse

SEEDS = (2026, 7, 13, 29)


def make_inputs(torch, size, seed):
    """构造一组输入；偶数种子用线性 X，奇数种子用饱和随机 X。"""
    torch.manual_seed(seed)
    a = (torch.rand(size, size) * 2 - 1).half()
    b = (torch.rand(size, size) * 2 - 1).half()
    if seed % 2 == 0:
        x = torch.linspace(-2.0, 2.0, size * size).reshape(size, size)
    else:
        x = torch.rand(size, size) * 8 - 4
    return a, b, x


def run_case(torch, tla, from_dlpack, kernel, size, seed):
    """跑一组用例，返回问题描述列表；为空表示通过。"""
    a, b, x = make_inputs(torch, size, seed)
    expected = torch.relu(a.float() @ b.float() + x)

    device_a, device_b, device_x = a.npu(), b.npu(), x.npu()
    device_y = torch.full((size, size), -9999.0, dtype=torch.float32, device="npu")

    def wrap(tensor):
        return from_dlpack(
            tensor, layout_tag=tla.arch.RowMajor, origin_shape=(size, size)
        )

    ta, tb, tx, ty = map(wrap, (device_a, device_b, device_x, device_y))
    artifact = tla.compile(kernel, ta, tb, tx, ty, options="--npu-arch 3510")
    artifact(ta, tb, tx, ty, block_num=1)
    torch.npu.synchronize()

    actual = device_y.cpu()
    tolerance = torch.maximum(torch.ones_like(expected), expected.abs()) / 256.0
    problems = []
    if not bool((torch.isfinite(actual) & (actual >= 0)).all()):
        problems.append("存在非有限值或负值")
    if not bool(((actual - expected).abs() <= tolerance).all()):
        problems.append("与 PyTorch 参考结果的偏差超出容差")
    if not torch.equal(device_a.cpu(), a) or not torch.equal(device_b.cpu(), b):
        problems.append("输入 A/B 被改写")
    if not torch.equal(device_x.cpu(), x):
        problems.append("输入 X 被改写")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="MatmulAddRelu 补充自检")
    parser.add_argument("--device", type=int, default=0, help="NPU 设备号，默认 0")
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device 不能为负数")

    import catlass.tla as tla
    from catlass.tla.runtime import from_dlpack
    import torch
    import torch_npu  # noqa: F401，注册 torch.npu 后端。

    from matmul_add_relu import matmul_add_relu_kernel

    torch.npu.set_device(args.device)
    size = 256
    failures = 0
    for seed in SEEDS:
        try:
            problems = run_case(
                torch, tla, from_dlpack, matmul_add_relu_kernel, size, seed
            )
        except Exception as exc:  # noqa: BLE001，逐组用例隔离失败。
            problems = ["执行异常：%s" % exc]
        if problems:
            failures += 1
            print("FAIL: seed=%d, %s" % (seed, "；".join(problems)))
        else:
            print("PASS: seed=%d, M=N=K=256, blocks=1" % seed)

    if failures:
        print("FAIL: %d/%d 组补充用例未通过" % (failures, len(SEEDS)))
        return 1
    print("PASS: %d/%d 组补充用例全部通过" % (len(SEEDS), len(SEEDS)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("FAIL: %s" % exc)
        raise
