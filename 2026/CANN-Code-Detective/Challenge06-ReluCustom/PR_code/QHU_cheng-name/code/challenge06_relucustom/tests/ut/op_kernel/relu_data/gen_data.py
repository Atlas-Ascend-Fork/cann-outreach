import numpy as np
import os
import sys
import glob

try:
    from ml_dtypes import bfloat16
except ImportError:
    bfloat16 = None

sys.path.insert(0, os.path.dirname(__file__))
from SqrtSoftplusTopK import impl

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# 逐元素 relu，任意维度可用；float64 往返对 float16/float32 无损，最后转回输入 dtype
def impl(x):
    dtype = x.dtype
    return np.maximum(0, x.astype(np.float64)).astype(dtype)


if __name__ == "__main__":
    # 清理bin文件
    for f in glob.glob("*.bin"):
        os.remove(f)

    # 从 JSON 第一个 case 获取参数
    d_type = "float32"
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": bfloat16,
        "float64": np.float64,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
        "bool": np.bool_,
        "fp8_e4m3fn": np.uint8,
        "fp8_e5m2": np.uint8,
    }
    np_type = d_type_dict[d_type]

    # 正负混合数据（含 0 边界），确保 max(0,x) 真实生效（透传型错误 kernel 无法通过）
    x = np.random.default_rng(42).uniform(low=-10.0, high=10.0, size=(8, 2048))
    x = np.where(np.abs(x) < 0.01, 0.0, x)
    input_x = x.astype(np.float32)

    # 计算 golden 数据
    golden = impl(input_x)

    # 保存数据到文件
    input_x.astype(np.float32).tofile(f"{d_type}_input_relu_x.bin")
    if golden is not None:
        if isinstance(golden, (list, tuple)):
            with open("float32_golden_relu.bin", "wb") as _f:
                for _g in golden:
                    _g.astype(np.float32).tofile(_f)
        else:
            golden.astype(np.float32).tofile("float32_golden_relu.bin")

    print(f"生成完成: dtype={d_type}")
