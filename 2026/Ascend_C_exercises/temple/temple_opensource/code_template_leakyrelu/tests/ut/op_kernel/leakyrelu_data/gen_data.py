#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from ml_dtypes import bfloat16

NEGATIVE_SLOPE = 0.01  # 属性 negative_slope，与 test_leakyrelu.cpp 中保持一致

def impl(x, negative_slope):
    # 逐元素 leaky relu：y = max(x, 0) + negative_slope * min(x, 0)
    # float64 中间计算，最后一次性转回输入 dtype
    dtype = x.dtype
    x64 = x.astype(np.float64)
    y = np.where(x64 > 0, x64, np.float64(negative_slope) * x64)
    return y.astype(dtype)


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

    # 生成输入数据：与 test_leakyrelu.cpp 完全一致的正负相间模式
    # (i % 16 - 8) * 0.25，覆盖 [-2.0, 1.75]，保证负半轴 α 缩放分支被验证到
    input_x = ((np.arange(8 * 2048) % 16).astype(np.int64) - 8) * 0.25
    input_x = input_x.astype(d_type_dict["float32"])

    # 计算 golden 数据
    golden = impl(input_x, NEGATIVE_SLOPE)

    # 保存数据到文件
    input_x.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_leakyrelu_x.bin")
    if golden is not None:
        if isinstance(golden, (list, tuple)):
            with open("float32_golden_leakyrelu.bin", "wb") as _f:
                for _g in golden:
                    _g.astype(d_type_dict["float32"]).tofile(_f)
        else:
            golden.astype(d_type_dict["float32"]).tofile("float32_golden_leakyrelu.bin")

    print(f"生成完成: dtype={d_type}, negative_slope={NEGATIVE_SLOPE}")
