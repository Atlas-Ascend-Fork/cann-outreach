#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from ml_dtypes import bfloat16

def impl(x):
    # ReLU：逐元素 max(x, 0)；float64 往返对 float16 无损，最后转回输入 dtype
    dtype = x.dtype
    return np.maximum(x.astype(np.float64), 0.0).astype(dtype)


if __name__ == "__main__":
    # 清理bin文件
    for f in glob.glob("*.bin"):
        os.remove(f)

    # 从 JSON 第一个 case 获取参数
    d_type = "float16"
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": bfloat16,
        "float64": np.float64,
    }
    np_type = d_type_dict[d_type]

    # 生成输入数据（与 test_relu.cpp 中的 xHost 保持一致：4D {8, 4, 2, 512}，正负 1 交替）
    input_x = np.ones((8, 4, 2, 512), dtype=np.float16)
    input_x.reshape(-1)[1::2] = -1.0

    # 计算 golden 数据
    golden = impl(input_x)

    # 保存数据到文件
    input_x.astype(np.float16).tofile(f"{d_type}_input_relu_x.bin")
    if golden is not None:
        if isinstance(golden, (list, tuple)):
            with open("float16_golden_relu.bin", "wb") as _f:
                for _g in golden:
                    _g.astype(np.float16).tofile(_f)
        else:
            golden.astype(np.float16).tofile("float16_golden_relu.bin")

    print(f"生成完成: dtype={d_type}")
