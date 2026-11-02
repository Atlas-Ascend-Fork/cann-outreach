#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np

def impl(x):
    # 验算函数：逐元素 sqrt，float64 中间计算，最后一次性转回输入 dtype（约定输入 x >= 0）
    dtype = x.dtype
    return np.sqrt(x.astype(np.float64)).astype(dtype)


if __name__ == "__main__":
    # 清理bin文件
    for f in glob.glob("*.bin"):
        os.remove(f)
    
    # 从 JSON 第一个 case 获取参数
    d_type = "float32"
    # 当前 kernel UT 仅运行 float32；无需为未测试的 dtype 引入额外依赖。
    np_type = np.float32
    
    # 生成输入数据
    indices = np.arange(8 * 2048, dtype=np.int64)
    input_x = (((indices * 37) % 257 + 1) / 16.0).astype(np.float32).reshape(8, 2048)
    input_x.flat[:4] = [0.0, 1e-6, 1.0, 1e4]

    
    # 计算 golden 数据
    golden = impl(input_x)
    
    # 保存数据到文件
    input_x.astype(np_type).tofile(f"{d_type}_input_sqrt_x.bin")
    if golden is not None:
        if isinstance(golden, (list, tuple)):
            with open("float32_golden_sqrt.bin", "wb") as _f:
                for _g in golden:
                    _g.astype(np_type).tofile(_f)
        else:
            golden.astype(np_type).tofile("float32_golden_sqrt.bin")
    
    print(f"生成完成: dtype={d_type}")
