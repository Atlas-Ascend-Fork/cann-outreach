# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# Golden 计算（直调通路与 PyTorch 通路共用）
# ============================================================================
#
# z = x + y
#
# 精度策略：输入是 float16，两个 float16 之和在 float32 中可精确表示，
# 因此参考结果统一用 float32 计算并保存，避免参考值自身被二次舍入，
# 从而把「算子输出的一次 float16 舍入误差」单独暴露给精度校验。
# ============================================================================

import numpy as np


def compute_golden(x, y):
    """计算 z = x + y 的参考输出（float32）。

    Args:
        x, y: numpy.ndarray（float16）或 torch.Tensor（half/fp32）

    Returns:
        float32 的参考输出，与输入同 shape
    """
    if isinstance(x, np.ndarray):
        return x.astype(np.float32) + y.astype(np.float32)
    return x.float() + y.float()
