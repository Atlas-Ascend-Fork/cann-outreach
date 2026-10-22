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
# 直调通路的用例规格（单一来源）
# ============================================================================
#
# DIM0 / DIM1 必须与 op_host/add_custom.asc 中的 kDim0 / kDim1 保持一致，
# 否则 gen_data.py 生成的 .bin 大小与 host 侧读取的字节数不匹配（ReadFile 会报错）。
# ============================================================================

DIM0 = 512   # 第 0 维（行数 N2）
DIM1 = 1024  # 第 1 维（列数 N1）
