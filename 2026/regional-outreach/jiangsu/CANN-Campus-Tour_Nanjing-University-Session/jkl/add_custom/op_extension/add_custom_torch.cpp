/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ⚠️ Stream 同步约定：
//   ❌ stream(false) + 函数调用 → 乱序：不清 queue，kernel 先于之前操作执行
//   ❌ lambda 内传 NPUStream + OpCommand → 死锁：queue 等 lambda，lambda 等 queue 空
//   ❌ zeros_like 创建输出 → 乱序：zeros_like 入 queue 但 kernel 不入 queue，用 empty_like
//   ✅ stream(true) + 函数调用 → 清 queue，安全（本文件使用此模式）

#include <cstdint>
#include "acl/acl.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "../op_kernel/add_custom_tiling.h"

// kernel 入口声明：签名取自 add_custom_kernel.asc 的核函数，去掉 __global__ __vector__ 并加 extern "C"
extern "C" void add_custom_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                                  uint8_t *x, uint8_t *y, uint8_t *z, uint8_t *tiling);

namespace ascend_kernel {

at::Tensor add_custom_torch(const at::Tensor& x, const at::Tensor& y)
{
    // 规格校验：float16 / 2 维 / ND 连续 / NPU 设备
    TORCH_CHECK(x.scalar_type() == at::kHalf, "add_custom: only float16 is supported, got ", x.scalar_type());
    TORCH_CHECK(y.scalar_type() == at::kHalf, "add_custom: y must be float16, got ", y.scalar_type());
    TORCH_CHECK(x.dim() == 2 && y.dim() == 2, "add_custom: inputs must be 2-D [N2, N1]");
    TORCH_CHECK(x.sizes() == y.sizes(), "add_custom: x and y must have the same shape");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(),
                "add_custom: inputs must be contiguous (ND format)");
    TORCH_CHECK(x.is_privateuseone() && y.is_privateuseone(), "add_custom: inputs must be on NPU");

    int64_t dim0 = x.size(0);
    int64_t dim1 = x.size(1);
    int64_t totalElements = x.numel();
    TORCH_CHECK(totalElements > 0, "add_custom: input tensors must not be empty");

    // 输出与输入同 shape / 同 dtype；用 empty_like 而非 zeros_like（见文件头约定）
    at::Tensor z = at::empty_like(x);

    // stream(true) 在返回 ACL stream 前会清 queue，确保与之前 NPU 操作正确同步
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    // 查询设备 Vector 核数并计算 tiling（与 op_host/add_custom.asc 共用同一份逻辑）
    int32_t deviceId = -1;
    aclrtGetDevice(&deviceId);
    int64_t availableCoreNum = 0;
    auto ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    TORCH_CHECK(ret == ACL_SUCCESS && availableCoreNum > 0, "add_custom: failed to get NPU core count");

    AddCustomTilingData tiling = ComputeAddCustomTiling(
        static_cast<uint64_t>(totalElements), static_cast<uint32_t>(dim0), static_cast<uint32_t>(dim1),
        static_cast<uint32_t>(availableCoreNum));
    TORCH_CHECK(tiling.blockNum > 0, "add_custom: invalid tiling, blockNum == 0");

    // tiling 数据搬到 device
    at::Tensor tilingTensor = at::empty(
        {static_cast<int64_t>(sizeof(AddCustomTilingData))},
        x.options().dtype(at::kByte));
    aclrtMemcpy(tilingTensor.mutable_data_ptr(), sizeof(AddCustomTilingData),
        &tiling, sizeof(AddCustomTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    add_custom_kernel(tiling.blockNum, nullptr, aclStream,
        reinterpret_cast<uint8_t*>(x.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(y.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(z.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(tilingTensor.mutable_data_ptr()));

    return z;
}

} // namespace ascend_kernel
