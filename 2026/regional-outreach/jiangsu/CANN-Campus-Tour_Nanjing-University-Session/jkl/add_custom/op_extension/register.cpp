/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <torch/library.h>
#include "ops.h"

namespace {

// 算子签名注册：z = x + y
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("add_custom(Tensor x, Tensor y) -> Tensor");
}

// NPU（PrivateUse1）实现绑定
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("add_custom", TORCH_FN(ascend_kernel::add_custom_torch));
}

// Meta 实现（torch.compile / fx 需要）：只推导输出 shape/dtype，不做实际计算
// 逐元素加法输出与输入同 shape/dtype，直接 empty_like 即可
at::Tensor add_custom_meta(const at::Tensor& x, const at::Tensor& y)
{
    return at::empty_like(x);
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("add_custom", &add_custom_meta);
}

} // namespace
