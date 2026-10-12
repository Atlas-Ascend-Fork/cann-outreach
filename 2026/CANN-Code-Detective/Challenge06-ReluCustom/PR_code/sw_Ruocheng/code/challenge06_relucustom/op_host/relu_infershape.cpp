/*!
 * \file relu_infershape.cpp
 * \brief Relu 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeRelu(gert::InferShapeContext* context)
{
    // Relu 是逐元素算子，输出 shape 与输入 x 完全一致。
    // 用 auto* 屏蔽不同 CANN 版本 GetInputShape 返回 Shape*/StorageShape* 的差异。
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto* outputShape = context->GetOutputShape(0);
    if (outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Relu).InferShape(InferShapeRelu);

} // namespace ops
