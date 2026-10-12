/*!
 * \file tanh_infershape.cpp
 * \brief Tanh 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeTanh(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // Tanh 是逐元素算子，输出 shape 与输入完全一致。
    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Tanh).InferShape(InferShapeTanh);

} // namespace ops
