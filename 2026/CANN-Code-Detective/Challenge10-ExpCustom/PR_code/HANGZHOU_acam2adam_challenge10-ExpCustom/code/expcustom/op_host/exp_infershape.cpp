/*!
 * \file exp_infershape.cpp
 * \brief Exp 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

namespace ops {

static ge::graphStatus InferShapeExp(gert::InferShapeContext* context)
{
    // Exp 为单输入（x）单输出（y）逐元素算子：y.shape = x.shape
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Exp).InferShape(InferShapeExp);

} // namespace ops
