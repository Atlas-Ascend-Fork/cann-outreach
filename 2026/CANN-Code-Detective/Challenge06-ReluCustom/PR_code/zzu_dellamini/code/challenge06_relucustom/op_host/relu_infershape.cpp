/*!
 * \file relu_infershape.cpp
 * \brief Relu operator shape inference.
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

namespace ops {

static ge::graphStatus InferShapeRelu(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Relu).InferShape(InferShapeRelu);

} // namespace ops
