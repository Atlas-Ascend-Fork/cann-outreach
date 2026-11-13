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
    if (context == nullptr) { return ge::GRAPH_FAILED; }
    const gert::Shape* input = context->GetInputShape(0);
    gert::Shape* output = context->GetOutputShape(0);
    if (input == nullptr || output == nullptr) { return ge::GRAPH_FAILED; }
    *output = *input;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Tanh).InferShape(InferShapeTanh);

} // namespace ops

