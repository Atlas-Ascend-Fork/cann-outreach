/*!
 * \file exp_infershape.cpp
 * \brief Exp preserves its input shape.
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

namespace ops {
static ge::graphStatus InferShapeExp(gert::InferShapeContext* context)
{
    const auto* input = context->GetInputShape(0);
    auto* output = context->GetOutputShape(0);
    if (input == nullptr || output == nullptr) return ge::GRAPH_FAILED;
    *output = *input;
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(Exp).InferShape(InferShapeExp);
} // namespace ops
