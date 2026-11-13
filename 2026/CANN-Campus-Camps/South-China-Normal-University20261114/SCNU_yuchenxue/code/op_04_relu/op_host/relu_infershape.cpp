#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

namespace ops {
static ge::graphStatus InferShapeRelu(gert::InferShapeContext* context)
{
    const auto* input = context->GetInputShape(0);
    auto* output = context->GetOutputShape(0);
    if (!input || !output) {
        return ge::GRAPH_FAILED;
    }
    *output = *input;
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(Relu).InferShape(InferShapeRelu);
} // namespace ops
