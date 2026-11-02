#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

namespace ops {
static ge::graphStatus InferShapeSqrt(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* input = context->GetInputShape(0);
    gert::Shape* output = context->GetOutputShape(0);
    if (input == nullptr || output == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *output = *input;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Sqrt).InferShape(InferShapeSqrt);
} // namespace ops
