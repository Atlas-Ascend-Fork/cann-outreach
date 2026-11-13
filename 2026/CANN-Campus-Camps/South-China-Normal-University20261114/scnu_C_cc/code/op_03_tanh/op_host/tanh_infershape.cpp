
/*!
 * \file tanh_infershape.cpp
 * \brief Tanh output shape is identical to the input shape.
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeTanh(
    gert::InferShapeContext* context)
{
    const gert::Shape* xShape =
        context->GetInputShape(0);

    gert::Shape* yShape =
        context->GetOutputShape(0);

    if (xShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    *yShape = *xShape;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Tanh)
    .InferShape(InferShapeTanh);

} // namespace ops
