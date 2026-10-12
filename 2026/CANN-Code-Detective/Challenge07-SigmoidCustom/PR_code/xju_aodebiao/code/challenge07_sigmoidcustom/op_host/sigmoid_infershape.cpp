/*!
 * \file sigmoid_infershape.cpp
 * \brief Sigmoid 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeSigmoid(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    if (xShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* yShape = context->GetOutputShape(0);
    if (yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    *yShape = *xShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Sigmoid).InferShape(InferShapeSigmoid);

} // namespace ops
