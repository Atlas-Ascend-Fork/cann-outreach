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
    // Sigmoid 逐元素计算，输出形状与输入形状一致
    const gert::Shape* x_shape = context->GetInputShape(0);
    if (x_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* y_shape = context->GetOutputShape(0);
    if (y_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *y_shape = *x_shape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Sigmoid).InferShape(InferShapeSigmoid);

} // namespace ops
