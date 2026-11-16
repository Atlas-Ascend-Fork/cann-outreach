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
    // tanh 为逐元素运算，输出 shape 与输入 shape 一致
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    if (xShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *yShape = *xShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Tanh).InferShape(InferShapeTanh);

} // namespace ops
