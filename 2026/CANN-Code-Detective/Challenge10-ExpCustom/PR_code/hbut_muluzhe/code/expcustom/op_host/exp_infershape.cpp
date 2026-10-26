/*!
 * \file exp_infershape.cpp
 * \brief Exp 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeExp(gert::InferShapeContext* context)
{
    // Exp 为逐元素算子，输出 shape 与输入 shape 完全一致
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    if (xShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *yShape = *xShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Exp).InferShape(InferShapeExp);

} // namespace ops
