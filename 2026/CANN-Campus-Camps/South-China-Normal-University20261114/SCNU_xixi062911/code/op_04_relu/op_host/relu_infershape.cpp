/*!
 * \file relu_infershape.cpp
 * \brief Relu 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

// 逐元素算子：输出 y 的 shape 与输入 x 完全一致
static ge::graphStatus InferShapeRelu(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    if (xShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *yShape = *xShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Relu).InferShape(InferShapeRelu);

} // namespace ops
