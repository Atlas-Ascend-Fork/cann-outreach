/*!
 * \file sqrt_infershape.cpp
 * \brief Sqrt 算子形状推导实现
 *
 * Sqrt 为逐元素单输入单输出算子，输出 shape 与输入 shape 完全一致
 * （含标量 shape {} 与动态维 -1，逐维原样透传）。
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeSqrt(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    if (xShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    *yShape = *xShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Sqrt).InferShape(InferShapeSqrt);

} // namespace ops
