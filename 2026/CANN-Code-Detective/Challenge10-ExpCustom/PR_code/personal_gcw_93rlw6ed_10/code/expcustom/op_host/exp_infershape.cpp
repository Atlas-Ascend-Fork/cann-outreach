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
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* input_shape = context->GetInputShape(0);
    if (input_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 逐元素算子：输出 shape 与输入 shape 完全一致
    gert::Shape* output_shape = context->GetOutputShape(0);
    if (output_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *output_shape = *input_shape;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Exp).InferShape(InferShapeExp);

} // namespace ops
