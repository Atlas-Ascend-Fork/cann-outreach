/*!
 * \file relu_infershape.cpp
 * \brief Relu 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeRelu(gert::InferShapeContext* context)
{
    // Relu 为单输入单输出算子，输出 shape 与输入 shape 完全一致
    const gert::Shape* input_shape = context->GetInputShape(0);
    if (input_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* output_shape = context->GetOutputShape(0);
    if (output_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    *output_shape = *input_shape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Relu).InferShape(InferShapeRelu);

} // namespace ops
