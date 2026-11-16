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
    // Relu 单输入 x -> 单输出 y，输出 shape 与输入 x 完全一致
    const gert::Shape* input_shape = context->GetInputShape(0);
    if (input_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < 1; i++) {
        gert::Shape* output_shape = context->GetOutputShape(i);
        if (output_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const gert::Shape* in_shape = context->GetInputShape(i);
        if (in_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *output_shape = *in_shape;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Relu).InferShape(InferShapeRelu);

} // namespace ops