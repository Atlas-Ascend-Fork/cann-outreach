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
    // 获取输入 shape
    const gert::Shape* input_shape = context->GetInputShape(0);
    if (input_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 获取输出 shape
    gert::Shape* output_shape = context->GetOutputShape(0);
    if (output_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // Sigmoid 不改变张量形状，直接赋值
    *output_shape = *input_shape;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Sigmoid).InferShape(InferShapeSigmoid);

} // namespace ops