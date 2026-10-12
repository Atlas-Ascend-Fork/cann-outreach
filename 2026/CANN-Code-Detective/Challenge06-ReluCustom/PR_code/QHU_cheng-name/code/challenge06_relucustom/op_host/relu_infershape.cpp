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
    // 形状推导：输出形状与输入一致（y = max(0, x)，逐元素不改变形状）
    for (size_t i = 0; true; i++) {
        gert::Shape* output_shape = context->GetOutputShape(i);
        if (output_shape == nullptr) {
            break;  // 输出耗尽
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
