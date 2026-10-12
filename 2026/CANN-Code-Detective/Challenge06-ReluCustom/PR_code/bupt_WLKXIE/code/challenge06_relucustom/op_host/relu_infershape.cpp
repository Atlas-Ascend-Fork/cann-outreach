/*!
 * \file relu_infershape.cpp
 * \brief Relu 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeRelu(gert::InferShapeContext* context) {
    const gert::Shape* in_shape = context->GetInputShape(0);
    if (in_shape == nullptr) {
        return ge::GRAPH_FAILED;  // 必须保证输入存在
    }
    gert::Shape* out_shape = context->GetOutputShape(0);
    if (out_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *out_shape = *in_shape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Relu).InferShape(InferShapeRelu);

} // namespace ops
