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
    // Tanh 为逐元素计算，输出 shape 与输入 shape 保持一致
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    if (x_shape == nullptr || y_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *y_shape = *x_shape;
    return ge::GRAPH_SUCCESS;
}
 
IMPL_OP_INFERSHAPE(Tanh).InferShape(InferShapeTanh);
 
} // namespace ops
