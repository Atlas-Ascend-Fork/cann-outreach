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
    for (size_t i = 0; i < 1; i++) {
        gert::Shape* output_shape = context->GetOutputShape(i);
        if (output_shape == nullptr) {
            return ge::GRAPH_FAILED;   // 输出 shape 获取失败直接返回
        }
        
        const gert::Shape* in_shape = context->GetInputShape(i);
        if (in_shape == nullptr) {
            return ge::GRAPH_FAILED;   // 新增：输入 shape 为 nullptr 时立即返回失败
        }
        
        *output_shape = *in_shape;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Exp).InferShape(InferShapeExp);

} // namespace ops
