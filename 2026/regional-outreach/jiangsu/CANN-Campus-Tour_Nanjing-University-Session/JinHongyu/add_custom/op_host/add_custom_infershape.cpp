/*!
 * \file add_custom_infershape.cpp
 * \brief add_custom 算子形状/数据类型推导
 *
 * 逐元素算子：输出 shape 与输入 x 完全一致；输出 dtype 与输入 dtype 一致。
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeAddCustom(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (inputShape != nullptr) {
        *outputShape = *inputShape;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAddCustom(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AddCustom).InferShape(InferShapeAddCustom).InferDataType(InferDataTypeAddCustom);

} // namespace ops
