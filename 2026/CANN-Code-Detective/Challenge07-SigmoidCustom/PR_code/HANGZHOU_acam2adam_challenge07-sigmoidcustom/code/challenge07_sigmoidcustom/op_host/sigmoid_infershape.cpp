/*!
 * \file sigmoid_infershape.cpp
 * \brief Sigmoid 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeSigmoid(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* outputShape = context->GetOutputShape(0);
    if (outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeSigmoid(gert::InferDataTypeContext* context)
{
    const ge::DataType inputDtype = context->GetInputDataType(0);
    if (inputDtype != ge::DT_FLOAT16 && inputDtype != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    return context->SetOutputDataType(0, inputDtype) ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

IMPL_OP_INFERSHAPE(Sigmoid).InferShape(InferShapeSigmoid).InferDataType(InferDataTypeSigmoid);

} // namespace ops
