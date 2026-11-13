/*!
 * \file abs_def.cpp
 * \brief Abs 算子定义
 */
#include "register/op_def_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

namespace ge {
static graphStatus InferShapeAbs(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return GRAPH_FAILED;
    }
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Abs : public OpDef {
public:
    explicit Abs(const char* name) : OpDef(name)
    {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();

        this->SetInferShape(ge::InferShapeAbs);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(Abs);
} // namespace ops

