/*!
 * \file exp_def.cpp
 * \brief Exp 算子定义
 */
#include "register/op_def_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

namespace ge {
static graphStatus InferShapeExp(gert::InferShapeContext* context)
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
class Exp : public OpDef {
public:
    explicit Exp(const char* name) : OpDef(name)
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

        this->SetInferShape(ge::InferShapeExp);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(Exp);
} // namespace ops
