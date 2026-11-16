#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::Tensor* tensorX =
        context->GetRequiredInputTensor(0);

    uint32_t DT_X = static_cast<uint32_t>(
        tensorX->GetDataType());

    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    MulTilingData* tiling =
        context->GetTilingData<MulTilingData>();

    tiling->totalLength = static_cast<uint32_t>(
        context->GetInputShape(0)
            ->GetStorageShape()
            .GetShapeSize());

    // 固定形状 (8, 2048)，每核处理 2048 个元素。
    context->SetBlockDim(8);

    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext* context)
{
    *context->GetOutputShape(0) =
        *context->GetInputShape(0);

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(
    gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(
        0, context->GetInputDataType(0));

    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class Mul : public OpDef {
public:
    explicit Mul(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Mul);

}  // namespace ops