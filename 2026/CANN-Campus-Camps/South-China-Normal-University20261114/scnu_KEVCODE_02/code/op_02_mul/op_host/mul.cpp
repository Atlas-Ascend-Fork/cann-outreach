#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int32_t availableCores = platform.GetCoreNumAiv();
    if (availableCores < 4) {
        return ge::GRAPH_FAILED;
    }
    const gert::Tensor* x = context->GetRequiredInputTensor(0);
    const uint32_t length = static_cast<uint32_t>(x->GetShapeSize());
    if (length != 16384) {
        return ge::GRAPH_FAILED;
    }
    uint32_t blockNum = 4;
    const uint32_t DT_X = static_cast<uint32_t>(x->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X);
    MulTilingData* tiling = context->GetTilingData<MulTilingData>();
    tiling->length = length;
    context->SetBlockDim(blockNum);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context)
{
    *context->GetOutputShape(0) = *context->GetInputShape(0);
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Mul : public OpDef {
public:
    explicit Mul(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(Mul);
}