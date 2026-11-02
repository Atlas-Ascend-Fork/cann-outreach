#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    int32_t coreNum = platform.GetCoreNumAiv();
    if (coreNum < 1) {
        coreNum = 1;
    }

    const gert::Tensor* tensorX = context->GetRequiredInputTensor(0);
    const gert::Tensor* tensorY = context->GetRequiredInputTensor(1);
    if (tensorX == nullptr || tensorY == nullptr) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType dtypeX = tensorX->GetDataType();
    if ((dtypeX != ge::DT_FLOAT && dtypeX != ge::DT_FLOAT16) ||
        tensorY->GetDataType() != dtypeX ||
        tensorX->GetShapeSize() != tensorY->GetShapeSize()) {
        return ge::GRAPH_FAILED;
    }

    uint32_t length = static_cast<uint32_t>(tensorX->GetShapeSize());

    uint32_t DT_X = static_cast<uint32_t>(dtypeX);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    MulTilingData* tiling = context->GetTilingData<MulTilingData>();
    tiling->length = length;

    // 每个 tile 处理 2048 个元素，与 Kernel 侧保持一致。
    constexpr uint32_t TILE_LENGTH = 2048;
    uint32_t tileCount =
        length / TILE_LENGTH + (length % TILE_LENGTH != 0);

    uint32_t blockNum = static_cast<uint32_t>(coreNum);
    if (blockNum > tileCount) {
        blockNum = tileCount;
    }
    if (blockNum == 0) {
        blockNum = 1;
    }

    context->SetBlockDim(blockNum);

    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    workspaceSizes[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);

    if (inputShape == nullptr || outputShape == nullptr) {
        return GRAPH_FAILED;
    }

    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    return context->SetOutputDataType(
        0, context->GetInputDataType(0));
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