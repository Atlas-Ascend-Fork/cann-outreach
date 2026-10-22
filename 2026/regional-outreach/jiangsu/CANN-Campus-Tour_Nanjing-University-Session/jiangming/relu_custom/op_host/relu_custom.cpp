#include "relu_custom_tiling.h"
#include "relu_tiling_policy.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto* inputShape = context->GetInputShape(0);
    const auto* inputDesc = context->GetInputDesc(0);
    const auto* outputDesc = context->GetOutputDesc(0);
    if (inputShape == nullptr || inputDesc == nullptr || outputDesc == nullptr ||
        context->GetPlatformInfo() == nullptr ||
        inputDesc->GetDataType() != ge::DT_FLOAT16 ||
        outputDesc->GetDataType() != ge::DT_FLOAT16 ||
        inputDesc->GetStorageFormat() != ge::FORMAT_ND ||
        outputDesc->GetStorageFormat() != ge::FORMAT_ND) {
        return ge::GRAPH_FAILED;
    }

    const auto& shape = inputShape->GetStorageShape();
    if (shape.GetDimNum() != 4 || inputShape->GetOriginShape().GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }
    int64_t dims[4];
    for (size_t i = 0; i < 4; ++i) {
        dims[i] = shape.GetDim(i);
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    relu_custom::TilingPlan plan{};
    if (!relu_custom::MakeTilingPlan(dims, 4, platform.GetCoreNumAiv(), ubBytes, plan)) {
        return ge::GRAPH_FAILED;
    }

    ReluCustomTilingData tiling;
    tiling.set_totalLength(plan.totalLength);
    tiling.set_blocksPerCore(plan.blocksPerCore);
    tiling.set_extraBlocks(plan.extraBlocks);
    tiling.set_tileLength(plan.tileLength);
    auto* raw = context->GetRawTilingData();
    auto* workspace = context->GetWorkspaceSizes(1);
    if (raw == nullptr || raw->GetData() == nullptr || workspace == nullptr ||
        raw->GetCapacity() < tiling.GetDataSize()) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(raw->GetData(), raw->GetCapacity());
    raw->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(plan.blockDim);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context)
{
    const auto* x = context->GetInputShape(0);
    auto* y = context->GetOutputShape(0);
    if (x == nullptr || y == nullptr || x->GetDimNum() != 4) {
        return GRAPH_FAILED;
    }
    for (size_t i = 0; i < 4; ++i) {
        if (x->GetDim(i) < -1) {
            return GRAPH_FAILED;
        }
    }
    *y = *x;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    if (context->GetInputDataType(0) != DT_FLOAT16) {
        return GRAPH_FAILED;
    }
    return context->SetOutputDataType(0, DT_FLOAT16);
}
}  // namespace ge

namespace ops {
class ReluCustom : public OpDef {
public:
    explicit ReluCustom(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(ReluCustom);
}  // namespace ops
