#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"
#include "relu_plan.h"

namespace optiling {
static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    const auto* desc = context->GetInputDesc(0);
    const auto* shape = context->GetInputShape(0);
    if (!desc || !shape || !context->GetPlatformInfo()) {
        return ge::GRAPH_FAILED;
    }
    const auto dtype = desc->GetDataType();
    const auto& dims = shape->GetStorageShape();
    if ((dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) ||
        dims.GetDimNum() != 2 || dims.GetDim(1) != 2048 ||
        (dims.GetDim(0) != 8 && dims.GetDim(0) != 2048)) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    CourseRelu::Plan plan{};
    if (!CourseRelu::MakePlan(
            static_cast<uint32_t>(dims.GetShapeSize()),
            dtype == ge::DT_FLOAT ? 4 : 2,
            platform.GetCoreNumAiv(), ubBytes, RELU_BUFFER_NUM, plan)) {
        return ge::GRAPH_FAILED;
    }
    auto* tiling = context->GetTilingData<ReluTilingData>();
    auto* workspace = context->GetWorkspaceSizes(1);
    if (!tiling || !workspace) {
        return ge::GRAPH_FAILED;
    }
    tiling->totalLength = plan.totalLength;
    tiling->tileNum = plan.tileNum;
    tiling->blockLength = plan.totalLength / plan.blockDim;
    tiling->tileLength = plan.tileLength;
    context->SetBlockDim(plan.blockDim);
    uint32_t DT_X = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu(gert::TilingParseContext*)
{
    return ge::GRAPH_SUCCESS;
}
struct ReluCompileInfo {};
IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);
} // namespace optiling
