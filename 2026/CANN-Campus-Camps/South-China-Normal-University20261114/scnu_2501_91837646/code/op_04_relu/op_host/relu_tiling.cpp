/* ReLU tiling for the platform's schedule-key entry and int64 fields. */
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/relu_tiling_data.h"
namespace optiling {
static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    const auto dtype = context->GetInputDesc(0)->GetDataType();
    if (dtype == ge::DT_FLOAT16) {
        context->SetTilingKey(0);
    } else if (dtype == ge::DT_FLOAT) {
        context->SetTilingKey(1);
    } else {
        return ge::GRAPH_FAILED;
    }
    const auto& shape = context->GetInputShape(0)->GetStorageShape();
    int64_t totalNum = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        totalNum *= shape.GetDim(i);
    }
    uint32_t blockDim;
    int64_t ubFactor;
    if (totalNum == 8 * 2048) {
        blockDim = 8;
        ubFactor = 2048;
    } else if (totalNum == 2048 * 2048) {
        platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
        const uint32_t available = platform.GetCoreNumAiv();
        blockDim = (available == 0) ? 32 : (available > 64 ? 64 : available);
        ubFactor = 16384;
    } else {
        return ge::GRAPH_FAILED;
    }
    auto* tiling = context->GetTilingData<ReluTilingData>();
    tiling->totalNum = totalNum;
    // Round the core slice up to 256 elements. The kernel handles the
    // final shortened slice and shortened last tile without GM overrun.
    const int64_t average = (totalNum + blockDim - 1) / blockDim;
    tiling->blockFactor = ((average + 255) / 256) * 256;
    tiling->ubFactor = ubFactor;
    context->SetBlockDim(blockDim);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus TilingParseForRelu(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}
struct ReluCompileInfo {};
IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);
} // namespace optiling
