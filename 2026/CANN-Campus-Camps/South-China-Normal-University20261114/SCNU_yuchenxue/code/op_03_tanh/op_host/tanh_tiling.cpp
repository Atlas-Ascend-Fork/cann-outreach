#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"

namespace optiling {
static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
    const auto* desc = context->GetInputDesc(0);
    const auto* shape = context->GetInputShape(0);

    if (!desc || !shape || !context->GetPlatformInfo()) {
        return ge::GRAPH_FAILED;
    }

    const auto dtype = desc->GetDataType();
    const auto& dims = shape->GetStorageShape();

    if ((dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) ||
        dims.GetDimNum() != 2 ||
        dims.GetDim(0) != 8 ||
        dims.GetDim(1) != 2048) {
        return ge::GRAPH_FAILED;
    }

    auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    const uint32_t cores = platform.GetCoreNumAiv();
    uint64_t ubBytes = 0;

    platform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB, ubBytes);

    if (cores == 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t blockDim = 8;
    while (blockDim > cores) {
        blockDim /= 2;
    }

    constexpr uint32_t totalLength = 8 * 2048;
    constexpr uint32_t tileNum = 8;
    constexpr uint32_t bufferNum = 2;

    const uint32_t tileLength =
        totalLength / blockDim / tileNum / bufferNum;

    const uint32_t typeBytes =
        dtype == ge::DT_FLOAT ? 4 : 2;

    // 输入、输出各两块，加四个 float32 临时张量。
    const uint64_t requiredUb =
        tileLength * (4 * typeBytes + 4 * sizeof(float));

    if (requiredUb > ubBytes) {
        return ge::GRAPH_FAILED;
    }

    auto* tiling = context->GetTilingData<TanhTilingData>();
    auto* workspace = context->GetWorkspaceSizes(1);

    if (!tiling || !workspace) {
        return ge::GRAPH_FAILED;
    }

    tiling->totalLength = totalLength;
    tiling->tileNum = tileNum;
    context->SetBlockDim(blockDim);

    uint32_t DT_X = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTanh(gert::TilingParseContext*)
{
    return ge::GRAPH_SUCCESS;
}

struct TanhCompileInfo {};

IMPL_OP_OPTILING(Tanh)
    .Tiling(TanhTilingFunc)
    .TilingParse<TanhCompileInfo>(TilingParseForTanh);
}