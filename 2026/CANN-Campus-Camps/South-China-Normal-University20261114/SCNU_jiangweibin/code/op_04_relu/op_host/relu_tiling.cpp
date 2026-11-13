#include "register/op_def_registry.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {
static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    auto* platformInfo = context->GetPlatformInfo();
    const auto* shape = context->GetInputShape(0);
    const auto* desc = context->GetInputDesc(0);
    if (platformInfo == nullptr || shape == nullptr || desc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto dtype = desc->GetDataType();
    const int64_t total = shape->GetStorageShape().GetShapeSize();
    if ((dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT) || total <= 0 || total % 16 != 0) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const int64_t coreCount = platform.GetCoreNumAiv();
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    constexpr int64_t tile = 4096;
    constexpr uint64_t requiredUb = 4 * tile * sizeof(float);
    if (coreCount <= 0 || ubBytes < requiredUb) {
        return ge::GRAPH_FAILED;
    }
    const int64_t usefulCores = (total + 1023) / 1024;
    const int64_t targetCores = coreCount < usefulCores ? coreCount : usefulCores;
    const int64_t units = total / 16;
    const int64_t block = ((units + targetCores - 1) / targetCores) * 16;
    auto* tiling = context->GetTilingData<ReluTilingData>();
    tiling->totalNum = total;
    tiling->blockFactor = block;
    tiling->ubFactor = tile;
    context->SetBlockDim((total + block - 1) / block);
    context->SetTilingKey(dtype == ge::DT_FLOAT16 ? GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0)
                                                : GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1));
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};
IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);
} // namespace optiling
