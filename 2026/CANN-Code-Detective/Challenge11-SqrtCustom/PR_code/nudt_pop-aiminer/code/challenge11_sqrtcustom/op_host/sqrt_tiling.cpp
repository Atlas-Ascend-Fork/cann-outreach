#include <algorithm>
#include <cstdint>
#include <limits>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/sqrt_tiling_data.h"
#include "../op_kernel/sqrt_tiling_key.h"

namespace optiling {
constexpr uint64_t DMA_BLOCK_BYTES = 32;
constexpr uint64_t RESERVED_UB_BYTES = 1024;
constexpr uint64_t BUFFER_COUNT = 4; // input/output, each double-buffered
constexpr uint64_t MAX_TILE_BYTES = 255 * 256;

static ge::graphStatus SqrtTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* input = context->GetInputShape(0);
    const auto* output = context->GetOutputShape(0);
    const auto* inputDesc = context->GetInputDesc(0);
    const auto* outputDesc = context->GetOutputDesc(0);
    if (input == nullptr || output == nullptr || inputDesc == nullptr || outputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dtype = inputDesc->GetDataType();
    if ((dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) ||
        outputDesc->GetDataType() != dtype) {
        return ge::GRAPH_FAILED;
    }
    const auto& shape = input->GetStorageShape();
    const auto& outputShape = output->GetStorageShape();
    if (shape.GetDimNum() != outputShape.GetDimNum()) {
        return ge::GRAPH_FAILED;
    }
    int64_t totalNum = 1; // A rank-zero tensor contains one element.
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim < 0 || dim != outputShape.GetDim(i) ||
            (dim != 0 && totalNum > std::numeric_limits<int64_t>::max() / dim)) {
            return ge::GRAPH_FAILED;
        }
        totalNum *= dim;
    }

    auto* rawTiling = context->GetRawTilingData();
    if (rawTiling == nullptr || rawTiling->GetCapacity() < sizeof(SqrtTilingData)) {
        return ge::GRAPH_FAILED;
    }
    auto* tiling = context->GetTilingData<SqrtTilingData>();
    auto* workspace = context->GetWorkspaceSizes(1);
    if (tiling == nullptr || workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *tiling = SqrtTilingData{};
    tiling->totalNum = totalNum;
    rawTiling->SetDataSize(sizeof(SqrtTilingData));
    workspace[0] = 0;
    const int dispatchId = dtype == ge::DT_FLOAT ? SQRT_FLOAT32 : SQRT_FLOAT16;
    ASCENDC_TPL_SEL_PARAM(context, dispatchId);
    if (totalNum == 0) {
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    auto* platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(platformInfo);
    const uint32_t coreNum = platform.GetCoreNumAiv();
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    if (coreNum == 0 || ubBytes <= RESERVED_UB_BYTES) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t typeBytes = dtype == ge::DT_FLOAT ? sizeof(float) : sizeof(uint16_t);
    const uint64_t alignment = DMA_BLOCK_BYTES / typeBytes;
    const uint64_t total = static_cast<uint64_t>(totalNum);
    const uint64_t perCore = (total + coreNum - 1) / coreNum;
    // Core boundaries are DMA-aligned; only the last core can have a short tail.
    const uint64_t blockFactor = ((perCore + alignment - 1) / alignment) * alignment;
    if (blockFactor > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t usedCores = (total + blockFactor - 1) / blockFactor;
    const uint64_t availableElements = (ubBytes - RESERVED_UB_BYTES) / BUFFER_COUNT / typeBytes;
    const uint64_t maxElements = std::min(availableElements, MAX_TILE_BYTES / typeBytes);
    const uint64_t tileCapacity = maxElements / alignment * alignment;
    if (tileCapacity == 0) {
        return ge::GRAPH_FAILED;
    }
    tiling->blockFactor = static_cast<int64_t>(blockFactor);
    tiling->ubFactor = static_cast<int64_t>(std::min(blockFactor, tileCapacity));
    context->SetBlockDim(static_cast<uint32_t>(usedCores));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSqrt([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SqrtCompileInfo {};
IMPL_OP_OPTILING(Sqrt).Tiling(SqrtTilingFunc).TilingParse<SqrtCompileInfo>(TilingParseForSqrt);
} // namespace optiling
