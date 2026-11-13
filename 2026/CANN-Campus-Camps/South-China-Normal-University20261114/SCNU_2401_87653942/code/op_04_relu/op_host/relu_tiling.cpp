/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include <algorithm>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {
namespace {
constexpr int64_t COPY_BLOCK_BYTES = 32;
constexpr int64_t MAX_TILE_BYTES = 16 * 1024;
constexpr int64_t IO_BUFFERS_PER_SLOT = 2;
constexpr int64_t DOUBLE_BUFFER_SLOTS = 2;

int64_t CeilDivPositive(int64_t value, int64_t divisor)
{
    return 1 + (value - 1) / divisor;
}

ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    const auto* inputDesc = context->GetInputDesc(0);
    const auto* inputShape = context->GetInputShape(0);
    auto* tiling = context->GetTilingData<ReluTilingData>();
    auto* workspace = context->GetWorkspaceSizes(1);
    if (inputDesc == nullptr || inputShape == nullptr || tiling == nullptr || workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto dtype = inputDesc->GetDataType();
    if (dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }
    const bool isHalf = dtype == ge::DT_FLOAT16;
    const int64_t elementBytes = isHalf ? 2 : 4;
    const int64_t elementsPerCopyBlock = COPY_BLOCK_BYTES / elementBytes;
    const auto& shape = inputShape->GetStorageShape();
    const int64_t totalNum = shape.GetDimNum() == 0 ? 1 : shape.GetShapeSize();
    if (totalNum < 0 || totalNum > std::numeric_limits<int64_t>::max() - (elementsPerCopyBlock - 1)) {
        return ge::GRAPH_FAILED;
    }

    workspace[0] = 0;
    context->SetTilingKey(isHalf ? GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0)
                                : GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1));
    tiling->totalNum = totalNum;
    if (totalNum == 0) {
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    auto* platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(platformInfo);
    const int64_t availableCores = platform.GetCoreNumAiv();
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    if (availableCores <= 0) {
        return ge::GRAPH_FAILED;
    }

    // 以 32B 块为单位划分连续区间，最后一核以总元素数为界。
    const int64_t totalBlocks = CeilDivPositive(totalNum, elementsPerCopyBlock);
    const int64_t blocksPerCore = CeilDivPositive(totalBlocks, availableCores);
    const int64_t blockFactor = blocksPerCore * elementsPerCopyBlock;
    const int64_t ubBlocks = static_cast<int64_t>(ubBytes / COPY_BLOCK_BYTES);
    const int64_t tileCap = MAX_TILE_BYTES / COPY_BLOCK_BYTES;
    const int64_t singleTileLimit = std::min(tileCap, ubBlocks / IO_BUFFERS_PER_SLOT);
    const int64_t tileBlocks = blocksPerCore <= singleTileLimit ? blocksPerCore
        : std::min(tileCap, ubBlocks / (IO_BUFFERS_PER_SLOT * DOUBLE_BUFFER_SLOTS));
    if (tileBlocks == 0) {
        return ge::GRAPH_FAILED;
    }

    tiling->blockFactor = blockFactor;
    tiling->ubFactor = tileBlocks * elementsPerCopyBlock;
    context->SetBlockDim(static_cast<uint32_t>(CeilDivPositive(totalNum, blockFactor)));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};
} // namespace

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
