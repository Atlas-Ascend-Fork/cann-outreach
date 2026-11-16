/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace optiling {
namespace {
constexpr uint64_t DMA_ALIGNMENT = 32;
constexpr uint64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr uint64_t QUEUE_COUNT = 2; // 输入和输出各一个队列
constexpr uint64_t BUFFER_NUM = 2;  // 每个队列双缓冲
constexpr uint64_t MAX_TILE_BYTES = 32 * 1024;
constexpr uint64_t PREFERRED_GM_ALIGNMENT = 512;
constexpr uint64_t MIN_BYTES_PER_CORE_FOR_PREFERRED_ALIGNMENT = 4 * 1024;

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return value / divisor + (value % divisor != 0);
}
} // namespace

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    const auto* inputDesc = context->GetInputDesc(0);
    const auto* inputShape = context->GetInputShape(0);
    auto* tiling = context->GetTilingData<ReluTilingData>();
    auto* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);

    const auto dtype = inputDesc->GetDataType();
    if (dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    // schMode 是模板自定义参数。relu.cpp 的 mode 0 对应 half，mode 1 对应 float。
    const uint64_t tilingKey = dtype == ge::DT_FLOAT16
        ? GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0)
        : GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);

    const auto& shape = inputShape->GetStorageShape();
    bool empty = false;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim < 0) {
            return ge::GRAPH_FAILED;
        }
        empty = empty || dim == 0;
    }

    uint64_t totalLength = empty ? 0U : 1U;
    constexpr uint64_t maxElements = std::numeric_limits<int64_t>::max();
    if (!empty) {
        for (size_t i = 0; i < shape.GetDimNum(); ++i) {
            const uint64_t dim = static_cast<uint64_t>(shape.GetDim(i));
            if (dim > maxElements / totalLength) {
                return ge::GRAPH_FAILED;
            }
            totalLength *= dim;
        }
    }

    uint32_t blockDim = 1;
    tiling->totalNum = static_cast<int64_t>(totalLength);
    tiling->blockFactor = 1;
    tiling->ubFactor = 0;
    if (totalLength != 0) {
        auto* platformInfo = context->GetPlatformInfo();
        OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
        auto platform = platform_ascendc::PlatformAscendC(platformInfo);
        const uint64_t availableCores = platform.GetCoreNumAiv();
        uint64_t ubSize = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

        const uint64_t singleBufferGroups = std::min(ubSize / QUEUE_COUNT,
            MAX_TILE_BYTES) / DMA_ALIGNMENT;
        if (availableCores == 0 || singleBufferGroups == 0) {
            return ge::GRAPH_FAILED;
        }

        // 小任务按 32 B 对齐。数据量足够时，将核间 GM 偏移对齐到 512 B。
        const uint64_t elementBytes = dtype == ge::DT_FLOAT16 ? 2U : 4U;
        const uint64_t elementsPerGroup = DMA_ALIGNMENT / elementBytes;
        const uint64_t groups = CeilDiv(totalLength, elementsPerGroup);
        const uint64_t targetCores = std::min(availableCores, CeilDiv(totalLength, MIN_SPLIT_THRESHOLD));
        uint64_t groupsPerCore = CeilDiv(groups, targetCores);
        if (groupsPerCore >= MIN_BYTES_PER_CORE_FOR_PREFERRED_ALIGNMENT / DMA_ALIGNMENT) {
            constexpr uint64_t groupsPerAlignment = PREFERRED_GM_ALIGNMENT / DMA_ALIGNMENT;
            groupsPerCore = CeilDiv(groupsPerCore, groupsPerAlignment) * groupsPerAlignment;
        }

        // 一块能放下时只需两个 UB buffer。多块时为输入、输出各预留两个。
        uint64_t groupsPerTile = groupsPerCore;
        if (groupsPerTile > singleBufferGroups) {
            const uint64_t doubleBufferGroups = std::min(ubSize / (QUEUE_COUNT * BUFFER_NUM),
                MAX_TILE_BYTES) / DMA_ALIGNMENT;
            if (doubleBufferGroups == 0) {
                return ge::GRAPH_FAILED;
            }
            groupsPerTile = doubleBufferGroups;
        }
        tiling->blockFactor = static_cast<int64_t>(
            std::min(totalLength, groupsPerCore * elementsPerGroup));
        tiling->ubFactor = static_cast<int64_t>(groupsPerTile * elementsPerGroup);
        blockDim = static_cast<uint32_t>(CeilDiv(groups, groupsPerCore));
    }

    context->SetBlockDim(blockDim);
    context->SetTilingKey(tilingKey);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
