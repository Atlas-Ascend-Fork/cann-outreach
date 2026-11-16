/*!
 * \file tanh_tiling.cpp
 * \brief Tanh 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace optiling {
namespace {
constexpr uint64_t ELEMENTS_PER_GROUP = 64;
constexpr uint64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr uint64_t MAX_TILE_LENGTH = 1024;
// FP32: 4 个队列 buffer 和 4 个 FP32 临时 buffer。
// FP16: 4 个队列 buffer 和 6 个 FP32 临时 buffer。两种路径均为 32 B/元素。
constexpr uint64_t UB_BYTES_PER_ELEMENT = 32;
constexpr uint64_t MAX_MASK_BYTES = MAX_TILE_LENGTH / 8;

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return value / divisor + (value % divisor != 0);
}
} // namespace

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
    const auto* inputDesc = context->GetInputDesc(0);
    const auto* inputShape = context->GetInputShape(0);
    auto* tiling = context->GetTilingData<TanhTilingData>();
    auto* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);

    const auto dtype = inputDesc->GetDataType();
    if (dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    // schMode 是模板自定义参数。tanh.cpp 的 mode 0 对应 half，mode 1 对应 float。
    const uint64_t tilingKey = dtype == ge::DT_FLOAT16
        ? GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0)
        : GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1);

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
        if (availableCores == 0 || ubSize <= MAX_MASK_BYTES) {
            return ge::GRAPH_FAILED;
        }
        const uint64_t ubGroups = (ubSize - MAX_MASK_BYTES) /
            (UB_BYTES_PER_ELEMENT * ELEMENTS_PER_GROUP);
        if (ubGroups == 0) {
            return ge::GRAPH_FAILED;
        }

        // 核起点和 UB 分块起点以 64 个元素对齐，最后一块只搬运有效元素。
        const uint64_t groups = CeilDiv(totalLength, ELEMENTS_PER_GROUP);
        const uint64_t targetCores = std::min(availableCores, CeilDiv(totalLength, MIN_SPLIT_THRESHOLD));
        const uint64_t groupsPerCore = CeilDiv(groups, targetCores);
        const uint64_t groupsPerTile = std::min(
            std::min(ubGroups, MAX_TILE_LENGTH / ELEMENTS_PER_GROUP), groupsPerCore);
        tiling->blockFactor = static_cast<int64_t>(
            std::min(totalLength, groupsPerCore * ELEMENTS_PER_GROUP));
        tiling->ubFactor = static_cast<int64_t>(groupsPerTile * ELEMENTS_PER_GROUP);
        blockDim = static_cast<uint32_t>(CeilDiv(groups, groupsPerCore));
    }

    context->SetBlockDim(blockDim);
    context->SetTilingKey(tilingKey);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTanh([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TanhCompileInfo {};

IMPL_OP_OPTILING(Tanh).Tiling(TanhTilingFunc).TilingParse<TanhCompileInfo>(TilingParseForTanh);

} // namespace optiling
