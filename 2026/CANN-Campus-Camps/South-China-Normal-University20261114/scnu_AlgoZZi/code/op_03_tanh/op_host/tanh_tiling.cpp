/*!
 * \file tanh_tiling.cpp
 * \brief Tanh 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling/tiling_api.h"
#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"

namespace optiling {

constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t MAX_UB_FACTOR = 4096;
constexpr uint64_t UB_RESERVED_BYTES = 8192;

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
    auto* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const int64_t coreNum = platform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (coreNum <= 0 || ubSize <= UB_RESERVED_BYTES) {
        return ge::GRAPH_FAILED;
    }

    const auto* inputShape = context->GetInputShape(0);
    const auto* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const auto dtype = inputDesc->GetDataType();
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }
    const bool isHalf = dtype == ge::DT_FLOAT16;
    const int64_t typeSize = isHalf ? 2 : 4;
    const int64_t alignNum = 32 / typeSize;
    const int64_t totalNum = inputShape->GetStorageShape().GetShapeSize();
    if (totalNum < 0) {
        return ge::GRAPH_FAILED;
    }

    int64_t blockDim = (totalNum + MIN_SPLIT_THRESHOLD - 1) / MIN_SPLIT_THRESHOLD;
    if (blockDim > coreNum) {
        blockDim = coreNum;
    }
    if (blockDim < 1) {
        blockDim = 1;
    }
    int64_t blockFactor = (totalNum + blockDim - 1) / blockDim;
    blockFactor = (blockFactor + alignNum - 1) / alignNum * alignNum;
    if (blockFactor < alignNum) {
        blockFactor = alignNum;
    }
    blockDim = totalNum > 0 ? (totalNum + blockFactor - 1) / blockFactor : 1;

    int64_t ubFactor = blockFactor < MAX_UB_FACTOR ? blockFactor : MAX_UB_FACTOR;
    uint32_t tmpBufferSize = 0;
    while (ubFactor >= alignNum) {
        uint32_t maxTmpSize = 0;
        uint32_t minTmpSize = 0;
        // FP16 同样转为 FP32 后计算，临时空间按 FP32 查询。
        ge::Shape shape({ubFactor});
        AscendC::GetTanhMaxMinTmpSize(shape, sizeof(float), false, maxTmpSize, minTmpSize);
        tmpBufferSize = (maxTmpSize + 31U) / 32U * 32U;
        if (tmpBufferSize == 0) {
            tmpBufferSize = 32;
        }
        const uint64_t queueBytes = 4 * ubFactor * typeSize;
        const uint64_t castBytes = isHalf ? 2 * ubFactor * sizeof(float) : 0;
        if (queueBytes + castBytes + tmpBufferSize <= ubSize - UB_RESERVED_BYTES) {
            break;
        }
        ubFactor = (ubFactor / 2) / alignNum * alignNum;
    }
    if (ubFactor < alignNum) {
        return ge::GRAPH_FAILED;
    }

    auto* tiling = context->GetTilingData<TanhTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    tiling->tmpBufferSize = tmpBufferSize;
    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    context->SetTilingKey(isHalf ? GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0)
                               : GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1));
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
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
