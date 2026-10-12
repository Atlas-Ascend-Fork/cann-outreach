/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include <algorithm>
#include <cstdint>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t BLOCK_BYTES = 32;
// 双缓冲下，输入/输出各有两个 buffer。限制单个 buffer 最大 32 KiB，
// 总 UB 占用最多约 128 KiB，为运行时及其它开销保留充足空间。
constexpr int64_t MAX_TILE_BYTES = 32 * 1024;

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// 当前 TilingData 只记录统一的 blockFactor，没有尾核字段，因此选择一个能整除
// totalNum 的最大可用 AIV 核数，保证每个核处理的数据量完全一致。
static int64_t SelectBlockDim(int64_t totalNum, int64_t coreNum)
{
    int64_t blockDim = std::min(totalNum, coreNum);
    while (blockDim > 1 && totalNum % blockDim != 0) {
        --blockDim;
    }
    return blockDim;
}

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const int64_t totalNum = inputShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum must be greater than 0"), return ge::GRAPH_FAILED);

    const auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    int64_t typeSize = 0;
    uint64_t tilingKey = 0;
    if (inputDesc->GetDataType() == ge::DT_FLOAT16) {
        typeSize = static_cast<int64_t>(sizeof(uint16_t));
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    } else if (inputDesc->GetDataType() == ge::DT_FLOAT) {
        typeSize = static_cast<int64_t>(sizeof(float));
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    } else {
        OP_LOGE(context, "Unsupported input dtype: %d", static_cast<int32_t>(inputDesc->GetDataType()));
        return ge::GRAPH_FAILED;
    }

    const int64_t blockDim = SelectBlockDim(totalNum, coreNum);
    OP_CHECK_IF(blockDim <= 0, OP_LOGE(context, "Invalid blockDim"), return ge::GRAPH_FAILED);
    const int64_t blockFactor = totalNum / blockDim;

    // Relu 是轻量级基础矢量指令，不需要额外临时 UB。尽量让每核一次完成；
    // 数据较大时把单个 queue buffer 控制在 32 KiB 以内。
    const int64_t usableTileBytes = std::min<int64_t>(MAX_TILE_BYTES, static_cast<int64_t>(ubSize / 8));
    const int64_t maxTileElems = usableTileBytes / typeSize;
    int64_t ubFactor = std::min(blockFactor, maxTileElems);

    // DataCopy 以 32B block 为基本搬运单位。题目/模板 shape 均天然满足该对齐；
    // 对更一般的较大 shape 也优先让 tile 保持 32B 对齐。
    const int64_t alignElems = BLOCK_BYTES / typeSize;
    if (ubFactor >= alignElems) {
        ubFactor = (ubFactor / alignElems) * alignElems;
    }
    OP_CHECK_IF(ubFactor <= 0, OP_LOGE(context, "Invalid ubFactor"), return ge::GRAPH_FAILED);

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
