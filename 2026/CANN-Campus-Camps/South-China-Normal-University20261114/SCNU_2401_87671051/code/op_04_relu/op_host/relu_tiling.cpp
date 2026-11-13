/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;

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

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const gert::StorageShape* in_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    int64_t totalLength = 1;
    for (size_t i = 0; i < in_shape->GetStorageShape().GetDimNum(); i++) {
        totalLength *= in_shape->GetStorageShape().GetDim(i);
    }

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const bool isFp16 = (inputDesc->GetDataType() == ge::DT_FLOAT16);
    const int64_t typeSize = isFp16 ? 2 : 4;
    const int64_t alignNum = 32 / typeSize;

    const bool isSmall = (totalLength <= 65536);
    uint64_t tilingKey;
    if (isSmall) {
        tilingKey = isFp16 ? GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0)
                           : GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    } else {
        tilingKey = isFp16 ? GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_2)
                           : GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_3);
    }
    context->SetTilingKey(tilingKey);

    if (totalLength <= 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    int64_t usedCoreNum = 1;
    int64_t blockFactor = totalLength;
    int64_t ubFactor = alignNum;

    if (isSmall) {
        // 小 shape：4 核一次搬完，单 TBuf + 手动事件最省开销
        usedCoreNum = coreNum < 4 ? coreNum : 4;
        blockFactor = (totalLength + usedCoreNum - 1) / usedCoreNum;
        blockFactor = (blockFactor + alignNum - 1) / alignNum * alignNum;
        usedCoreNum = (totalLength + blockFactor - 1) / blockFactor;
        ubFactor = blockFactor;
    } else {
        // 大 shape：32 核是本环境有效上限，40 个 block 会被分成两波反而更慢
        usedCoreNum = coreNum < 32 ? coreNum : 32;
        blockFactor = (totalLength + usedCoreNum - 1) / usedCoreNum;
        blockFactor = (blockFactor + alignNum - 1) / alignNum * alignNum;
        usedCoreNum = (totalLength + blockFactor - 1) / blockFactor;

        // 双缓冲下 2 个队列 × 2 块，tile 取 2 的幂（实测优于更大的 12288 tile）
        int64_t maxTile = static_cast<int64_t>(ubSize * 85 / 100) / (4 * typeSize);
        int64_t tile = alignNum;
        while (tile * 2 <= maxTile && tile * 2 <= blockFactor) {
            tile *= 2;
        }
        ubFactor = tile;
    }

    tiling->totalNum = totalLength;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<int32_t>(usedCoreNum));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
