/*!
 * \file exp_tiling.cpp
 * \brief Exp 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/exp_tiling_data.h"
#include "../op_kernel/exp_tiling_key.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t ALIGN_ELEMS = 32;
constexpr int64_t QUEUE_BUFFER_NUM = 2;

static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "AIV core number is invalid"), return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "UB size is 0"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ExpTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo failed"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize failed"),
        return ge::GRAPH_FAILED);

    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    const int64_t totalNum =
        static_cast<int64_t>(inputShape->GetStorageShape().GetShapeSize());
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "Input tensor is empty"), return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    int64_t typeSize = 0;
    if (inputDesc->GetDataType() == ge::DT_FLOAT16) {
        typeSize = 2;
    } else if (inputDesc->GetDataType() == ge::DT_FLOAT) {
        typeSize = 4;
    } else {
        OP_LOGE(context, "Exp only supports float16 and float32");
        return ge::GRAPH_FAILED;
    }

    // 每个 block 负责一段连续数据。按 32 个元素对齐，保证本题数据搬运长度满足对齐要求。
    int64_t blockFactor = (totalNum + coreNum - 1) / coreNum;
    blockFactor = ((blockFactor + ALIGN_ELEMS - 1) / ALIGN_ELEMS) * ALIGN_ELEMS;
    if (blockFactor > totalNum) {
        blockFactor = totalNum;
    }
    OP_CHECK_IF(blockFactor <= 0, OP_LOGE(context, "blockFactor is invalid"), return ge::GRAPH_FAILED);

    // 输入队列和输出队列各有 BUFFER_NUM 个 buffer。额外预留一半 UB，避免占满 UB。
    const int64_t maxUbFactor =
        static_cast<int64_t>(ubSize / (2 * QUEUE_BUFFER_NUM * typeSize * 2));
    OP_CHECK_IF(
        maxUbFactor < ALIGN_ELEMS,
        OP_LOGE(context, "UB is too small for Exp tiling"),
        return ge::GRAPH_FAILED);

    int64_t ubFactor = (blockFactor < maxUbFactor) ? blockFactor : maxUbFactor;
    if (ubFactor >= ALIGN_ELEMS) {
        ubFactor = (ubFactor / ALIGN_ELEMS) * ALIGN_ELEMS;
    }
    OP_CHECK_IF(ubFactor <= 0, OP_LOGE(context, "ubFactor is invalid"), return ge::GRAPH_FAILED);

    const int64_t blockNum = (totalNum + blockFactor - 1) / blockFactor;

    ExpTilingData* tiling = context->GetTilingData<ExpTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(blockNum));

    // float16 使用 half kernel，float32 使用 float kernel。
    uint64_t tilingKey = GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_1);
    if (inputDesc->GetDataType() == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(EXP_TPL_SCH_MODE_0);
    }
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForExp([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ExpCompileInfo {};

IMPL_OP_OPTILING(Exp).Tiling(ExpTilingFunc).TilingParse<ExpCompileInfo>(TilingParseForExp);

} // namespace optiling
