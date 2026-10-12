/*!
 * \file sigmoid_tiling.cpp
 * \brief Sigmoid 算子 Tiling 实现
 */

#include <algorithm>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/sigmoid_tiling_data.h"
#include "../op_kernel/sigmoid_tiling_key.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t DEFAULT_BLOCK_DIM = 8;
constexpr int64_t DEFAULT_UB_FACTOR = 256;

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

static ge::graphStatus SigmoidTilingFunc(gert::TilingContext* context)
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

    // 题目约束输入为 (8, 2048)，共 16384 个元素。
    // 使用 8 个 Vector Core，每核处理 2048 个元素；每次 UB 处理 256 个元素。
    const int64_t blockDim = std::min<int64_t>(DEFAULT_BLOCK_DIM, coreNum);
    OP_CHECK_IF(totalNum % blockDim != 0,
                OP_LOGE(context, "totalNum(%ld) must be divisible by blockDim(%ld)", totalNum, blockDim),
                return ge::GRAPH_FAILED);

    const int64_t blockFactor = totalNum / blockDim;
    const int64_t ubFactor = std::min<int64_t>(DEFAULT_UB_FACTOR, blockFactor);
    OP_CHECK_IF(blockFactor % ubFactor != 0,
                OP_LOGE(context, "blockFactor(%ld) must be divisible by ubFactor(%ld)", blockFactor, ubFactor),
                return ge::GRAPH_FAILED);

    SigmoidTilingData* tiling = context->GetTilingData<SigmoidTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(blockDim);

    uint64_t tilingKey = 0;
    const auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    if (inputDesc->GetDataType() == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(SIGMOID_TPL_SCH_MODE_0);
    } else if (inputDesc->GetDataType() == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(SIGMOID_TPL_SCH_MODE_1);
    } else {
        OP_LOGE(context, "Unsupported input dtype: %d", static_cast<int32_t>(inputDesc->GetDataType()));
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSigmoid([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SigmoidCompileInfo {};

IMPL_OP_OPTILING(Sigmoid).Tiling(SigmoidTilingFunc).TilingParse<SigmoidCompileInfo>(TilingParseForSigmoid);

} // namespace optiling
