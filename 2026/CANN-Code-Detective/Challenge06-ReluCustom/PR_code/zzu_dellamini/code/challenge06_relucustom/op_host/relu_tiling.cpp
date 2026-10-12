/*!
 * \file relu_tiling.cpp
 * \brief Relu operator tiling implementation.
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t SYSTEM_WORKSPACE_SIZE = 0U;
constexpr int64_t MIN_ELEMENTS_PER_CORE = 1024;
constexpr int64_t FLOAT_BYTES = 4;
constexpr int64_t HALF_BYTES = 2;
constexpr int64_t QUEUE_BUFFER_COUNT = 4; // Two input and two output queue slots.

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "AIV core count must be positive"), return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "UB size must be positive"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspaceSize(gert::TilingContext* context)
{
    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    workspaceSizes[0] = SYSTEM_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "Failed to query platform information"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "Failed to set workspace size"), return ge::GRAPH_FAILED);

    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const int64_t totalNum = inputShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "Input tensor must not be empty"), return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dataType = inputDesc->GetDataType();
    OP_CHECK_IF(dataType != ge::DT_FLOAT && dataType != ge::DT_FLOAT16,
                OP_LOGE(context, "Only float32 and float16 are supported"), return ge::GRAPH_FAILED);

    int64_t usedCoreNum = CeilDiv(totalNum, MIN_ELEMENTS_PER_CORE);
    usedCoreNum = usedCoreNum < coreNum ? usedCoreNum : coreNum;
    usedCoreNum = usedCoreNum > 0 ? usedCoreNum : 1;

    const int64_t blockFactor = CeilDiv(totalNum, usedCoreNum);
    const int64_t typeBytes = dataType == ge::DT_FLOAT16 ? HALF_BYTES : FLOAT_BYTES;
    const int64_t alignNum = static_cast<int64_t>(GetUbBlockSize(context)) / typeBytes;
    OP_CHECK_IF(alignNum <= 0, OP_LOGE(context, "Invalid UB alignment"), return ge::GRAPH_FAILED);

    const int64_t maxUbFactor = FloorAlign(
        static_cast<int64_t>(ubSize) / (QUEUE_BUFFER_COUNT * typeBytes), alignNum);
    OP_CHECK_IF(maxUbFactor <= 0, OP_LOGE(context, "UB is too small for one aligned tile"),
                return ge::GRAPH_FAILED);
    const int64_t alignedBlockFactor = CeilAlign(blockFactor, alignNum);
    const int64_t ubFactor = alignedBlockFactor < maxUbFactor ? alignedBlockFactor : maxUbFactor;

    ReluTilingData* tilingData = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    tilingData->totalNum = totalNum;
    tilingData->blockFactor = blockFactor;
    tilingData->ubFactor = ubFactor;
    context->SetBlockDim(usedCoreNum);

    const uint64_t tilingKey = dataType == ge::DT_FLOAT16
        ? GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0)
        : GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
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
