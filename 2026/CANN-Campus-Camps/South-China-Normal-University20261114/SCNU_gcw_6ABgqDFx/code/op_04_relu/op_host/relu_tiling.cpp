/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include <algorithm>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 4096;
constexpr int64_t ALIGN_BYTES = 256;
constexpr int64_t MAX_TILE_BYTES = 49152;
constexpr int64_t FULL_CORE_THRESHOLD = 2 * 1024 * 1024;
constexpr int64_t BUFFER_COUNT = 4;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum,
    int64_t& aicNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    aicNum = ascendcPlatform.GetCoreNumAic();
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
    int64_t aicNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum, aicNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    const ge::DataType dtype = inputDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16,
        OP_LOGE(context, "Unsupported dtype"), return ge::GRAPH_FAILED);

    const int64_t typeSize = dtype == ge::DT_FLOAT16 ? 2 : TYPE_SIZE;
    const int64_t alignNum = ALIGN_BYTES / typeSize;
    const int64_t totalNum = EnsureNotScalar(inputShape->GetStorageShape()).GetShapeSize();
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "Unresolved shape"), return ge::GRAPH_FAILED);

    if (totalNum == 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
    } else {
        const int64_t waveCoreNum = aicNum > 0 ? std::min<int64_t>(coreNum, aicNum) : coreNum;
        const int64_t maxCoreNum = totalNum >= FULL_CORE_THRESHOLD ? coreNum : waveCoreNum;
        const int64_t usedCoreNum = std::min(
            maxCoreNum, std::max<int64_t>(1, totalNum / MIN_SPLIT_THRESHOLD));
        const int64_t perCoreNum = CeilDiv(totalNum, usedCoreNum);
        const int64_t blockFactor = std::max<int64_t>(alignNum, CeilAlign(perCoreNum, alignNum));
        const int64_t maxTileElems = std::min<int64_t>(
            MAX_TILE_BYTES / typeSize,
            FloorAlign(static_cast<int64_t>(ubSize) / (BUFFER_COUNT * typeSize), alignNum));
        const int64_t ubFactor = std::min<int64_t>(blockFactor, maxTileElems);
        OP_CHECK_IF(ubFactor < alignNum, OP_LOGE(context, "Insufficient UB"), return ge::GRAPH_FAILED);

        tiling->totalNum = totalNum;
        tiling->blockFactor = blockFactor;
        tiling->ubFactor = ubFactor;
        context->SetBlockDim(CeilDiv(totalNum, blockFactor));
    }

    uint64_t tilingKey;
    if (dtype == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    }
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

