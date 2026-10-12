/*!
 * \file tanh_tiling.cpp
 * \brief Tanh 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static int64_t GetDataTypeSize(const ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:
            return 4;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return 2;
        default:
            return 4;
    }
}

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

static ge::graphStatus TanhTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    TanhTilingData* tiling = context->GetTilingData<TanhTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    int64_t typeSize = GetDataTypeSize(dtype);

    // 注意：TilingContext 返回的是 StorageShape*
    const gert::StorageShape* inputStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputStorageShape);
    const gert::Shape& inputShape = EnsureNotScalar(inputStorageShape->GetShape());
    int64_t totalNum = inputShape.GetShapeSize();
    if (totalNum < 0) {
        totalNum = 0;
    }

    int64_t alignNum = 32 / typeSize;
    if (alignNum <= 0) {
        alignNum = 1;
    }

    int64_t blockFactor = CeilDiv(totalNum, coreNum);
    blockFactor = CeilAlign(blockFactor, alignNum);
    if (blockFactor <= 0) {
        blockFactor = alignNum;
    }

    int64_t usedCoreNum = CeilDiv(totalNum, blockFactor);
    if (usedCoreNum <= 0) {
        usedCoreNum = 1;
    }
    if (usedCoreNum > coreNum) {
        usedCoreNum = coreNum;
    }

    int64_t ubCapacity = static_cast<int64_t>(ubSize) / (BUFFER_NUM * typeSize * 2);
    int64_t ubFactor = FloorAlign(ubCapacity, alignNum);
    if (ubFactor <= 0) {
        ubFactor = alignNum;
    }
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(usedCoreNum);

    uint64_t tilingKey;
    if (dtype == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTanh([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TanhCompileInfo {};

IMPL_OP_OPTILING(Tanh).Tiling(TanhTilingFunc).TilingParse<TanhCompileInfo>(TilingParseForTanh);

} // namespace optiling