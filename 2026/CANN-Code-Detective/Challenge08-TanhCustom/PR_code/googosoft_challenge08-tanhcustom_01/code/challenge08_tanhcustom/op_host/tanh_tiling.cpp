/*!
 * \file tanh_tiling.cpp
 * \brief Tanh 算子 Tiling 实现
 */

#include <algorithm>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/tanh_tiling_data.h"
#include "../op_kernel/tanh_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t VECTOR_BYTES = 256;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
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
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const auto dtype = inputDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT,
        OP_LOGE(context, "Tanh only supports float16 and float32"), return ge::GRAPH_FAILED);

    const int64_t typeSize = dtype == ge::DT_FLOAT16 ? 2 : 4;
    const int64_t alignNum = VECTOR_BYTES / typeSize;
    const int64_t totalNum = EnsureNotScalar(inputShape->GetStorageShape()).GetShapeSize();
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "Invalid input shape"), return ge::GRAPH_FAILED);

    const int64_t requestedCores = std::min(coreNum,
        std::max<int64_t>(1, CeilDiv(totalNum, MIN_SPLIT_THRESHOLD)));
    const int64_t blockFactor = std::max(alignNum,
        CeilAlign(CeilDiv(totalNum, requestedCores), alignNum));
    const int64_t blockDim = std::max<int64_t>(1, CeilDiv(totalNum, blockFactor));

    // 固定预留 Tanh 临时空间，余下 UB 用于两组双缓冲队列。
    uint32_t maxLiveNodeCount = 0;
    uint32_t extraBuf = 0;
    AscendC::GetTanhTmpBufferFactorSize(typeSize, maxLiveNodeCount, extraBuf);
    const int64_t availableBytes = static_cast<int64_t>(ubSize) - TANH_TMP_BUFFER_BYTES;
    const int64_t bytesPerElement = 2 * BUFFER_NUM * typeSize;
    OP_CHECK_IF(availableBytes < alignNum * bytesPerElement,
        OP_LOGE(context, "UB is too small for Tanh"), return ge::GRAPH_FAILED);
    const int64_t tmpBytesPerElement = static_cast<int64_t>(maxLiveNodeCount) * typeSize;
    const int64_t availableTmpBytes = TANH_TMP_BUFFER_BYTES - extraBuf;
    OP_CHECK_IF(availableTmpBytes < alignNum * tmpBytesPerElement,
        OP_LOGE(context, "Temporary buffer is too small for Tanh"), return ge::GRAPH_FAILED);
    const int64_t tmpTileLimit = tmpBytesPerElement == 0 ? TANH_MAX_TILE_LENGTH
        : FloorAlign(availableTmpBytes / tmpBytesPerElement, alignNum);
    const int64_t tileLimit = std::min(TANH_MAX_TILE_LENGTH, tmpTileLimit);
    const int64_t ubFactor = std::min(std::min(blockFactor, tileLimit),
        FloorAlign(availableBytes / bytesPerElement, alignNum));

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    context->SetBlockDim(blockDim);
    context->SetTilingKey(dtype == ge::DT_FLOAT16
        ? GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0)
        : GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_1));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTanh([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TanhCompileInfo {};

IMPL_OP_OPTILING(Tanh).Tiling(TanhTilingFunc).TilingParse<TanhCompileInfo>(TilingParseForTanh);

} // namespace optiling
