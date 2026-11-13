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

#include <algorithm>

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t COPY_ALIGN_BYTES = 32;
constexpr int64_t TILE_ALIGN_BYTES = 256;
constexpr int64_t MIN_BYTES_PER_CORE = 4096;
constexpr int64_t QUEUE_BUFFER_COUNT = 4;  // 输入和输出队列各有两个缓冲。
constexpr int64_t FP32_BUFFER_COUNT = 4;   // 对应 Kernel 中的四块临时缓冲。

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

    const auto* inputDesc = context->GetInputDesc(0);
    const auto* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const auto dtype = inputDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT,
        OP_LOGE(context, "Tanh expects float16 or float32"), return ge::GRAPH_FAILED);

    const int64_t elemBytes = dtype == ge::DT_FLOAT16 ? 2 : 4;
    const int64_t totalNum = EnsureNotScalar(inputShape->GetStorageShape()).GetShapeSize();
    const int64_t copyAlign = COPY_ALIGN_BYTES / elemBytes;
    OP_CHECK_IF(totalNum <= 0 || totalNum % copyAlign != 0,
        OP_LOGE(context, "Tanh expects a nonempty 32-byte-aligned input"), return ge::GRAPH_FAILED);

    const int64_t tileAlign = TILE_ALIGN_BYTES / elemBytes;
    const int64_t requestedCores = std::max<int64_t>(1, totalNum / (MIN_BYTES_PER_CORE / elemBytes));
    const int64_t usedCores = std::min(coreNum, requestedCores);
    const int64_t blockFactor = CeilAlign(CeilDiv(totalNum, usedCores), tileAlign);

    const int64_t ubBytesPerElem = QUEUE_BUFFER_COUNT * elemBytes + FP32_BUFFER_COUNT * 4;
    const int64_t maxUbFactor = FloorAlign(static_cast<int64_t>(ubSize) / ubBytesPerElem, tileAlign);
    OP_CHECK_IF(maxUbFactor <= 0, OP_LOGE(context, "UB is too small for Tanh"), return ge::GRAPH_FAILED);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = std::min(blockFactor, maxUbFactor);
    context->SetBlockDim(CeilDiv(totalNum, blockFactor));
    context->SetTilingKey(dtype == ge::DT_FLOAT16 ? GET_TPL_TILING_KEY(TANH_TPL_SCH_MODE_0)
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
