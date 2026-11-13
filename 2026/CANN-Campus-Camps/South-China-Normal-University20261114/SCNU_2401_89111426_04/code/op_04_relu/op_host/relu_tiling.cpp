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
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TILE_ALIGN_BYTES = 256;
constexpr int64_t MIN_BYTES_PER_CORE = 4 * 1024;
constexpr int64_t MAX_TILE_BYTES = 60 * 1024;
constexpr int64_t MAX_UB_BUFFERS = 2;  // ReLU 原地计算，每个 tile 只占用一块 UB

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

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
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

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const auto* inputShape = context->GetInputShape(0);
    const auto* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    const auto dtype = inputDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT,
        OP_LOGE(context, "Relu only supports float16 and float32"), return ge::GRAPH_FAILED);
    const bool isHalf = dtype == ge::DT_FLOAT16;
    const int64_t elementBytes = isHalf ? 2 : 4;
    const int64_t totalNum = EnsureNotScalar(inputShape->GetStorageShape()).GetShapeSize();
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "Invalid input shape size"), return ge::GRAPH_FAILED);

    const int64_t workNum = std::max<int64_t>(totalNum, 1);
    const int64_t alignElements = TILE_ALIGN_BYTES / elementBytes;
    const int64_t minElementsPerCore = MIN_BYTES_PER_CORE / elementBytes;
    const int64_t targetCores = std::min(coreNum, CeilDiv(workNum, minElementsPerCore));
    const int64_t blockFactor = CeilAlign(CeilDiv(workNum, targetCores), alignElements);
    const int64_t blockDim = CeilDiv(workNum, blockFactor);

    // 为交替使用的两块 UB 预留容量。
    const int64_t ubCapacity = FloorAlign(static_cast<int64_t>(ubSize / (MAX_UB_BUFFERS * elementBytes)),
        alignElements);
    const int64_t ubFactor = std::min({blockFactor, ubCapacity, MAX_TILE_BYTES / elementBytes});
    OP_CHECK_IF(ubFactor <= 0, OP_LOGE(context, "Insufficient UB for Relu"), return ge::GRAPH_FAILED);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    context->SetTilingKey(isHalf ? GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0)
                                : GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
